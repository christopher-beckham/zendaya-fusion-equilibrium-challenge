"""MLP trunk, conv decoder for ψ, and a small head for q95 and βN.

Upsample-then-conv, not ConvTranspose: a checkerboard would show up in slope-based
quantities such as li even when the pixel values look fine.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import GRID, INPUT_CLIP, N_COIL, N_IMAGE, N_PSI, broadcast_coil_stats, load_rz, tail_widths


def _coord_map(r: np.ndarray, z: np.ndarray) -> torch.Tensor:
    """(2, 65, 65) with R and Z each scaled to [-1, 1]. Channel 0 is R, channel 1 is Z."""
    def _unit(v):
        return 2.0 * (v - v.min()) / (v.max() - v.min()) - 1.0
    rr, zz = np.meshgrid(_unit(r), _unit(z), indexing="xy")
    # meshgrid(..., indexing="xy") gives shape (len(z), len(r)): axis0 = Z, axis1 = R.
    return torch.from_numpy(np.stack([rr, zz], axis=0).astype(np.float32))


class ResBlock(nn.Module):
    """Pre-norm residual MLP block. Width is unchanged."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(self.norm(x)))
        return x + self.fc2(h)


def _plasma_coord_map(r: np.ndarray, z: np.ndarray, r0: float, z0: float, a: float) -> torch.Tensor:
    """(3, 65, 65): (R − R0)/a, (Z − Z0)/a and R/R0. The same formula on every machine, so a value means the
    same place relative to the plasma; a larger grid shows up as values beyond the training range."""
    rr, zz = np.meshgrid(r, z, indexing="xy")
    return torch.from_numpy(np.stack([(rr - r0) / a, (zz - z0) / a, rr / r0], axis=0).astype(np.float32))


# Plasma prior for --output-prior: Ip spread over an ellipse of half-width PRIOR_HALF·a and elongation PRIOR_KAPPA,
# centred at Z0 and at R = R0 + a·R_ax_th (the Thomson pressure-peak feature), tabulated over R_ax_th. The current
# density falls as (1 − ρ²)^PRIOR_PROFILE_POWER from the centre to the edge, carried by a PRIOR_FILAMENTS² grid of
# filaments clipped to the ellipse. A dense, peaked distribution gives one smooth flux extreme at the centre; 21 equal
# filaments gave a spiky field whose extremes pinned the derived magnetic axis to the filament positions.
PRIOR_RAX = np.linspace(-0.3, 0.8, 111)
PRIOR_HALF, PRIOR_KAPPA, PRIOR_DEFAULT_RAX = 0.6, 1.6, 0.1
PRIOR_FILAMENTS, PRIOR_PROFILE_POWER = 41, 1.0


def _picture_sign() -> float:
    """The global sign baseline_pictures.npz applied to the filament convention, so the prior matches the pictures."""
    from greens import filament_psi, load_baseline
    from pathlib import Path
    b = load_baseline(Path(__file__).resolve().parent / "baseline_pictures.npz")
    r, z = load_rz()
    rr, zz = np.meshgrid(r, z, indexing="xy")
    ref = filament_psi(rr.ravel(), zz.ravel(), b["centroid_r"], b["centroid_z"])
    return float(np.sign(ref @ b["ring"].astype(np.float64)))


def plasma_prior_table(r: np.ndarray, z: np.ndarray, r0: float, z0: float, a: float) -> np.ndarray:
    """(len(PRIOR_RAX), 4225) flux per kA of plasma current on the (r, z) grid, one row per axis position."""
    from greens import filament_psi
    rr, zz = np.meshgrid(np.asarray(r, np.float64), np.asarray(z, np.float64), indexing="xy")
    u = (np.arange(PRIOR_FILAMENTS) + 0.5) / PRIOR_FILAMENTS * 2.0 - 1.0
    fu, fw = np.meshgrid(u, u)
    rho2 = fu ** 2 + fw ** 2
    ins = rho2 < 1.0
    weight = (1.0 - rho2[ins]) ** PRIOR_PROFILE_POWER
    weight = weight / weight.sum()
    du, dv = fu[ins] * PRIOR_HALF * a, fw[ins] * PRIOR_HALF * a * PRIOR_KAPPA
    R, Z = rr.ravel(), zz.ravel()
    out = np.empty((len(PRIOR_RAX), R.size))
    for i, rax in enumerate(PRIOR_RAX):
        centre = r0 + a * rax
        acc = np.zeros(R.size)
        for w, x, y in zip(weight, du, dv):
            acc += w * filament_psi(R, Z, centre + x, z0 + y)
        out[i] = acc
    return _picture_sign() * out


def prior_from_table(table: torch.Tensor, ip_ka: torch.Tensor, rax_th: torch.Tensor) -> torch.Tensor:
    """(B, 4225) prior flux in Wb/rad from a plasma_prior_table on some grid: interpolated at R_ax_th (missing ->
    PRIOR_DEFAULT_RAX), times Ip."""
    rax = torch.nan_to_num(rax_th.float(), nan=PRIOR_DEFAULT_RAX)
    lo, hi, n = float(PRIOR_RAX[0]), float(PRIOR_RAX[-1]), len(PRIOR_RAX)
    pos = ((rax.clamp(lo, hi) - lo) / (hi - lo) * (n - 1)).clamp(0, n - 1 - 1e-6)
    i0 = pos.floor().long()
    w = (pos - i0.float()).unsqueeze(1)
    unit = (1 - w) * table[i0] + w * table[i0 + 1]
    return unit * ip_ka.float().view(-1, 1)


def flux_scale(ip_ka: torch.Tensor, r0: float, floor_ka: float = 100.0) -> torch.Tensor:
    """μ0 R0 |Ip| / (2π) in Wb/rad, with |Ip| floored at `floor_ka`. The plasma's own flux scale, from inputs."""
    return (4.0e-7 * np.pi) * r0 * ip_ka.float().abs().clamp_min(floor_ka) * 1000.0 / (2.0 * np.pi)


class UpStage(nn.Module):
    """Nearest ×2, append the coordinate channels and the coil picture, conv 3×3, GroupNorm, SiLU."""

    def __init__(self, cin: int, cout: int, groups: int = 8, n_coord: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(cin + n_coord + 1, cout, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, coord: torch.Tensor, picture: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # coord is (2, H, W). interpolate wants a batch dimension; a 3-D tensor would be read as 1-D.
        grid = coord.unsqueeze(0) if coord.dim() == 3 else coord
        c = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=True)
        pic = F.interpolate(picture, size=x.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, c.expand(x.shape[0], -1, -1, -1), pic], dim=1)
        return self.act(self.norm(self.conv(x)))


class ScalarHead(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@contextlib.contextmanager
def strict_fp32(device_type: str):
    """Autocast off and TF32 off.

    cuDNN runs float32 convolutions in TF32 by default, which rounds inputs to about three
    significant digits, the same order as bf16. The axis and the boundary sit where ψ is
    flat, so that rounding moves them. Finite differences amplify it too.
    """
    cudnn = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = matmul


def _f32(value, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32).contiguous()
    if tuple(tensor.shape) != shape:
        raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    return tensor


class Net(nn.Module):
    """Trunk -> ψ decoder and scalar head.

    The trunk is Linear then `n_blocks` residual MLP blocks. The decoder projects to
    `seed_channels`×8×8 (256 by default) and runs three upsample stages, 8→16→32→64,
    channels seed→128→64→32. That projection holds most of the parameters, so
    `seed_channels` shrinks it without touching the trunk.
    Each stage sees the coil picture, downsampled, beside R and Z. A bilinear resize
    to 65×65 and a final conv, which also sees the full picture, emit the flux in
    normalised units. The picture is an input only. It is not added to the output.

    The final conv and the scalar head always run in float32 without TF32, even under
    bf16 autocast. bf16 output rounding moved the derived axis and boundary enough to
    cost about 0.004 S, and was about 70% of the late-training gradient loss.

    `picture` is the physical vacuum flux already summed for this batch, shape
    (B, 4225). The network does not know which coils produced it.

    `decoder_norm` sets what the decoder sees and emits:
      "target"    the picture enters as (picture − pixel_mean) / psi_std, the coordinates are R and Z scaled
                  to [−1, 1] over the grid, and the output is ψ in those same units. pixel_mean and psi_std
                  are statistics of the DIII-D training targets.
      "physical"  the picture enters as ψ_ext / s, the coordinates are (R − R0)/a, (Z − Z0)/a and R/R0,
                  and the network's output o gives ψ = s · o, where s = μ0 R0 |Ip| / (2π) per frame
                  (flux_scale). Nothing the network sees or emits depends on a training statistic, so the
                  same maps apply to another machine given its R0, a, Z0 and Ip. forward() still returns
                  (ψ − pixel_mean) / psi_std so the losses and callers are unchanged; that conversion is
                  exactly undone by the callers.
    """

    def __init__(self, n_in: int, width: int = 512, n_blocks: int = 3,
                 scalar_width: int = 128, scalar_depth: int = 2, *,
                 seed_channels: int = 256, pixel_mean=None, psi_std=None,
                 decoder_norm: str = "target", geometry: tuple[float, float, float] | None = None,
                 output_prior: bool = False):
        super().__init__()
        if pixel_mean is None or psi_std is None:
            raise ValueError("pixel_mean and psi_std are required")
        psi_std_t = torch.as_tensor(float(psi_std), dtype=torch.float32)
        if float(psi_std_t) <= 0:
            raise ValueError("psi_std must be positive")
        if decoder_norm not in ("target", "physical"):
            raise ValueError(f"decoder_norm must be 'target' or 'physical', got {decoder_norm!r}")
        self.decoder_norm = decoder_norm
        r, z = load_rz()
        if decoder_norm == "physical":
            if geometry is None:
                raise ValueError("decoder_norm 'physical' needs geometry (R0, Z0, a)")
            self.r0, z0, a = (float(v) for v in geometry)
            coord = _plasma_coord_map(r, z, self.r0, z0, a)
        else:
            self.r0 = None
            coord = _coord_map(r, z)
        n_coord = coord.shape[0]
        self.register_buffer("coord", coord, persistent=True)
        self.output_prior = bool(output_prior)
        if self.output_prior:
            if decoder_norm != "physical":
                raise ValueError("output_prior needs decoder_norm 'physical'")
            table = plasma_prior_table(r, z, self.r0, z0, a)
            self.register_buffer("prior_table", torch.from_numpy(table.astype(np.float32)), persistent=False)
        self.register_buffer("r_coord", torch.from_numpy(r.astype(np.float32)), persistent=True)
        self.register_buffer("z_coord", torch.from_numpy(z.astype(np.float32)), persistent=True)
        self.register_buffer("pixel_mean", _f32(pixel_mean, (N_PSI,)))
        self.register_buffer("psi_std", psi_std_t)
        self.in_proj = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(n_blocks)])
        self.seed_channels = int(seed_channels)
        self.to_grid = nn.Linear(width, self.seed_channels * 8 * 8)
        self.stages = nn.ModuleList([
            UpStage(self.seed_channels, 128, n_coord=n_coord),
            UpStage(128, 64, n_coord=n_coord),
            UpStage(64, 32, n_coord=n_coord),
        ])
        self.final = nn.Conv2d(32 + n_coord + 1, 1, kernel_size=3, padding=1)
        self.scalar = ScalarHead(width, scalar_width, scalar_depth)

    def set_machine(self, r: np.ndarray, z: np.ndarray, geometry: tuple[float, float, float]) -> None:
        """Point a 'physical' decoder at another grid or machine: its 65×65 grid (R, Z in metres) and (R0, Z0, a).

        The weights are unchanged. The coordinate channels become that grid in plasma units, R0 sets the flux scale
        μ0 R0 Ip / 2π of the picture and the output, and with output_prior the prior is rebuilt on that grid.
        r_coord and z_coord are only used by the training losses and are left as they are.
        """
        if self.decoder_norm != "physical":
            raise ValueError("only a 'physical' decoder can change machine")
        r0, z0, a = (float(v) for v in geometry)
        r = np.asarray(r, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        self.coord = _plasma_coord_map(r, z, r0, z0, a).to(device=self.coord.device, dtype=self.coord.dtype)
        self.r0 = r0
        if self.output_prior:
            table = plasma_prior_table(r, z, r0, z0, a)
            self.prior_table = torch.from_numpy(table.astype(np.float32)).to(self.coord.device)

    def plasma_prior(self, ip_ka: torch.Tensor, rax_th: torch.Tensor) -> torch.Tensor:
        """(B, 4225) prior flux in Wb/rad: the table interpolated at R_ax_th (missing -> PRIOR_DEFAULT_RAX), times Ip."""
        return prior_from_table(self.prior_table, ip_ka, rax_th)

    def forward(self, coils: torch.Tensor, picture: torch.Tensor,
                ip_ka: torch.Tensor | None = None,
                rax_th: torch.Tensor | None = None,
                coord: torch.Tensor | None = None,
                prior: torch.Tensor | None = None,
                r0: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """coils (B, n_in), picture (B, 4225) physical flux -> psi, scalars, both normalised.

        With decoder_norm "physical", `ip_ka` (B,) is the plasma current in kA, which sets the flux scale.
        With output_prior, ψ = picture + plasma prior + network output × flux scale, and `rax_th` (B,) is the
        unscaled R_ax_th feature that places the prior.
        `coord` (B, n_coord, 65, 65) and `prior` (B, 4225, Wb/rad) replace the net's own coordinate channels and
        prior for examples drawn on other grids (zoom augmentation, zoom.py). `r0` replaces the net's R0 in the flux
        scale, for examples from another machine (physics_loss.MastInputs).
        """
        h = self.blocks(self.in_proj(coils))
        with torch.autocast(device_type=coils.device.type, enabled=False):
            pic = picture.float().reshape(-1, N_PSI)
            if self.decoder_norm == "physical":
                if ip_ka is None:
                    raise ValueError("decoder_norm 'physical' needs ip_ka")
                scale = flux_scale(ip_ka, self.r0 if r0 is None else r0).view(-1, 1)
                pic_n = pic / scale
            else:
                pic_n = (pic - self.pixel_mean) / self.psi_std
            spatial = pic_n.view(-1, 1, GRID, GRID)
        coord_in = self.coord if coord is None else coord
        x = self.to_grid(h).view(-1, self.seed_channels, 8, 8)
        for stage in self.stages:
            x = stage(x, coord_in, spatial)
        x = F.interpolate(x, size=(GRID, GRID), mode="bilinear", align_corners=True)
        coord = coord_in.unsqueeze(0).expand(x.shape[0], -1, -1, -1) if coord_in.dim() == 3 else coord_in
        with strict_fp32(coils.device.type):
            psi = self.final(torch.cat([x.float(), coord.float(), spatial.float()], dim=1)).view(-1, N_PSI)
            if self.decoder_norm == "physical":
                # ψ = s·o in Wb/rad, then the same normalised units as the target path. Callers undo this exactly.
                phys = psi * scale
                if self.output_prior:
                    if prior is None:
                        if rax_th is None:
                            raise ValueError("output_prior needs rax_th")
                        prior = self.plasma_prior(ip_ka, rax_th)
                    phys = phys + pic + prior.float()
                psi = (phys - self.pixel_mean) / self.psi_std
            scalars = self.scalar(h.float())
        return psi, scalars


def flux_gradients(psi_flat: torch.Tensor, z_coord: torch.Tensor, r_coord: torch.Tensor):
    """dψ/dR and dψ/dZ for psi (B, 4225) or (B, 65, 65). Central differences, physical spacing.

    Matches `np.gradient(psi, Z, R)` in fusion_scoring/derive.py: axis0 is Z, axis1 is R.
    """
    psi = psi_flat.view(-1, GRID, GRID)
    dpsi_dz, dpsi_dr = torch.gradient(psi, spacing=(z_coord, r_coord), dim=(-2, -1), edge_order=1)
    return dpsi_dr, dpsi_dz


def load_checkpoint(path, device: torch.device):
    """Return (net, checkpoint) on `device`. The net is in eval mode.

    The coil picture is part of the architecture. Checkpoints from before that
    change do not load, and neither do checkpoints that added the picture to the output.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("picture_added", True):
        raise ValueError(f"{path}: trained with the coil picture added to the output, which was removed")
    a = ckpt["args"]
    norm = ckpt["norm"]
    n_in = int(a["n_in"]) if "n_in" in a else N_COIL + N_IMAGE
    net = Net(
        n_in=n_in,
        width=int(a["width"]),
        scalar_width=int(a["scalar_width"]),
        scalar_depth=int(a["scalar_depth"]),
        seed_channels=int(a.get("seed_channels", 256)),
        pixel_mean=norm["pixel_mean"],
        psi_std=float(norm["psi_std"]),
        decoder_norm=str(a.get("decoder_norm", "target")),
        geometry=norm.get("decoder_geometry"),
        output_prior=bool(a.get("output_prior", False)),
    )
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    return net, ckpt


def unpack_inside(packed: torch.Tensor) -> torch.Tensor:
    """(B, 529) uint8 from np.packbits -> (B, 65, 65) float 0/1. np.packbits puts the first cell in bit 7."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :N_PSI].view(-1, GRID, GRID).float()


def li_error(pred_n: torch.Tensor, true_n: torch.Tensor, pixel_mean: torch.Tensor, psi_std: torch.Tensor,
                    contour: torch.Tensor, inside: torch.Tensor,
                    z_coord: torch.Tensor, r_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Change in the scorer's li from true ψ to predicted ψ, with the true boundary held fixed.

    Same formula as fusion_scoring/derive.py:internal_inductance:
        li = A / (G / lp)²,  A = <Bp²> over cells inside the boundary with dV = 2π R dR dZ,
        G = Σ over boundary segments of the mean Bp at the two ends × segment length, lp = perimeter,
        Bp = |∇ψ| / R.
    li_pred is computed exactly on the true boundary:
        δli = li_true · ((A_pred / A_true) / max(G_pred / G_true, 0.5)² − 1).
    The floor only matters for a wildly wrong ψ early in training, where G_pred could approach 0.
    On real predictions this matched the scorer's li error with correlation 0.975 and scale 0.97.
    A first-order expansion did worse: the boundary passes through the X-point where |∇ψ| ≈ 0, and
    linearising |∇ψ| there overstated the error 1.6×. Holding the boundary fixed drops the
    boundary-motion part of the error.

    pred_n, true_n (B, 4225) normalised ψ. contour (B, K, 2) true boundary (R, Z) in metres. inside
    (B, 65, 65) 0/1. Returns δli and the true li, both (B,), in float32.
    """
    with torch.autocast(device_type=pred_n.device.type, enabled=False):
        rr = r_coord.view(1, 1, -1)
        cell = 2.0 * torch.pi * rr * (r_coord[1] - r_coord[0]) * (z_coord[1] - z_coord[0])
        w = inside * cell
        vol = w.sum(dim=(1, 2))
        # Bilinear sampling at the boundary, as derive._bilinear does. grid_sample wants x = R, y = Z in [-1, 1].
        gx = 2.0 * (contour[..., 0] - r_coord[0]) / (r_coord[-1] - r_coord[0]) - 1.0
        gy = 2.0 * (contour[..., 1] - z_coord[0]) / (z_coord[-1] - z_coord[0]) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)
        seg = (torch.roll(contour, -1, dims=1) - contour).norm(dim=-1)
        perim = seg.sum(dim=1)

        def pieces(psi_n):
            gr, gz = flux_gradients(psi_n.float() * psi_std + pixel_mean, z_coord, r_coord)
            g2 = (gr * gr + gz * gz) / (rr * rr)
            a = (g2 * w).sum(dim=(1, 2)) / vol
            # The small floor keeps the sqrt differentiable where ∇ψ is exactly 0.
            bp = torch.sqrt(g2 + 1e-20)
            on_c = F.grid_sample(bp.unsqueeze(1), grid, mode="bilinear", align_corners=True).view(len(psi_n), -1)
            g_line = (0.5 * (on_c + torch.roll(on_c, -1, dims=1)) * seg).sum(dim=1)
            return a, g_line

        a_t, g_t = pieces(true_n)
        a_p, g_p = pieces(pred_n)
        li = a_t / (g_t / perim) ** 2
        d_li = li * ((a_p / a_t) / (g_p / g_t).clamp_min(0.5) ** 2 - 1.0)
    return d_li, li


# Half-width, in grid cells, of the square searched for the X-point and averaged by the X-point term.
XPOINT_HALF = 2
# A frame whose boundary level is further than this, in metres of boundary shift, from every saddle
# node near the X-point is taken to be limited by the envelope rather than by a saddle, and masked.
ENVELOPE_RESID_M = 1e-3
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def _contour_grid(contour: torch.Tensor, z_coord: torch.Tensor, r_coord: torch.Tensor) -> torch.Tensor:
    """(B, K, 2) physical (R, Z) -> (B, 1, K, 2) grid_sample coordinates, x = R and y = Z in [-1, 1]."""
    gx = 2.0 * (contour[..., 0] - r_coord[0]) / (r_coord[-1] - r_coord[0]) - 1.0
    gy = 2.0 * (contour[..., 1] - z_coord[0]) / (z_coord[-1] - z_coord[0]) - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(1)


def _on_contour(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of (B, 65, 65) at the boundary points, as derive._bilinear does. Returns (B, K)."""
    out = F.grid_sample(field.unsqueeze(1), grid, mode="bilinear", align_corners=True)
    return out.view(field.shape[0], -1)


def saddle_nodes(psi: torch.Tensor) -> torch.Tensor:
    """(B, 65, 65) -> bool, True at interior nodes whose 8 neighbours change sign against it 4+ times.

    Going round the ring of neighbours, ψ_neighbour − ψ_node alternates above, below, above, below at
    a saddle, and changes sign at most twice at a regular point or an extremum.
    """
    pad = F.pad(psi.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    signs = torch.stack([torch.sign(pad[:, 1 + a:1 + a + GRID, 1 + b:1 + b + GRID] - psi) for a, b in _RING])
    changes = (signs != torch.roll(signs, -1, dims=0)).sum(dim=0)
    out = changes >= 4
    out[:, 0, :] = False
    out[:, -1, :] = False
    out[:, :, 0] = False
    out[:, :, -1] = False
    return out


@torch.no_grad()
def find_xpoint(psi: torch.Tensor, grad_c: torch.Tensor, level_c: torch.Tensor, contour: torch.Tensor,
                z_coord: torch.Tensor, r_coord: torch.Tensor, half: int = XPOINT_HALF
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """The grid node whose true ψ sets the scorer's boundary level, from the true ψ and boundary.

    The scorer's bisection settles on a level equal to a grid-node value (99.3% of val frames) at the
    saddle the boundary passes through. Candidates are the saddle nodes within `half` cells of the
    boundary point with the smallest |∇ψ|. The one whose ψ is closest to the boundary level (median
    of ψ sampled on the boundary) is taken. On val frames this found the scorer's node 93% of the
    time; the smallest-|∇ψ| node alone found it 70% of the time. Near the X-point several nodes
    sit within the level estimate's error, and only the saddle test separates them.

    A frame with no saddle candidate, or whose closest candidate is more than ENVELOPE_RESID_M of
    boundary shift from the level, is envelope-limited: the boundary stops at the machine envelope
    and no single node sets the level. That test caught 47 of 51 such val frames with no false
    positives in 7796; the four it missed had their level within 0.4 mm of a saddle anyway.

    psi (B, 65, 65) physical true ψ. grad_c, level_c (B, K) |∇ψ| and ψ sampled on the boundary.
    Returns the flat node index (B,) and ok (B,) bool.
    """
    b = psi.shape[0]
    rows = torch.arange(b, device=psi.device)
    i0 = grad_c.argmin(dim=1)
    point = contour[rows, i0]
    cr = torch.round((point[:, 0] - r_coord[0]) / (r_coord[1] - r_coord[0])).long().clamp(0, GRID - 1)
    cz = torch.round((point[:, 1] - z_coord[0]) / (z_coord[1] - z_coord[0])).long().clamp(0, GRID - 1)
    idx = torch.arange(GRID, device=psi.device)
    near_z = (idx.view(1, -1) - cz.view(-1, 1)).abs() <= half
    near_r = (idx.view(1, -1) - cr.view(-1, 1)).abs() <= half
    cand = saddle_nodes(psi) & near_z.unsqueeze(2) & near_r.unsqueeze(1)
    level = level_c.median(dim=1).values
    resid = torch.where(cand, (psi - level.view(-1, 1, 1)).abs(), torch.full_like(psi, float("inf")))
    best, node = resid.view(b, -1).min(dim=1)
    ok = torch.isfinite(best) & (best <= ENVELOPE_RESID_M * grad_c.median(dim=1).values)
    return node, ok


def boundary_error(pred_n: torch.Tensor, true_n: torch.Tensor, pixel_mean: torch.Tensor, psi_std: torch.Tensor,
                   contour: torch.Tensor, z_coord: torch.Tensor, r_coord: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order motion of the scorer's boundary from true ψ to predicted ψ, and the flux error at the X-point.

    The predicted boundary is the contour at the predicted X-point level, which moves by δψ_X, the flux
    error at the level-setting node (find_xpoint). A boundary point x_i then moves along its outward
    normal by
        δn_i = −σ (δψ(x_i) − δψ_X) / max(|∇ψ_true(x_i)|, cap),
    σ = +1 when ψ rises outward. Uncapped, 1/|∇ψ| blows up at the X-point and overstated the shift
    4–9× there. cap is the frame's 5th percentile of |∇ψ_true| on the boundary. A uniform error in ψ
    moves nothing, so only error relative to the X-point counts, as for the scorer.

    Returned per frame, in float32:
        band    mean of δn_i² over the boundary (m²). Points are equally spaced, so this is the arc mean.
        d_vol   ∮ 2π R δn dl (m³), the change in the scorer's volume.
        xpoint  mean of (δψ_j − δψ_X)² over the nodes within XPOINT_HALF cells of the X-point, over cap²
                (m²). An error there moves the corner where the boundary turns, which the band term
                sees only through the cap. Dividing by cap² puts it in the band's units.
        ok      False for envelope-limited frames. Use only ok frames.

    pred_n, true_n (B, 4225) normalised ψ. contour (B, K, 2) true boundary (R, Z) in metres, equally
    spaced by arc length.
    """
    with torch.autocast(device_type=pred_n.device.type, enabled=False):
        b = pred_n.shape[0]
        true = (true_n.float() * psi_std + pixel_mean).view(-1, GRID, GRID)
        d_psi = ((pred_n.float() - true_n.float()) * psi_std).view(-1, GRID, GRID)
        grid = _contour_grid(contour, z_coord, r_coord)
        with torch.no_grad():
            gr, gz = flux_gradients(true, z_coord, r_coord)
            gr_c = _on_contour(gr, grid)
            gz_c = _on_contour(gz, grid)
            # |∇ψ| is sampled as a grid field, as derive.py samples Bp, not rebuilt from sampled components.
            grad_c = _on_contour(torch.sqrt(gr * gr + gz * gz), grid)
            node, ok = find_xpoint(true, grad_c, _on_contour(true, grid), contour, z_coord, r_coord)
            # The floor only matters for a degenerate frame where the cap itself is 0.
            cap = torch.quantile(grad_c, 0.05, dim=1, keepdim=True).clamp_min(1e-6)
            denom = torch.maximum(grad_c, cap)
            # Outward normal from the central tangent. Its sign follows the boundary's orientation.
            tangent = torch.roll(contour, -1, dims=1) - torch.roll(contour, 1, dims=1)
            normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
            area = (contour[..., 0] * torch.roll(contour[..., 1], -1, dims=1)
                    - torch.roll(contour[..., 0], -1, dims=1) * contour[..., 1]).sum(dim=1)
            normal = normal * torch.sign(area).view(-1, 1, 1)
            sigma = torch.sign((gr_c * normal[..., 0] + gz_c * normal[..., 1]).sum(dim=1, keepdim=True))
            seg = (torch.roll(contour, -1, dims=1) - contour).norm(dim=-1)
            dl = 0.5 * (seg + torch.roll(seg, 1, dims=1))
            # The square around the X-point node. Clamping repeats edge nodes, which only matters
            # within two cells of the grid edge, where no diverted X-point sits.
            offs = torch.arange(-XPOINT_HALF, XPOINT_HALF + 1, device=pred_n.device)
            wz = (node // GRID).view(-1, 1, 1) + offs.view(1, -1, 1)
            wr = (node % GRID).view(-1, 1, 1) + offs.view(1, 1, -1)
            window = (wz.clamp(0, GRID - 1) * GRID + wr.clamp(0, GRID - 1)).view(b, -1)
        flat = d_psi.view(b, -1)
        d_x = flat.gather(1, node.view(-1, 1))
        shift = -sigma * (_on_contour(d_psi, grid) - d_x) / denom
        band = shift.square().mean(dim=1)
        d_vol = 2.0 * torch.pi * (contour[..., 0] * shift * dl).sum(dim=1)
        xpoint = (flat.gather(1, window) - d_x).square().mean(dim=1) / cap.view(-1).square()
    return band, d_vol, xpoint, ok


def grad_mse(pred_flat: torch.Tensor, true_flat: torch.Tensor,
             z_coord: torch.Tensor, r_coord: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the two slope components. Both maps are in the same units."""
    dpr, dpz = flux_gradients(pred_flat, z_coord, r_coord)
    dtr, dtz = flux_gradients(true_flat, z_coord, r_coord)
    return 0.5 * ((dpr - dtr).square().mean() + (dpz - dtz).square().mean())


def precision_context(device_type: str, use_amp: bool):
    """bf16 autocast, or strict float32 for a run trained with --no-amp."""
    if use_amp:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return strict_fp32(device_type)


def _clip_tail(values: torch.Tensor, norm: dict) -> torch.Tensor:
    """Clip the standardised image columns, and the safe columns when they are present.

    In the --robust-inputs layout a missing value arrives as NaN and becomes 0, the train median.
    """
    values = values.clone()
    if norm.get("feature_set"):
        # As train.load_split: missing values become 0, the train median, and every column is clipped.
        return torch.nan_to_num(values, nan=0.0).clamp(-INPUT_CLIP, INPUT_CLIP)
    if norm.get("robust_inputs", False):
        values = torch.nan_to_num(values, nan=0.0)
    n_safe, n_image = tail_widths(norm)
    values[:, -n_image:] = values[:, -n_image:].clamp(-INPUT_CLIP, INPUT_CLIP)
    if n_safe:
        sl = slice(-(n_safe + n_image), -n_image)
        values[:, sl] = values[:, sl].clamp(-INPUT_CLIP, INPUT_CLIP)
    return values


@torch.no_grad()
def predict_raw(net: Net, coils: np.ndarray, picture: np.ndarray, norm: dict, device: torch.device,
                batch_size: int = 2048, ip_ka: np.ndarray | None = None,
                rax_th: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirrored coils (N, n_in) and physical coil picture (N, 4225) -> raw ψ, q95, βN.

    A net with decoder_norm "physical" also needs the mirrored plasma current `ip_ka` (N,) in kA.

    A window is 21 columns per stamp. Each stamp is scaled with the same 21 mean/std
    pairs. The image columns, and the safe columns when present, use their own
    mean and std and are clipped to ±5 after standardisation. The coil columns are
    not clipped. With --robust-inputs the pairs are median and IQR/1.349. ψ is float32 here. Submissions keep it float32 (predict.py).
    """
    if net.decoder_norm == "physical" and (ip_ka is None or len(ip_ka) != len(coils)):
        raise ValueError("this net needs the plasma current for every row")
    if len(picture) != len(coils) or picture.shape[1] != N_PSI:
        raise ValueError(f"picture {picture.shape} does not match coils {coils.shape}")
    net.eval()
    if norm.get("feature_set"):
        mean_np, std_np = norm["feat_mean"], norm["feat_std"]
    else:
        mean_np, std_np = broadcast_coil_stats(
            norm["coil_mean"], norm["coil_std"], coils.shape[1],
            norm["image_mean"], norm["image_std"],
            norm.get("safe_mean"), norm.get("safe_std"),
        )
    coil_mean = torch.as_tensor(mean_np, device=device)
    coil_std = torch.as_tensor(std_np, device=device)
    pixel_mean = torch.as_tensor(norm["pixel_mean"], device=device)
    psi_std = float(norm["psi_std"])
    scal_mean = torch.as_tensor(norm["scal_mean"], device=device)
    scal_std = torch.as_tensor(norm["scal_std"], device=device)
    # A run trained with --no-amp is evaluated without autocast too.
    use_amp = device.type == "cuda" and bool(norm.get("amp", True))
    psi_out = np.empty((len(coils), GRID, GRID), dtype=np.float32)
    q_out = np.empty(len(coils), dtype=np.float32)
    b_out = np.empty(len(coils), dtype=np.float32)
    for start in range(0, len(coils), batch_size):
        stop = min(start + batch_size, len(coils))
        xb = torch.from_numpy(np.ascontiguousarray(coils[start:stop])).to(device)
        pb = torch.from_numpy(np.ascontiguousarray(picture[start:stop])).to(device)
        xb = _clip_tail((xb - coil_mean) / coil_std, norm)
        with precision_context(device.type, use_amp):
            ib = None if ip_ka is None else torch.from_numpy(np.ascontiguousarray(ip_ka[start:stop], dtype=np.float32)).to(device)
            rb = None if rax_th is None else torch.from_numpy(np.ascontiguousarray(rax_th[start:stop], dtype=np.float32)).to(device)
            psi_n, scal_n = net(xb, pb, ib, rb)
        psi = psi_n.float() * psi_std + pixel_mean
        scal = scal_n.float() * scal_std + scal_mean
        psi_out[start:stop] = psi.view(-1, GRID, GRID).cpu().numpy()
        q_out[start:stop] = scal[:, 0].cpu().numpy()
        b_out[start:stop] = scal[:, 1].cpu().numpy()
    return psi_out, q_out, b_out
