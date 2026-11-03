"""Linear decoder (train.py --decoder linear): the network outputs a current map, the flux follows from it.

    ψ = coil picture + G · J + c · s

J (A per cell, labels' sign convention) lives on the machine's envelope cells. The conv decoder of model.Net emits
one map per frame; on the envelope cells it becomes a non-negative weight (softplus, or exp for a softmax), and the
weights are scaled to sum to one and multiplied by −Ip · g, so Σ J = −Ip · g, the sign the true maps give
(J = −Δ*ψ / (μ0 R) · ΔR ΔZ sums to −Ip on DIII-D labels). g = exp(head(trunk)) is a per-frame total-current gain that
starts at 1. It is needed: the labels' total current is not the measured Ip (Σ J_true / −Ip is 0.998 at the median
on DIII-D val, 0.987 … 1.03 for 5–95% of frames, 1.40 at the 99th percentile), and forcing Σ J = −Ip exactly on the
true current drops R²ψ from 0.997 to 0.986 and the consistency scalars from 0.955 to 0.875 (archive/top/lindec_tests.py a).
`exact_ip` turns the gain off. G[p, c] is the flux at grid point p from 1 A in envelope
cell c (lindec_build.py, from gs_solver.gs.Grid). c is one offset per frame from a linear head on the trunk, in units
of the flux scale s = μ0 R0 |Ip| / 2π.

With the true current and the fitted-gain pictures (baseline_pictures_fitgain2.npz) this form reproduces DIII-D val
maps at R²ψ 0.997 and the scorer's consistency scalars at 0.95 (lindec_build.py --verify), so the decoder's ceiling
is set by how well the network predicts J.

The output is returned in model.Net's normalised units, (ψ − pixel_mean) / psi_std, so every loss and caller of
Net works unchanged. G is not stored in checkpoints; it is read from <cache>/lindec_G_<machine>.npz (or rebuilt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import CACHE, GRID, N_PSI
from model import Net, flux_scale, strict_fp32

MU0 = 4.0e-7 * np.pi
MACHINES = ("d3d", "mast")


def _erode(mask: np.ndarray, cells: int) -> np.ndarray:
    m = np.asarray(mask, bool).copy()
    for _ in range(cells):
        e = m.copy()
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m


def current_from_psi(phys: torch.Tensor, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """(B, 4225) ψ in Wb/rad, labels' convention -> (B, 4225) current per cell in A, J = −Δ*ψ / (μ0 R) · ΔR ΔZ.

    Central differences as lindec_build.true_current and physics_loss.CurrentLoss; 0 on the grid border.
    """
    psi = phys.float().view(-1, GRID, GRID)
    dr, dz = r[1] - r[0], z[1] - z[0]
    c = psi[:, 1:-1, 1:-1]
    p_r = (psi[:, 1:-1, 2:] - psi[:, 1:-1, :-2]) / (2 * dr)
    p_rr = (psi[:, 1:-1, 2:] - 2 * c + psi[:, 1:-1, :-2]) / dr ** 2
    p_zz = (psi[:, 2:, 1:-1] - 2 * c + psi[:, :-2, 1:-1]) / dz ** 2
    R = r[1:-1].view(1, 1, -1)
    inner = -(p_rr - p_r / R + p_zz) / (MU0 * R) * dr * dz
    return F.pad(inner, (1, 1, 1, 1)).view(-1, N_PSI)


class AttnField(nn.Module):
    """Coordinate-query decoder: each envelope cell asks the trunk how much current it carries.

    The trunk vector becomes `n_tokens` latent tokens. A cell's query is built from its plasma-unit coordinates
    ((R − R0)/a, (Z − Z0)/a, R/R0), low-frequency Fourier features of the first two, and the coil picture's flux at
    the cell in units of the frame's flux scale. Queries cross-attend to the tokens (`layers` pre-norm blocks), then a
    linear head gives the cell's logit. Nothing depends on the grid index or size, so a cell means the same place
    relative to the plasma on every machine (the conv decoder's 8×8 seed and 3×3 kernels work in grid cells).
    """

    def __init__(self, width: int, n_tokens: int = 32, dim: int = 128, heads: int = 4, layers: int = 2,
                 n_freq: int = 5, n_coord: int = 3):
        super().__init__()
        self.n_tokens, self.dim, self.heads = n_tokens, dim, heads
        self.tokens = nn.Linear(width, n_tokens * dim)
        self.tok_norm = nn.LayerNorm(dim)
        # coordinates span about ±4 a; the lowest frequency has a period of 8 a
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * (np.pi / 4), persistent=False)
        q_in = n_coord + 4 * n_freq + 1
        self.q_embed = nn.Sequential(nn.Linear(q_in, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(dim), "q": nn.Linear(dim, dim), "kv": nn.Linear(dim, 2 * dim),
                "o": nn.Linear(dim, dim), "ln_m": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))}))
        self.out = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, coord: torch.Tensor, pic_n: torch.Tensor) -> torch.Tensor:
        """h (B, width), coord (N, n_coord) plasma units, pic_n (B, N) picture / flux scale -> logits (B, N)."""
        B, N = pic_n.shape
        t = self.tok_norm(self.tokens(h).view(B, self.n_tokens, self.dim))
        ang = coord[:, :2, None] * self.freqs                                  # (N, 2, F)
        cell = torch.cat([coord, torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)], dim=1)
        q = self.q_embed(torch.cat([cell.unsqueeze(0).expand(B, -1, -1).to(pic_n.dtype),
                                    pic_n.clamp(-10, 10).unsqueeze(-1)], dim=-1).to(h.dtype))
        hd = self.dim // self.heads
        for blk in self.blocks:
            qq = blk["q"](blk["ln_q"](q)).view(B, N, self.heads, hd).transpose(1, 2)
            k, v = blk["kv"](t).view(B, self.n_tokens, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(qq, k, v).transpose(1, 2).reshape(B, N, self.dim)
            q = q + blk["o"](a)
            q = q + blk["mlp"](blk["ln_m"](q))
        return self.out(q).squeeze(-1)


class LinearDecoderNet(Net):
    """model.Net whose final conv emits a current map instead of ψ. Needs decoder_norm "physical".

    `erode` keeps J off the envelope cells within that many cells of its edge (0 = every envelope cell).
    `activation` "softplus" or "exp" turns the map into non-negative weights. `exact_ip` fixes Σ J = −Ip (no gain).
    `ip_floor` (kA, 0 = off) adds a learned current below that level: the total uses Ip + ip_floor · tanh(t) ·
    relu(1 − |Ip| / ip_floor), t from a zero-initialised head. Measured Ip near zero (e.g. the first frame of a DIII-D
    shot, where 40% of frames under 150 kA have the opposite sign to the rest of the shot) otherwise leaves J ≈ 0 and
    the map equal to the coil picture.
    Both machines' G are loaded at construction (non-persistent buffers); `machine` in forward() or set_machine()
    picks one. The output prior of Net is not used: G · J is the plasma's flux.
    """

    def __init__(self, *args, erode: int = 0, activation: str = "softplus", exact_ip: bool = False,
                 ip_floor: float = 0.0, exact_ip_mast: bool = False, attn_decoder: bool = False,
                 cache: Path = CACHE, mast_mask: str | None = None, **kw):
        if kw.get("output_prior"):
            raise ValueError("the linear decoder replaces the output prior; leave output_prior off")
        if kw.get("decoder_norm") != "physical":
            raise ValueError("the linear decoder needs decoder_norm 'physical'")
        super().__init__(*args, **kw)
        if activation not in ("softplus", "exp"):
            raise ValueError(f"activation must be 'softplus' or 'exp', got {activation!r}")
        self.erode, self.activation = int(erode), activation
        self.exact_ip = bool(exact_ip)
        self.ip_floor = float(ip_floor)
        # exact_ip_mast: Σ J = −Ip exactly on MAST (no gain). Nothing but a light current loss holds the gain there,
        # and it drifted to 0.89 when the Thomson pressure term reshaped the trunk.
        self.exact_ip_mast = bool(exact_ip_mast)
        # Zero-initialised heads on the trunk: the offset starts at 0 and the total-current gain at exp(0) = 1.
        self.offset = nn.Linear(self.in_proj.out_features, 1)
        self.total = None if self.exact_ip else nn.Linear(self.in_proj.out_features, 1)
        self.low = nn.Linear(self.in_proj.out_features, 1) if self.ip_floor > 0 else None
        for head in (self.offset, self.total, self.low):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        from lindec_build import load_g
        import mast_region
        # MAST current cells: the region the run was trained with (mast_region.py; default the FEC_MAST_MASK selection)
        self.mast_mask = mast_region.name() if mast_mask is None else mast_region._check(mast_mask)
        self.grids = {}
        for m in MACHINES:
            g = load_g(m, cache, self.mast_mask)
            region = _erode(g["mask"], self.erode).ravel()[g["idx"]]
            if not region.any():
                raise ValueError(f"{m}: eroding the envelope by {self.erode} cells leaves no cell")
            self.register_buffer(f"G_{m}", torch.from_numpy(np.ascontiguousarray(g["G"], np.float32)), persistent=False)
            self.register_buffer(f"idx_{m}", torch.from_numpy(g["idx"].astype(np.int64)), persistent=False)
            self.register_buffer(f"region_{m}", torch.from_numpy(region.astype(np.float32)), persistent=False)
            self.grids[m] = (np.asarray(g["r"]), np.asarray(g["z"]))
        self.machine = "d3d"
        # attn_decoder: the conv decoder is replaced by AttnField queried at the envelope cells
        self.attn_decoder = bool(attn_decoder)
        if self.attn_decoder:
            self.field = AttnField(self.in_proj.out_features, n_coord=self.coord.shape[0])
            self.to_grid, self.stages, self.final = None, None, None

    def set_machine(self, r: np.ndarray, z: np.ndarray, geometry: tuple[float, float, float],
                    machine: str | None = None) -> None:
        """As Net.set_machine, and select that machine's G. Without `machine`, the grid identifies it."""
        super().set_machine(r, z, geometry)
        if machine is None:
            for m, (gr, gz) in self.grids.items():
                if gr.shape == np.shape(r) and np.allclose(gr, r, atol=1e-6) and np.allclose(gz, z, atol=1e-6):
                    machine = m
            if machine is None:
                raise ValueError("this grid matches no machine with a G matrix; pass machine=")
        self.machine = machine

    def decode_field(self, coils, picture, ip_ka, coord, r0, machine):
        """As Net.decode with the AttnField decoder; logits outside the machine's envelope cells are 0 (unused)."""
        h = self.blocks(self.in_proj(coils))
        idx = getattr(self, f"idx_{machine}")
        with torch.autocast(device_type=coils.device.type, enabled=False):
            pic = picture.float().reshape(-1, N_PSI)
            scale = flux_scale(ip_ka, self.r0 if r0 is None else r0).view(-1, 1)
            pic_n = (pic / scale)[:, idx]
            coord_in = self.coord if coord is None else coord
            cells = coord_in.reshape(coord_in.shape[-3], -1).T[idx].float()
        out = self.field(h, cells, pic_n)
        logits = torch.zeros(pic.shape[0], N_PSI, device=pic.device, dtype=torch.float32)
        logits[:, idx] = out.float()
        return h, logits, pic, scale

    def current(self, logits: torch.Tensor, ip_ka: torch.Tensor, machine: str,
                gain: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 4225) conv output -> J (B, N_env) in A on that machine's envelope cells, Σ J = −Ip · gain (B, 1)."""
        idx, region = getattr(self, f"idx_{machine}"), getattr(self, f"region_{machine}")
        x = logits.float()[:, idx]
        if self.activation == "softplus":
            w = F.softplus(x) * region
            frac = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        else:
            frac = torch.softmax(x.masked_fill(region == 0, float("-inf")), dim=1)
        total = -(ip_ka.float() * 1000.0).view(-1, 1)
        return total * frac if gain is None else total * gain * frac

    def forward(self, coils: torch.Tensor, picture: torch.Tensor,
                ip_ka: torch.Tensor | None = None,
                rax_th: torch.Tensor | None = None,
                coord: torch.Tensor | None = None,
                prior: torch.Tensor | None = None,
                r0: float | None = None,
                machine: str | None = None,
                return_current: bool = False):
        """As Net.forward (rax_th and prior are accepted and ignored). `machine` picks G, default self.machine.
        With return_current, also returns J (B, N_env) in A."""
        if ip_ka is None:
            raise ValueError("the linear decoder needs ip_ka")
        machine = machine or self.machine
        if self.attn_decoder:
            h, logits, pic, scale = self.decode_field(coils, picture, ip_ka, coord, r0, machine)
        else:
            h, logits, pic, scale = self.decode(coils, picture, ip_ka, coord, r0)
        with strict_fp32(coils.device.type):
            gain = None if self.total is None or (self.exact_ip_mast and machine == "mast") \
                else torch.exp(self.total(h.float()))
            ip_eff = ip_ka.float()
            if self.low is not None:
                fade = (1.0 - ip_eff.abs() / self.ip_floor).clamp_min(0.0)
                ip_eff = ip_eff + self.ip_floor * torch.tanh(self.low(h.float())).view(-1) * fade
            J = self.current(logits, ip_eff, machine, gain)
            phys = pic + J @ getattr(self, f"G_{machine}").T + self.offset(h.float()) * scale
            psi = (phys - self.pixel_mean) / self.psi_std
            scalars = self.scalar(h.float())
        if return_current:
            return psi, scalars, J
        return psi, scalars
