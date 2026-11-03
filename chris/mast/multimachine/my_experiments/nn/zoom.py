"""Zoom augmentation for the physical decoder (train.py --zoom-aug).

Each training example is drawn on its own 65×65 grid instead of DIII-D's: the native grid scaled about its centre
by zr in R and zz in Z and shifted outward by `shift`·a in R. The decoder then sees pixel spacings and positions in
plasma units like MAST's (its grid is about 1.3× wider, 1.45× taller and 0.33a further outboard than DIII-D's), and
the coordinate channels tell it which grid it is on.

For each example:
  coil picture   sampled from coil pictures precomputed on a fine canvas that contains every zoomed grid
  plasma prior   sampled from the prior table on the same canvas (model.plasma_prior_table), at the frame's R_ax_th
  coordinates    (R − R0)/a, (Z − Z0)/a, R/R0 of the zoomed grid
  target         EFIT's ψ resampled onto the zoomed grid (bicubic); pixels outside DIII-D's grid are masked out
The ψ and slope losses are computed only over unmasked pixels, the slopes with the example's own pixel spacing.
Validation and scoring stay on the native grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ECOIL_COL, F_COILS, F_SLICE, GRID, IP_COL, N_PSI, load_rz

CANVAS = 257


def _canvas_axes(zr_max: float, zz_max: float, shift_max_m: float, shift_min_m: float):
    r, z = load_rz()
    rc, zc = 0.5 * (r[0] + r[-1]), 0.5 * (z[0] + z[-1])
    hr, hz = 0.5 * (r[-1] - r[0]), 0.5 * (z[-1] - z[0])
    r_lo = rc - zr_max * hr + min(shift_min_m, 0.0)
    r_hi = rc + zr_max * hr + max(shift_max_m, 0.0)
    pad_r, pad_z = (r_hi - r_lo) / (CANVAS - 1), 2 * zz_max * hz / (CANVAS - 1)
    return (np.linspace(r_lo - pad_r, r_hi + pad_r, CANVAS),
            np.linspace(zc - zz_max * hz - pad_z, zc + zz_max * hz + pad_z, CANVAS))


def _coil_canvas(rc: np.ndarray, zc: np.ndarray, data_file: Path, ecoil_scale: float, sign: float) -> np.ndarray:
    """(19, CANVAS, CANVAS): F-coil pictures in coils.npy order, then ECOILA × ecoil_scale, per unit current."""
    from shift_report import coil_pictures
    pics = coil_pictures(data_file, "d3d", {"r": rc, "z": zc})
    rows = [pics[f"magnetics_{c}"] for c in F_COILS] + [pics["magnetics_ECOILA"] * ecoil_scale]
    return (sign * np.stack(rows)).reshape(len(rows), CANVAS, CANVAS)


class ZoomAug:
    def __init__(self, geometry: tuple[float, float, float], zr: tuple[float, float], zz: tuple[float, float],
                 shift: tuple[float, float], native_frac: float, data_file: Path, cache_dir: Path, device):
        from features_v2 import D3D_ECOIL_SCALE
        from model import PRIOR_RAX, _picture_sign, plasma_prior_table
        self.R0, self.Z0, self.a = (float(v) for v in geometry)
        self.zr, self.zz, self.shift, self.native_frac = zr, zz, shift, float(native_frac)
        self.device = device
        r, z = load_rz()
        self.r_nat = torch.as_tensor(r, dtype=torch.float32, device=device)
        self.z_nat = torch.as_tensor(z, dtype=torch.float32, device=device)
        self.rc, self.zc = 0.5 * (r[0] + r[-1]), 0.5 * (z[0] + z[-1])
        rax, zax = _canvas_axes(zr[1], zz[1], shift[1] * self.a, shift[0] * self.a)
        self.r_lo, self.r_hi, self.z_lo, self.z_hi = rax[0], rax[-1], zax[0], zax[-1]
        key = f"zoom_canvas_{CANVAS}_{zr[1]:.3f}_{zz[1]:.3f}_{shift[0]:.3f}_{shift[1]:.3f}.npz"
        path = Path(cache_dir) / key
        if path.exists():
            with np.load(path) as d:
                coil, prior = d["coil"], d["prior"]
        else:
            print(f"building the zoom canvas ({CANVAS}×{CANVAS}, R {rax[0]:.2f}..{rax[-1]:.2f} m, "
                  f"Z {zax[0]:.2f}..{zax[-1]:.2f} m) -> {path}", flush=True)
            sign = _picture_sign()
            coil = _coil_canvas(rax, zax, data_file, D3D_ECOIL_SCALE, sign).astype(np.float32)
            prior = plasma_prior_table(rax, zax, self.R0, self.Z0, self.a).reshape(len(PRIOR_RAX), CANVAS, CANVAS)
            prior = prior.astype(np.float32)
            np.savez(path, coil=coil, prior=prior)
        # Branch fitgain-pictures: per-coil gains (18 F-coils, then ECOILA on top of ×D3D_ECOIL_SCALE) fitted on DIII-D
        # train labels with the plasma current read off the true maps (archive/gs_solver/test0b_calibration.py). The same gains
        # are in baseline_pictures_fitgain.npz, so zoomed and native examples share one coil calibration.
        # fitgain2: the ECOILA row keeps its level (envelope mean of the ×135 picture on DIII-D's grid) and only its
        # shape about that level is scaled, as in baseline_pictures_fitgain2.npz.
        here = Path(__file__).resolve().parent / "gs_solver"
        gains = np.load(here / "d3d_coil_gains.npy").astype(np.float32)
        level = float(np.load(here / "d3d_ecoil_level.npy"))
        coil = coil.copy()
        coil[:18] = coil[:18] * gains[:18, None, None]
        coil[18] = level + gains[18] * (coil[18] - level)
        self.coil = torch.as_tensor(coil, device=device)
        self.prior = torch.as_tensor(prior, device=device)
        self.prior_rax = (float(PRIOR_RAX[0]), float(PRIOR_RAX[-1]), len(PRIOR_RAX))

    def sample(self, b: int) -> dict:
        """Per-example zoom factors and outward shift in metres. A native_frac share stays on the native grid."""
        dev = self.device
        zr = torch.empty(b, device=dev).uniform_(*self.zr)
        zz = torch.empty(b, device=dev).uniform_(*self.zz)
        sh = torch.empty(b, device=dev).uniform_(*self.shift) * self.a
        native = torch.rand(b, device=dev) < self.native_frac
        zr = torch.where(native, torch.ones_like(zr), zr)
        zz = torch.where(native, torch.ones_like(zz), zz)
        sh = torch.where(native, torch.zeros_like(sh), sh)
        return {"zr": zr, "zz": zz, "shift": sh}

    def grids(self, p: dict):
        """Physical R (B, W) and Z (B, H) of each example's grid."""
        r = self.rc + p["shift"].view(-1, 1) + p["zr"].view(-1, 1) * (self.r_nat.view(1, -1) - self.rc)
        z = self.zc + p["zz"].view(-1, 1) * (self.z_nat.view(1, -1) - self.zc)
        return r, z

    @staticmethod
    def _sample_grid(r: torch.Tensor, z: torch.Tensor, r_lo, r_hi, z_lo, z_hi) -> torch.Tensor:
        """(B, H, W, 2) grid_sample coordinates (x = R, y = Z in [-1, 1]) for the box [r_lo, r_hi] × [z_lo, z_hi]."""
        x = 2.0 * (r - r_lo) / (r_hi - r_lo) - 1.0
        y = 2.0 * (z - z_lo) / (z_hi - z_lo) - 1.0
        b, w, h = r.shape[0], r.shape[1], z.shape[1]
        return torch.stack([x.view(b, 1, w).expand(b, h, w), y.view(b, h, 1).expand(b, h, w)], dim=-1)

    def batch(self, stamp: torch.Tensor, rax_th: torch.Tensor, psi_n: torch.Tensor, pixel_mean: torch.Tensor,
              psi_std: torch.Tensor, p: dict) -> dict:
        """Inputs and masked target for one batch. stamp (B, 21) mirrored currents, psi_n (B, 4225) native target."""
        with torch.autocast(device_type=stamp.device.type, enabled=False):
            r, z = self.grids(p)
            b = r.shape[0]
            g = self._sample_grid(r, z, self.r_lo, self.r_hi, self.z_lo, self.z_hi)
            coil = F.grid_sample(self.coil.unsqueeze(0).expand(b, -1, -1, -1), g, mode="bilinear",
                                 align_corners=True)
            cur = torch.cat([stamp[:, F_SLICE], stamp[:, ECOIL_COL:ECOIL_COL + 1]], dim=1).float()
            picture = torch.einsum("bc,bchw->bhw", cur, coil).reshape(b, N_PSI)
            lo, hi, n = self.prior_rax
            rax = torch.nan_to_num(rax_th.float(), nan=0.1)
            pos = ((rax.clamp(lo, hi) - lo) / (hi - lo) * (n - 1)).clamp(0, n - 1 - 1e-6)
            i0 = pos.floor().long()
            w = (pos - i0.float()).view(-1, 1)
            pair = torch.stack([self.prior[i0], self.prior[i0 + 1]], dim=1)
            pr = F.grid_sample(pair, g, mode="bilinear", align_corners=True).view(b, 2, N_PSI)
            prior = ((1 - w) * pr[:, 0] + w * pr[:, 1]) * stamp[:, IP_COL].float().view(-1, 1)
            rr = r.view(b, 1, GRID).expand(b, GRID, GRID)
            zz = z.view(b, GRID, 1).expand(b, GRID, GRID)
            coord = torch.stack([(rr - self.R0) / self.a, (zz - self.Z0) / self.a, rr / self.R0], dim=1)
            # Target: EFIT's ψ on the native grid, resampled onto the example's grid; outside it, masked.
            phys = (psi_n.float() * psi_std + pixel_mean).view(b, 1, GRID, GRID)
            gn = self._sample_grid(r, z, float(self.r_nat[0]), float(self.r_nat[-1]),
                                   float(self.z_nat[0]), float(self.z_nat[-1]))
            target = F.grid_sample(phys, gn, mode="bicubic", align_corners=True).view(b, N_PSI)
            eps = 1e-6
            mask = ((gn[..., 0].abs() <= 1 + eps) & (gn[..., 1].abs() <= 1 + eps)).view(b, N_PSI)
            target_n = (target - pixel_mean) / psi_std
        return {"picture": picture, "prior": prior, "coord": coord, "target_n": target_n, "mask": mask,
                "dr": (r[:, 1] - r[:, 0]), "dz": (z[:, 1] - z[:, 0])}


def masked_losses(pred_n: torch.Tensor, target_n: torch.Tensor, mask: torch.Tensor,
                  dr: torch.Tensor, dz: torch.Tensor):
    """Masked ψ MSE and masked slope MSE, both in the normalised units train.py uses.

    The slope term takes central differences with each example's spacing, over interior pixels whose two
    neighbours are unmasked too. pixel_mean cancels in pred − target, so it is the slope of the error / psi_std.
    """
    d = (pred_n.float() - target_n.float())
    m = mask.float()
    mse = (d.square() * m).sum() / m.sum().clamp_min(1.0)
    e = d.view(-1, GRID, GRID)
    mm = mask.view(-1, GRID, GRID)
    gr = (e[:, :, 2:] - e[:, :, :-2]) / (2.0 * dr.view(-1, 1, 1))
    mr = (mm[:, :, 2:] & mm[:, :, :-2] & mm[:, :, 1:-1]).float()
    gz = (e[:, 2:, :] - e[:, :-2, :]) / (2.0 * dz.view(-1, 1, 1))
    mz = (mm[:, 2:, :] & mm[:, :-2, :] & mm[:, 1:-1, :]).float()
    grad = 0.5 * ((gr.square() * mr).sum() / mr.sum().clamp_min(1.0) + (gz.square() * mz).sum() / mz.sum().clamp_min(1.0))
    return mse, grad
