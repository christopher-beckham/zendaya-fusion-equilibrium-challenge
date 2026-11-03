"""Label-free current-consistency loss on predicted flux maps (train.py --lambda-current, --mast-current).

From ψ (Wb/rad) on a 65×65 grid, the toroidal current a map implies in each pixel is
J = −Δ*ψ / (μ0 R) · ΔR ΔZ, with Δ*ψ = ψ_RR − ψ_R / R + ψ_ZZ by central differences. Inside the vessel envelope
eroded by 2 pixels (so the stencil never leaves it), let j = −J / Ip per pixel. A real equilibrium carries the
measured plasma current in one direction and nothing in the vacuum around it: on DIII-D val the true maps give
Σj = 0.998 and a wrong-sign share Σ relu(−j) of 0.000 (90% of frames below 0.004). Loss per frame:

    Σ relu(−j) + |Σj − 1|

It needs ψ and Ip only, so it applies to MAST test inputs as well (inputs only; no MAST label is read).
mast_diag/current_check.py and mast_diag/vacuum_check.py measure the same quantities on CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import INPUT_CLIP, stack_history
from model import _plasma_coord_map, plasma_prior_table, prior_from_table

MU0 = 4.0e-7 * np.pi


def _erode(mask: np.ndarray, cells: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(cells):
        e = m.copy()
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m


class CurrentLoss:
    """Per-frame current-consistency loss on one machine's grid. Rows of ψ are Z, columns R."""

    def __init__(self, r: np.ndarray, z: np.ndarray, envelope: np.ndarray, device, erode_px: int = 2,
                 min_ip_ka: float = 150.0):
        r = np.asarray(r, np.float64); z = np.asarray(z, np.float64)
        self.dr, self.dz = float(r[1] - r[0]), float(z[1] - z[0])
        region = _erode(np.asarray(envelope, bool), erode_px)[1:-1, 1:-1]
        self.region = torch.as_tensor(region.astype(np.float32), device=device)
        self.R = torch.as_tensor(r[1:-1].astype(np.float32), device=device).view(1, 1, -1)
        self.min_ip = float(min_ip_ka)

    def terms(self, phys: torch.Tensor, ip_ka: torch.Tensor):
        """phys (B, 4225) Wb/rad, ip_ka (B,) -> (wrong-sign share, Σj − 1), each (B,), float32."""
        psi = phys.float().view(-1, 65, 65)
        c = psi[:, 1:-1, 1:-1]
        p_r = (psi[:, 1:-1, 2:] - psi[:, 1:-1, :-2]) / (2 * self.dr)
        p_rr = (psi[:, 1:-1, 2:] - 2 * c + psi[:, 1:-1, :-2]) / self.dr ** 2
        p_zz = (psi[:, 2:, 1:-1] - 2 * c + psi[:, :-2, 1:-1]) / self.dz ** 2
        dstar = p_rr - p_r / self.R + p_zz
        ip = ip_ka.float().view(-1, 1, 1)
        safe = torch.where(ip.abs() < 1.0, torch.ones_like(ip), ip)
        j = dstar / (MU0 * self.R) * (self.dr * self.dz / 1000.0) / safe * self.region
        return torch.relu(-j).sum((1, 2)), j.sum((1, 2)) - 1.0

    def __call__(self, phys: torch.Tensor, ip_ka: torch.Tensor):
        """Mean loss over frames with |Ip| ≥ min_ip_ka, and the two mean terms for logging."""
        wrong, total = self.terms(phys, ip_ka)
        ok = (ip_ka.float().abs() >= self.min_ip).float()
        n = ok.sum().clamp_min(1.0)
        loss = ((wrong + total.abs()) * ok).sum() / n
        return loss, float((wrong * ok).sum() / n), float((total.abs() * ok).sum() / n)


class MastInputs:
    """MAST public-test decoder inputs on the GPU for the label-free loss: features, coil picture, Ip, prior.

    Reads <cache>/features_<set>_mast_public_test.npy and <cache>/mast_inputs.npz (build_mast_inputs.py), and the
    MAST coil pictures (with `variant`, the files build_mast_sol_half.py writes). Features are standardised and clipped exactly as train.load_split does for DIII-D.
    Only frames with |Ip| ≥ min_ip_ka are drawn.
    """

    def __init__(self, cache: Path, norm: dict, device, min_ip_ka: float = 150.0, variant: str = "",
                 history_step: int = 4):
        import pandas as pd
        import features_v2
        k_hist = int(norm.get("history_frames", 0))
        # variant "_sol0.5": the solenoid halved (build_mast_sol_half.py), in both the features and the pictures.
        feat = np.load(cache / f"features_{norm['feature_set']}_mast_public_test{variant}.npy", mmap_mode="r")
        inp = np.load(cache / "mast_inputs.npz")
        pics = features_v2.load_pictures(cache / "features_v2" / f"pictures_mast{variant}.npz")
        if list(inp["columns"]) != list(pics["columns"]):
            raise SystemExit("mast_inputs.npz columns do not match the MAST pictures")
        cur, ip = inp["cur"], inp["ip"]
        if len(cur) != len(feat):
            raise SystemExit(f"mast_inputs.npz has {len(cur)} rows, the MAST features {len(feat)}")
        keep = np.abs(ip) >= min_ip_ka
        self.rows = np.nonzero(keep)[0]          # row in the MAST caches of each kept frame
        cols = list(norm["feature_columns"])
        if k_hist > 0:
            # History windows per shot, `history_step` MAST frames per offset (5 ms frames: step 4 keeps DIII-D's
            # 20 ms), exactly as predict.predict_mast builds them; then the kept frames.
            index = pd.read_parquet(cache / f"features_{norm['feature_set']}_mast_public_test{variant}.parquet")
            blocks = [stack_history(np.asarray(feat[int(r.start):int(r.start) + int(r.n)], np.float32)[:, cols],
                                    k_hist, history_step) for r in index.sort_values("start").itertuples()]
            stacked = np.concatenate(blocks)
            if len(stacked) != len(feat):
                raise SystemExit(f"history stacking gave {len(stacked)} rows, the MAST features {len(feat)}")
            raw_all = stacked[self.rows]
        else:
            raw_all = np.asarray(feat[self.rows], np.float32)[:, cols]
        z = (raw_all - norm["feat_mean"]) / norm["feat_std"]
        z = np.clip(np.nan_to_num(z, nan=0.0), -INPUT_CLIP, INPUT_CLIP)
        # The current frame's R_ax_th, unscaled (the middle stamp of a history window).
        rax = raw_all[:, k_hist * len(cols) + norm["feature_names"].index("R_ax_th")]
        self.coils = torch.as_tensor(z.astype(np.float32), device=device)
        self.cur = torch.as_tensor(cur[keep].astype(np.float32), device=device)
        self.ip = torch.as_tensor(ip[keep].astype(np.float32), device=device)
        self.rax = torch.as_tensor(rax.astype(np.float32), device=device)
        self.green = torch.as_tensor(np.asarray(pics["green"], np.float32), device=device)
        g = features_v2.geometry("mast")
        self.r0 = float(g["R0"])
        self.coord = _plasma_coord_map(g["r"], g["z"], g["R0"], g["Z0"], g["a"]).to(device)
        self.prior_table = torch.as_tensor(
            plasma_prior_table(g["r"], g["z"], g["R0"], g["Z0"], g["a"]).astype(np.float32), device=device)
        env = np.load(Path(__file__).resolve().parents[2] / "fusion_scoring/masks/mast_envelope.npz")["mask_coarse"]
        self.loss = CurrentLoss(g["r"], g["z"], env, device, min_ip_ka=min_ip_ka)
        self.n = int(keep.sum())
        print(f"MAST inputs for the current loss: {self.n} frames with |Ip| ≥ {min_ip_ka:g} kA "
              f"of {len(ip)}", flush=True)

    def batch(self, b: int) -> dict:
        idx = torch.randint(0, self.n, (b,), device=self.coils.device)
        ip = self.ip[idx]
        return {"idx": idx, "coils": self.coils[idx], "picture": self.cur[idx] @ self.green, "ip": ip, "rax": self.rax[idx],
                "prior": prior_from_table(self.prior_table, ip, self.rax[idx])}


class AxisPull:
    """Keep the map's magnetic axis near the Thomson pressure peak (train.py --lambda-axis). Label-free.

    The axis is the centroid of the cap of φ = −ψ within `depth` (in units of μ0 R0 |Ip| / 2π) of its maximum over
    the eroded envelope; on DIII-D val this is within 0.7 cm of the scorer's O-point on 90% of frames (depth 0.05).
    d = (R_axis − R_th) / a with R_th = R0 + a·R_ax_th. True DIII-D maps have d in −0.18 … +0.07 (2–98%), so the
    penalty is one-sided with slack: relu(d − hi)² + relu(lo − d)², over frames with a Thomson value.
    """

    def __init__(self, r, z, envelope, r0: float, a: float, device, depth: float = 0.05,
                 hi: float = 0.08, lo: float = -0.20, min_ip_ka: float = 150.0,
                 z0: float | None = None, z_slack: float = 0.0):
        # With z0 and z_slack > 0, also relu(|Z_axis − z0| / a − z_slack)²: without it nothing in the MAST losses fixes
        # the vertical position, and a model trained with the ordering losses moved the MAST axis ±0.3 m off the
        # midplane. True DIII-D axes sit in −0.07 … +0.21 a of Z0 (2–98%); 0.25 is symmetric for MAST.
        rr, zz = np.meshgrid(np.asarray(r, np.float64), np.asarray(z, np.float64), indexing="xy")
        self.R = torch.as_tensor(rr.ravel().astype(np.float32), device=device)
        self.Z = torch.as_tensor(zz.ravel().astype(np.float32), device=device)
        self.z0, self.z_slack = z0, z_slack
        self.region = torch.as_tensor(_erode(np.asarray(envelope, bool), 2).ravel().astype(np.float32), device=device)
        self.r0, self.a, self.depth, self.hi, self.lo, self.min_ip = float(r0), float(a), depth, hi, lo, min_ip_ka

    def offset(self, phys: torch.Tensor, ip_ka: torch.Tensor, rax_th: torch.Tensor) -> torch.Tensor:
        """(B,) (R_axis − R_th) / a."""
        scale = MU0 * self.r0 * ip_ka.float().abs().clamp_min(100.0) * 1000.0 / (2 * np.pi)
        phi = -phys.float() / scale.view(-1, 1)
        top = torch.where(self.region > 0, phi, torch.full_like(phi, -1e9)).amax(1, keepdim=True)
        w = torch.relu(phi - (top - self.depth)) * self.region
        r_axis = (w * self.R).sum(1) / w.sum(1).clamp_min(1e-12)
        self._dz = None
        if self.z0 is not None and self.z_slack > 0:
            self._dz = ((w * self.Z).sum(1) / w.sum(1).clamp_min(1e-12) - self.z0) / self.a
        return (r_axis - (self.r0 + self.a * rax_th.float())) / self.a

    def __call__(self, phys, ip_ka, rax_th):
        """Mean penalty over frames with a Thomson value and |Ip| ≥ min_ip_ka, and the median offset for logging."""
        d = self.offset(phys, ip_ka, rax_th)
        big = ip_ka.float().abs() >= self.min_ip
        ok = torch.isfinite(d) & big
        loss = phys.new_zeros(())
        if self._dz is not None and bool(big.any()):
            loss = loss + torch.relu(self._dz[big].abs() - self.z_slack).square().mean()
        if not bool(ok.any()):
            return loss, float("nan")
        d = d[ok]
        loss = loss + (torch.relu(d - self.hi).square() + torch.relu(self.lo - d).square()).mean()
        return loss, float(d.median())


class TeOrder:
    """Hotter Thomson points sit on flux surfaces nearer the axis (train.py --lambda-te). Label-free.

    Te is constant on a flux surface and falls from the axis outwards, so for two channels with
    Te_i > (1 + margin)·Te_j the map must have φ_i ≥ φ_j, with φ = −ψ / (μ0 R0 |Ip| / 2π) (axis at the maximum).
    Penalty per frame: mean over sampled pairs of relu(φ_j − φ_i). φ is read off the map at each channel's (R, Z)
    by bilinear interpolation. Channels used: inside the vessel envelope eroded by 2 pixels, with Te between
    floor·Te95 and spike·Te95 (Te95 = the frame's 95th percentile), which drops cold edge points and single-channel
    spikes. Te, R and Z per frame come from build_thomson_raw.py.
    """

    def __init__(self, raw: Path, rows: np.ndarray, r, z, envelope, r0: float, device, margin: float = 0.3,
                 n_pairs: int = 64, floor: float = 0.05, spike: float = 1.5, min_ip_ka: float = 150.0,
                 quantity: str = "te", scale_free: bool = False):
        # quantity "te", "ne" or "p" (= ne·Te): each is constant on a flux surface and falls outwards. ne and p
        # come from the thomson_raw_ne_* file next to `raw` (build_thomson_raw.py --quantity ne).
        d = np.load(raw)
        te = d["te"][rows].astype(np.float32)
        if quantity != "te":
            ne = np.load(Path(raw).with_name(Path(raw).name.replace("thomson_raw_", "thomson_raw_ne_")))["te"]
            ne = ne[rows].astype(np.float32)
            te = ne if quantity == "ne" else te * ne
        self.quantity = quantity
        # scale_free: divide each frame's gaps by its φ range over the channels (no gradient through the divisor).
        # Without it the penalty shrinks when φ is flat along the chord, which a model can reach by moving the axis
        # far off the chord: two models trained that way put the MAST axis ±0.3 m off the midplane.
        self.scale_free = scale_free
        rr, zz = d["r"][rows].astype(np.float32), d["z"][rows].astype(np.float32)
        r = np.asarray(r, np.float64); z = np.asarray(z, np.float64)
        env = _erode(np.asarray(envelope, bool), 2)
        ir = np.clip(np.round((rr - r[0]) / (r[1] - r[0])).astype(int), 0, len(r) - 1)
        iz = np.clip(np.round((zz - z[0]) / (z[1] - z[0])).astype(int), 0, len(z) - 1)
        ok = np.isfinite(te) & env[iz, ir]
        with np.errstate(all="ignore"):
            t95 = np.nanpercentile(np.where(ok, te, np.nan), 95, axis=1, keepdims=True)
        ok &= (te >= floor * t95) & (te <= spike * t95)
        te = np.where(ok, te, np.nan)
        self.te = torch.as_tensor(te, device=device)
        # grid_sample coordinates in [-1, 1] (x = R, y = Z), align_corners=True.
        gx = 2 * (rr - r[0]) / (r[-1] - r[0]) - 1
        gy = 2 * (zz - z[0]) / (z[-1] - z[0]) - 1
        self.grid = torch.as_tensor(np.stack([gx, gy], -1).astype(np.float32), device=device)
        self.r0, self.margin, self.n_pairs, self.min_ip = float(r0), margin, n_pairs, min_ip_ka
        print(f"{quantity}-order loss: {len(te)} frames, valid channels per frame median "
              f"{np.median(ok.sum(1)):.0f} of {te.shape[1]}", flush=True)

    def __call__(self, phys: torch.Tensor, ip_ka: torch.Tensor, frames: torch.Tensor):
        """phys (B, 4225) Wb/rad; frames (B,) row indices into this loss's arrays. -> mean penalty, violated share."""
        import torch.nn.functional as F
        b = phys.shape[0]
        scale = MU0 * self.r0 * ip_ka.float().abs().clamp_min(100.0) * 1000.0 / (2 * np.pi)
        phi = (-phys.float() / scale.view(-1, 1)).view(b, 1, 65, 65)
        te, grid = self.te[frames], self.grid[frames]                       # (B, C), (B, C, 2)
        val = F.grid_sample(phi, grid.unsqueeze(1), align_corners=True).view(b, -1)
        c = te.shape[1]
        i = torch.randint(0, c, (b, self.n_pairs), device=phys.device)
        j = torch.randint(0, c, (b, self.n_pairs), device=phys.device)
        ti, tj = te.gather(1, i), te.gather(1, j)
        hot = (ti > (1 + self.margin) * tj) & torch.isfinite(ti) & torch.isfinite(tj)
        hot &= (ip_ka.float().abs() >= self.min_ip).view(-1, 1)
        gap = torch.relu(val.gather(1, j) - val.gather(1, i))
        if self.scale_free:
            v = torch.where(torch.isfinite(te), val, torch.full_like(val, float("nan")))
            span = (torch.nan_to_num(v, nan=-1e9).amax(1) - torch.nan_to_num(v, nan=1e9).amin(1)).clamp_min(0.05)
            gap = gap / span.detach().view(-1, 1)
        n = hot.float().sum().clamp_min(1.0)
        loss = (gap * hot.float()).sum() / n
        return loss, float(((gap > 0) & hot).float().sum() / n)
