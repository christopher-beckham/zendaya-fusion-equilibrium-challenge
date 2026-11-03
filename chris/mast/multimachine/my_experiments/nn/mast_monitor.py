"""Per-epoch label-free MAST monitor (train.py, on unless --no-mast-monitor). MAST inputs only; no MAST label.

A fixed set of real MAST frames (consecutive triplets, so jitter can be measured) goes through the network after each
epoch, with no gradient and no effect on training. Logged (medians unless stated):
  mast_dR / mast_dZ       axis offset from the Thomson pressure peak, (R_axis − R_th) / a, and |Z_axis − Z0| / a; the
  mast_dRabs              median of |R_axis − R_th| / a per frame: tracks the leaderboard R_axis term across 7 submitted
                          maps (proxy_calibration/REPORT.md in the main repo); the signed median mast_dR does not.
                          axis is AxisPull's cap centroid (within 0.7 cm of the scorer's O-point on DIII-D val)
  mast_cur                current carried by the map inside the eroded MAST region (mast_region.py) ÷ measured Ip
                          (CurrentLoss)
  mast_wrong              share of that current with the wrong sign
  mast_jit_p95            95th percentile of |R_t − (R_{t−1} + R_{t+1}) / 2| / a over the triplets (5 ms frames)
  mast_spread, mast_coll  linear decoder only: rms distance of |J| from its centroid ÷ nominal a, and the share of
                          frames below 0.38 (every DIII-D true map is above 0.41)
  mast_peak, mast_peaked  linear decoder only: max |J| ÷ mean |J| over cells carrying current, and the share of frames
                          above 30. The spread measure misses collapse onto a compact core of a few cells; the peak
                          ratio does not (collapsed MAST frames ~107, normal ~7, 99th pct ~14).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model import prior_from_table
from physics_loss import AxisPull, MastInputs

COLLAPSE = 0.38
PEAKED = 30.0


class MastMonitor:
    def __init__(self, cache: Path, norm: dict, device, variant: str = "_sol0.5", n_centres: int = 512,
                 mast: MastInputs | None = None, seed: int = 0):
        import features_v2
        self.mi = mast if mast is not None else MastInputs(cache, norm, device, variant=variant)
        rng = np.random.default_rng(seed)
        c = self.mi.centres.cpu().numpy()
        c = np.sort(rng.choice(c, size=min(n_centres, len(c)), replace=False))
        c = torch.as_tensor(c, device=device)
        self.idx = torch.cat([c, c - 1, c + 1])
        self.n = len(c)
        g = features_v2.geometry("mast")
        import mast_region
        self.axis = AxisPull(g["r"], g["z"], mast_region.mask(), g["R0"], g["a"], device, z0=g["Z0"], z_slack=1e-6)
        self.a = float(g["a"])
        rr, zz = np.meshgrid(g["r"], g["z"], indexing="xy")
        self.rr = rr.ravel()
        self.zz = zz.ravel()
        self.env_rz = None

    @torch.no_grad()
    def __call__(self, net, linear: bool, batch: int = 768) -> dict:
        mi = self.mi
        net.eval()
        dR, dZ, cur, wrong, sp, rax, pk = [], [], [], [], [], [], []
        for i in range(0, len(self.idx), batch):
            ii = self.idx[i:i + batch]
            ip, rx = mi.ip[ii], mi.rax[ii]
            kw = dict(coord=mi.coord, prior=prior_from_table(mi.prior_table, ip, rx), r0=mi.r0)
            if linear:
                kw.update(machine="mast", return_current=True)
            out = net(mi.coils[ii], mi.cur[ii] @ mi.green, ip, rx, **kw)
            phys = out[0].float() * net.psi_std + net.pixel_mean
            d = self.axis.offset(phys, ip, rx)
            dR.append(d); dZ.append(self.axis._dz.abs())
            w_, t_ = mi.loss.terms(phys, ip)
            cur.append(t_ + 1.0); wrong.append(w_ / (t_ + 1.0).abs().clamp_min(1e-6))
            # axis R for jitter: offset × a + Thomson R (undo the Thomson reference)
            rax.append(d * self.axis.a + self.axis.r0 + self.axis.a * rx.float())
            if linear:
                if self.env_rz is None:
                    idx = getattr(net, "idx_mast").cpu().numpy()
                    self.env_rz = (torch.as_tensor(self.rr[idx], dtype=torch.float32, device=phys.device),
                                   torch.as_tensor(self.zz[idx], dtype=torch.float32, device=phys.device))
                R, Z = self.env_rz
                w = out[2].float().abs()
                w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
                cR = (w * R).sum(1, keepdim=True); cZ = (w * Z).sum(1, keepdim=True)
                sp.append(((w * ((R - cR) ** 2 + (Z - cZ) ** 2)).sum(1)).sqrt() / self.a)
                on = (w > 1e-6 * w.amax(1, keepdim=True)).float()
                pk.append(w.amax(1) / ((w * on).sum(1) / on.sum(1).clamp_min(1)))
        cat = lambda xs: torch.cat(xs).float()  # noqa: E731
        dR, dZ, cur, wrong, rax = cat(dR), cat(dZ), cat(cur), cat(wrong), cat(rax)
        n = self.n
        r0, rm, rp = rax[:n], rax[n:2 * n], rax[2 * n:]
        zig = ((r0 - 0.5 * (rm + rp)).abs() / self.a)
        zig = zig[torch.isfinite(zig)]
        med = lambda x: float(x[torch.isfinite(x)].median()) if bool(torch.isfinite(x).any()) else float("nan")  # noqa: E731
        out = {"mast_dR": med(dR), "mast_dRabs": med(dR.abs()), "mast_dZ": med(dZ), "mast_cur": med(cur), "mast_wrong": med(wrong),
               "mast_jit_p95": float(torch.quantile(zig, 0.95)) if len(zig) else float("nan")}
        if sp:
            s = cat(sp)
            out.update(mast_spread=med(s), mast_coll=float((s < COLLAPSE).float().mean()))
            q = cat(pk)
            out.update(mast_peak=med(q), mast_peaked=float((q > PEAKED).float().mean()))
        return out

    @staticmethod
    def line(m: dict) -> str:
        s = (f"MAST dR {m['mast_dR']:+.3f} |dR| {m['mast_dRabs']:.3f} dZ {m['mast_dZ']:.3f} cur {m['mast_cur']:.3f} jit {m['mast_jit_p95']:.3f}")
        if "mast_spread" in m:
            s += f" spread {m['mast_spread']:.2f} coll {m['mast_coll']:.1%} peak {m['mast_peak']:.0f} peaked {m['mast_peaked']:.1%}"
        return s + "  "
