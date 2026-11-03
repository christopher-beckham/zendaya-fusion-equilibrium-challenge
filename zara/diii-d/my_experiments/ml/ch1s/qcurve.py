"""q profile of a predicted (or true) flux map relative to its last closed surface, for any boundary fraction x_b afterwards:

    q(L) = F0 / (2 pi) * loop-integral dl / (R |grad psi|) on psi = psi_ax + L (psi_sep - psi_ax),  L in LEVELS
    q95(x_b) = interpolation of log q at L = 0.95 x_b   (q95_from_curve)

psi_ax = spline value at the scorer O-point, psi_sep = mean spline psi on the scorer-extracted LCFS; F0 = vacuum R B_T from the
TF coil current. Also stores psi at the inner-wall lines R = 1.020 / 1.033 / 1.046 m (|Z| <= 0.3) relative to the map
(x_wall = (min psi_wall - psi_ax) / (psi_sep - psi_ax)) and the LCFS Rmin and its Z, as boundary-model features.

    python qcurve.py --stems hold0 --preds <dir> --out <npz> [--true]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physrun import stems_of  # noqa: E402

LEVELS = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90,
                   0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
EXTRA = ["psi_ax", "psi_sep", "x_wall1020", "x_wall1033", "x_wall1046", "Rmin", "Z_Rmin", "Rmax", "Zmax", "Zmin", "ok"]
ZL = np.linspace(-0.3, 0.3, 25)


def q_at(sp, phi, iz, ir, lev, F0):
    import physfeat as pf
    c = pf._contour_at(phi, -lev, iz, ir)
    if c is None:
        return np.nan
    P = pf.grid_to_physical(c, pf.GR, pf.GZ)
    for _ in range(2):
        v = sp.ev(P[:, 1], P[:, 0]) - lev
        gR, gZ = sp.ev(P[:, 1], P[:, 0], dy=1), sp.ev(P[:, 1], P[:, 0], dx=1)
        g2 = np.maximum(gR * gR + gZ * gZ, 1e-12)
        P = P - np.column_stack([v * gR / g2, v * gZ / g2])
    P2 = np.vstack([P, P[:1]])
    sg = np.hypot(np.diff(P2[:, 0]), np.diff(P2[:, 1]))
    Pm = 0.5 * (P2[:-1] + P2[1:])
    g = np.hypot(sp.ev(Pm[:, 1], Pm[:, 0], dx=1), sp.ev(Pm[:, 1], Pm[:, 0], dy=1))
    return F0 / (2 * math.pi) * float(np.sum(sg / (Pm[:, 0] * np.maximum(g, 1e-9))))


def frame(psi, bt):
    import physfeat as pf
    from scipy.interpolate import RectBivariateSpline
    q = np.full(len(LEVELS), np.nan)
    ex = np.full(len(EXTRA), np.nan)
    try:
        psi = np.asarray(psi, np.float64)
        sp = RectBivariateSpline(pf.GZ, pf.GR, psi, kx=3, ky=3)
        Ra, Za, iz, ir, _ = pf.find_o_point(-psi, pf.GR, pf.GZ, pf.MASK)
        pa = float(sp.ev(Za, Ra))
        C, _ = pf.extract_lcfs_with_sign(psi, pf.GR, pf.GZ, "DIII-D", pf.MASK, pf.MASKF, n_iter=22, axis_sign=-1.0)
        ex[0] = pa
        if C is None:
            return q, ex
        ps = float(np.mean(sp.ev(C[:, 1], C[:, 0])))
        ex[1] = ps
        if not ps > pa:
            return q, ex
        for j, Rw in enumerate((1.020, 1.033, 1.046)):
            ex[2 + j] = (float(np.min(sp.ev(ZL, np.full_like(ZL, Rw)))) - pa) / (ps - pa)
        jr = int(np.argmin(C[:, 0]))
        ex[5], ex[6], ex[7], ex[8], ex[9] = C[jr, 0], C[jr, 1], C[:, 0].max(), C[:, 1].max(), C[:, 1].min()
        F0 = pf.MU0 * 144 * float(bt) / (2 * math.pi)
        for j, L in enumerate(LEVELS):
            q[j] = q_at(sp, -psi, iz, ir, pa + L * (ps - pa), F0)
        ex[10] = 1.0
    except Exception:
        pass
    return q, ex


def job(a):
    s, pdir, cache, true = a
    with np.load(cache / f"{s}.npz") as d:
        bt = d["bt"].astype(np.float64)
        psi = d["psi"] if true else None
    if not true:
        with np.load(Path(pdir) / f"{s}.npz") as p:
            psi = p["psi"]
    Q, E = zip(*[frame(psi[k], bt[k]) for k in range(len(bt))])
    return np.array(Q), np.array(E)


def q95_from_curve(Q, xb):
    """q at L = 0.95 x_b by linear interpolation of log q over LEVELS (NaN levels skipped); xb clipped to [0.3/0.95, 1]."""
    L = 0.95 * np.clip(xb, LEVELS[0] / 0.95, 1.0)
    out = np.full(len(L), np.nan)
    lq = np.log(np.where(Q > 0, Q, np.nan))
    for i in range(len(L)):
        m = np.isfinite(lq[i])
        if m.sum() >= 2:
            out[i] = np.exp(np.interp(L[i], LEVELS[m], lq[i, m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="hold0")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--true", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    stems, cache = stems_of(a.stems)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(job, [(s, a.preds, cache, a.true) for s in stems], chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    Q = np.concatenate([r[0] for r in res]); E = np.concatenate([r[1] for r in res])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, Q=Q, E=E, levels=LEVELS, extra=np.array(EXTRA), stems=np.array(stems))
    print(f"{len(stems)} shots, {len(Q):,} frames, {time.time() - t0:.0f}s -> {a.out}; ok {np.nanmean(E[:, -1]):.4f}", flush=True)


if __name__ == "__main__":
    main()
