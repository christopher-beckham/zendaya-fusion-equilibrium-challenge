"""Shape calibration: move the LCFS extremes of a predicted map (top point, bottom / X-point, outboard and inboard
midplane points) toward targets with a smooth local warp of the map.

psi'(r) = psi(r - u(r)),  u(r) = sum_k exp(-|r - P_k|^2 / 2 sigma^2) d_k,  k in {top, bottom, outboard, inboard}
P_k = the extreme points of the map's own LCFS (scorer extraction), d_k = displacement toward the target
(top / bottom: (dR, dZ) of the point (R at Zmax, Zmax) / (R at Zmin, Zmin); outboard / inboard: dR of Rmax / Rmin).
d_k is refined by a few fixed-point steps on the scorer's own LCFS of the warped map. The magnetic axis (>= ~0.4 m from
every P_k) is essentially untouched; values at the displaced points come from a bicubic spline of the map. Kept only if the
targeted extremes moved closer to the targets and the axis moved < 1 mm.

Targets: map extremes + alpha (head - map) with head = the mean "geo" output (Zmax, Zmin, Rmax, Rmin, R at Zmax, R at Zmin)
of --src prediction dirs (--aux-geo networks), or the truth (--oracle SEED, diagnostic: extremes of the reference contours).

    python shapecal.py --preds runs/<ens>/preds --out runs/<x>/preds --src runs/<a>/preds ... --alpha 0.5
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "fusion_scoring"))
sys.path.insert(0, str(HERE))
_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = _M["grid_R"].astype(np.float64), _M["grid_Z"].astype(np.float64)
MC, MF = _M["mask_coarse"].astype(bool), _M["mask_coarse"].astype(np.float64)
RR, ZZ = np.meshgrid(R, Z)


def geo(C):
    return np.array([C[:, 1].max(), C[:, 1].min(), C[:, 0].max(), C[:, 0].min(), C[np.argmax(C[:, 1]), 0], C[np.argmin(C[:, 1]), 0]])


def points(C):
    """top, bottom, outboard, inboard extreme points (R, Z)."""
    return (C[np.argmax(C[:, 1])], C[np.argmin(C[:, 1])], C[np.argmax(C[:, 0])], C[np.argmin(C[:, 0])])


def lcfs(psi):
    from common import AXIS_SIGN
    from lcfs import extract_lcfs_with_sign
    C, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", MC, MF, n_iter=22, axis_sign=AXIS_SIGN["DIII-D"])
    return C


def warp(spl, P, D, sigma):
    uR = np.zeros_like(RR)
    uZ = np.zeros_like(RR)
    for p, d in zip(P, D):
        w = np.exp(-((RR - p[0]) ** 2 + (ZZ - p[1]) ** 2) / (2 * sigma ** 2))
        uR += w * d[0]
        uZ += w * d[1]
    return spl.ev(ZZ - uZ, RR - uR)


def cal_frame(psi, tgt, sigma=0.15, n_iter=4, max_d=0.04, tol=3e-4):
    from derive import magnetic_axis
    from scipy.interpolate import RectBivariateSpline
    C = lcfs(psi)
    if C is None:
        return None, {"why": "no lcfs"}
    g0 = geo(C)
    tm = np.isfinite(tgt)                          # targeted extremes; the others get no displacement
    t = np.where(tm, tgt, g0)
    if np.all(np.abs(t - g0) < tol):
        return None, None
    P = points(C)
    # displacement of each extreme point: top (dR from R@Zmax, dZ from Zmax), bottom (R@Zmin, Zmin), out (Rmax), in (Rmin)
    def resid(g):
        e = np.where(tm, t - g, 0.0)
        return [np.array([e[4], e[0]]), np.array([e[5], e[1]]), np.array([e[2], 0.0]), np.array([e[3], 0.0])]
    D = resid(g0)
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    best, best_err, g_best = None, np.abs(t - g0)[tm].sum(), g0
    for _ in range(n_iter):
        D = [np.clip(d, -max_d, max_d) for d in D]
        new = warp(spl, P, D, sigma)
        C1 = lcfs(new)
        if C1 is None:
            break
        g1 = geo(C1)
        err = np.abs(t - g1)[tm].sum()
        if err < best_err:
            best, best_err, g_best = new, err, g1
        if np.all(np.abs(t - g1) < tol):
            break
        D = [d + r for d, r in zip(D, resid(g1))]
    info = {"g0": g0.tolist(), "t": t.tolist(), "g1": g_best.tolist()}
    if best is None:
        info["kept"] = False
        return None, info
    best = best.astype(np.float32).astype(np.float64)
    a0 = magnetic_axis(psi, R, Z, "DIII-D", MC)[:2]
    a1 = magnetic_axis(best, R, Z, "DIII-D", MC)[:2]
    da = 1e3 * float(np.hypot(a1[0] - a0[0], a1[1] - a0[1]))
    info.update({"axis_move_mm": da, "dpsi_rms": float(np.sqrt(np.mean((best - psi) ** 2)))})
    info["kept"] = bool(da < 1.0)
    return (best.astype(np.float32) if info["kept"] else None), info


def one_shot(args):
    src, dst, tgt, sigma = args
    p = dict(np.load(src))
    infos = []
    for k in range(len(p["psi"])):
        new, info = cal_frame(p["psi"][k].astype(np.float64), tgt[k], sigma=sigma)
        if info is not None:
            info["k"] = k
            infos.append(info)
        if new is not None:
            p["psi"][k] = new
    np.savez(dst, **p)
    return Path(src).stem, infos


def map_geo(args):
    src = args
    psi = np.load(src)["psi"].astype(np.float64)
    out = np.full((len(psi), 6), np.nan)
    for k in range(len(psi)):
        C = lcfs(psi[k])
        if C is not None:
            out[k] = geo(C)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, nargs="*", default=[])
    ap.add_argument("--oracle", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--members", type=Path, nargs="*", default=[], help="(geo mode) member prediction dirs of the ensemble map")
    ap.add_argument("--w-members", type=float, default=0.5, help="weight of the member-mean extremes vs the head mean")
    ap.add_argument("--mode", choices=["geo", "cons"], default="geo",
                    help="geo: targets = blended LCFS extremes ('geo' head outputs); cons: blended kappa / tri_top / tri_bot "
                         "('cons' head outputs) realised by a vertical stretch and shifts of R at Zmax / Zmin")
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--only", default="", help="comma list of extremes to target (Zmax,Zmin,Rmax,Rmin,R_Zmax,R_Zmin); default all")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    names = ["Zmax", "Zmin", "Rmax", "Rmin", "R_Zmax", "R_Zmin"]
    keep = np.array([True] * 6) if not a.only else np.array([n in a.only.split(",") for n in names])
    files = sorted(a.preds.glob("*.npz"))
    if a.mode == "geo":
        a.src = [s for s in a.src if "geo" in np.load(s / files[0].name).files]   # only heads with LCFS-extreme outputs
    if a.oracle is None:
        print(f"shape targets from {len(a.src)} head dirs: {[str(s) for s in a.src]} and {len(a.members)} member maps",
              flush=True)
        assert a.src or a.members, "no head dir with 'geo' and no member maps"
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        if a.oracle is not None:
            with open(f"{FEQ_CACHE}/nn/holdout_refs_holdout10_seed{a.oracle}.pkl", "rb") as fh:
                c = pickle.load(fh)
            tg = {s: np.stack([geo(np.asarray(C)) if C is not None else np.full(6, np.nan) for C in r[0]])
                  for s, r in zip(c["stems"], c["refs"])}
            tgts = [tg[f.stem] for f in files]
        elif a.mode == "geo":
            mg = list(pool.map(map_geo, [str(f) for f in files], chunksize=2))
            if a.members:
                # extremes of each member map (the ensemble-mean map smooths the X-point region: the mean of the members'
                # extremes is a better estimate of kappa / triangularity than the extremes of the mean map)
                memg = [list(pool.map(map_geo, [str(md / f.name) for f in files], chunksize=2)) for md in a.members]
            tgts = []
            for i, (f, m) in enumerate(zip(files, mg)):
                parts, wts = [], []
                if a.src:
                    parts.append(np.mean([np.load(sd / f.name)["geo"].astype(np.float64) for sd in a.src], 0))
                    wts.append(1.0 - a.w_members if a.members else 1.0)
                if a.members:
                    parts.append(np.nanmean(np.stack([mm[i] for mm in memg]), 0))
                    wts.append(a.w_members if a.src else 1.0)
                h = sum(w * p for w, p in zip(wts, parts)) / sum(wts)
                tgts.append(m + a.alpha * (h - m))
        else:
            # mode cons: kappa / tri_top / tri_bot targets from the heads' functionals (blend), realised with the smallest
            # change of the extremes: symmetric vertical stretch for kappa (Zmax, Zmin), R at Zmax / R at Zmin for tri
            mg = list(pool.map(map_geo, [str(f) for f in files], chunksize=2))
            tgts = []
            for f, m in zip(files, mg):
                h = np.mean([np.load(sd / f.name)["cons"][:, 2:5].astype(np.float64) for sd in a.src], 0)
                Zmax, Zmin, Rmax, Rmin, Rt, Rb = m.T
                W, aa = Rmax - Rmin, 0.5 * (Rmax - Rmin)
                Rg = 0.5 * (Rmax + Rmin)
                k0, t0_, b0 = (Zmax - Zmin) / W, (Rg - Rt) / aa, (Rg - Rb) / aa
                dk = a.alpha * (h[:, 0] - k0)
                dt = a.alpha * (h[:, 1] - t0_)
                db = a.alpha * (h[:, 2] - b0)
                tgts.append(np.c_[Zmax + 0.5 * W * dk, Zmin - 0.5 * W * dk, Rmax, Rmin, Rt - aa * dt, Rb - aa * db])
        tgts = [np.where(keep[None], x, np.nan) for x in tgts]
        jobs = [(str(f), str(a.out / f.name), t, a.sigma) for f, t in zip(files, tgts)]
        rep = {}
        for stem, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
    n_t = sum(len(v) for v in rep.values())
    n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "shapecal_report.json").write_text(json.dumps(rep, default=float))
    print(f"{len(files)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
