"""Axis calibration: move the scorer's magnetic axis of a predicted map toward a target without moving its LCFS.

Interior warp of the predicted map: psi'(r) = psi(r - delta s(x(r))), x = normalised flux of the map (0 on its LCFS, 1 at its
axis, 0 outside), s(x) = x^2 (3 - 2 x) (s = 0 and s' = 0 on the LCFS, s = 1 and s' = 0 at the axis): flux surfaces near
the axis are translated by delta, the LCFS and everything outside it are unchanged. Values at the displaced points come
from a bicubic spline of the map. delta is found by a few Newton steps on the scorer's own axis (o_point + parabolic
refinement). Kept only if the LCFS moved < 1 mm and the new axis is closer to the target. Afterwards the li calibration
(lical.py) can be applied (it keeps the axis cell and is affine at the axis).

Targets: the mean of head predictions (prediction files with "cons" (T, 7) from --aux-cons networks),
    R* = R_map + alpha_R (R_head - R_map),  Z* = Z_map + alpha_Z (Z_head - Z_map)
or the truth (--oracle SEED, diagnostic).

    python axcal.py --preds runs/<ens>/preds --out runs/<x>/preds --src runs/<a>/preds ... --alpha-r 0.5 --alpha-z 0.75
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


def axis_of(psi):
    from derive import magnetic_axis
    Ra, Za, iz, ir = magnetic_axis(psi, R, Z, "DIII-D", MC)
    return Ra, Za


def warp(psi, sm, dR, dZ, spl):
    out = psi.copy()
    m = sm > 0
    out[m] = spl.ev(ZZ[m] - dZ * sm[m], RR[m] - dR * sm[m])
    return out


def cal_frame(psi, tgt, max_shift=0.03, n_newton=4):
    """-> (new psi or None, info). tgt = (R*, Z*) (either may be nan = keep)."""
    from common import AXIS_SIGN
    from contour import symmetric_hausdorff
    from derive import _points_in_polygon
    from lical import lcfs_and_level
    from scipy.interpolate import RectBivariateSpline
    s = AXIS_SIGN["DIII-D"]
    C, L, iz, ir = lcfs_and_level(s * psi)
    if C is None:
        return None, {"why": "no lcfs"}
    R0, Z0 = axis_of(psi)
    if len(tgt) == 4:                       # (head R, head Z, alpha R, alpha Z): blend toward the heads from the map's own axis
        hR, hZ, aR, aZ = tgt
        tgt = (R0 + aR * (hR - R0) if np.isfinite(hR) else np.nan, Z0 + aZ * (hZ - Z0) if np.isfinite(hZ) else np.nan)
    tR = R0 if not np.isfinite(tgt[0]) else tgt[0]
    tZ = Z0 if not np.isfinite(tgt[1]) else tgt[1]
    if abs(tR - R0) < 2e-4 and abs(tZ - Z0) < 2e-4:
        return None, None
    ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    phi = s * psi
    pa = float(phi[iz, ir])
    x = np.clip((phi - L) / (pa - L), 0.0, 1.0) * ins
    sm = x * x * (3.0 - 2.0 * x)
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    dR, dZ = tR - R0, tZ - Z0
    new = psi
    for _ in range(n_newton):
        dR, dZ = float(np.clip(dR, -max_shift, max_shift)), float(np.clip(dZ, -max_shift, max_shift))
        new = warp(psi, sm, dR, dZ, spl)
        Rn, Zn = axis_of(new)
        eR, eZ = tR - Rn, tZ - Zn
        if abs(eR) < 2e-4 and abs(eZ) < 2e-4:
            break
        dR, dZ = dR + eR, dZ + eZ
    new = new.astype(np.float32).astype(np.float64)
    Rn, Zn = axis_of(new)
    C1, _, _, _ = lcfs_and_level(s * new)
    info = {"R0": R0, "Z0": Z0, "tR": tR, "tZ": tZ, "R1": Rn, "Z1": Zn, "dR": dR, "dZ": dZ}
    if C1 is None:
        info["kept"] = False
        return None, info
    sh = 1e3 * symmetric_hausdorff(C, C1)
    info["lcfs_shift_mm"] = sh
    closer = np.hypot(tR - Rn, tZ - Zn) < np.hypot(tR - R0, tZ - Z0)
    ok = sh < 1.0 and closer
    info["kept"] = bool(ok)
    return (new.astype(np.float32) if ok else None), info


def one_shot(args):
    src, dst, tgt, max_shift = args
    p = dict(np.load(src))
    infos = []
    for k in range(len(p["psi"])):
        new, info = cal_frame(p["psi"][k].astype(np.float64), tgt[k], max_shift=max_shift)
        if info is not None:
            info["k"] = k
            infos.append(info)
        if new is not None:
            p["psi"][k] = new
    np.savez(dst, **p)
    return Path(src).stem, infos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, nargs="*", default=[])
    ap.add_argument("--oracle", type=int, default=None)
    ap.add_argument("--targets", type=Path, default=None, help="(ch1k) dir of per-shot target files with absolute 'R', 'Z' (T,)")
    ap.add_argument("--alpha-r", type=float, default=0.5)
    ap.add_argument("--alpha-z", type=float, default=0.75)
    ap.add_argument("--max-shift", type=float, default=0.03, help="largest axis displacement per frame (m)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in a.preds.glob("*.npz"))
    if a.oracle is not None:
        with open(f"{FEQ_CACHE}/nn/holdout_refs_holdout10_seed{a.oracle}.pkl", "rb") as fh:
            c = pickle.load(fh)
        tgt_all = {s: np.where(r[2][:, :2], r[1][:, :2], np.nan) for s, r in zip(c["stems"], c["refs"])}
    jobs = []
    for s in stems:
        if a.oracle is not None:
            tgt = tgt_all[s]
        elif a.targets is not None:
            z = np.load(a.targets / f"{s}.npz")
            tgt = np.c_[z["R"].astype(np.float64), z["Z"].astype(np.float64)]
        else:
            head = np.mean([np.load(sd / f"{s}.npz")["cons"][:, :2].astype(np.float64) for sd in a.src], 0)
            tgt = np.c_[head, np.full(len(head), a.alpha_r), np.full(len(head), a.alpha_z)]
        jobs.append((str(a.preds / f"{s}.npz"), str(a.out / f"{s}.npz"), tgt, a.max_shift))
    t0 = time.time()
    rep = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for stem, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
    n_t = sum(len(v) for v in rep.values())
    n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "axcal_report.json").write_text(json.dumps(rep, default=float))
    print(f"{len(jobs)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
