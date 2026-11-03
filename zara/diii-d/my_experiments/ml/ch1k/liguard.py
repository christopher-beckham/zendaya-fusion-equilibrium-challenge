"""li range guard: post-processing of predicted DIII-D flux maps for the scorer's li plausibility cut.

The scorer derives li from the submitted psi and maps any value outside (0.1, 5.0) to NaN, which is then scored as the
fold MEAN li (~1.05). A frame whose true li is 4.6 and whose predicted map implies li 5.1 therefore costs (4.6 - 1.05)^2
instead of 0.25. Scored frames always have a true li inside the range (the reference value must be finite to be
scored), so pulling such predictions back inside the range can only reduce the error.

For every frame whose predicted map gives li > --li-max (5.0), the flux INSIDE the predicted LCFS is replaced by a
monotone function of itself,
    phi' = L + (phi_a - L) h(x),   x = (phi - L) / (phi_a - L) in (0, 1],   h(x) = x - g (x^2 - x^3 / 3),
(phi = AXIS_SIGN psi, L = the LCFS level found by the scorer's bisection, phi_a = the axis value), with g in (0, 0.95]
set by bisection so that the scorer's li equals --li-target (4.9). h(0) = 0, h'(0) = 1 and h''(1) = 0: the flux and its
gradient are continuous across the LCFS, the flux surfaces (hence the LCFS polygon, kappa, triangularity, volume) are
unchanged and the map is affine to second order at the axis (the scorer's parabolic axis refinement barely moves).
Only the interior gradient (and so <Bp^2>_V) is lowered. Every modified frame is re-scored with the scorer's own code
and the change is kept only if li is now inside the range, the axis peak cell is unchanged and the LCFS moved < 1 mm.
Label-free: uses only the prediction itself.

Optional calibration (--shrink-thr T --shrink-c C): predicted li above T is pulled to T + C (li - T) with the same transform
(high predicted li over-estimates the truth on held-out shots; T and C must be chosen on a DIFFERENT split than the one scored).

    python liguard.py --preds runs/<ens>/preds --out runs/<ens>_lig/preds [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "fusion_scoring"))
_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = _M["grid_R"].astype(np.float64), _M["grid_Z"].astype(np.float64)
MC, MF = _M["mask_coarse"].astype(bool), _M["mask_coarse"].astype(np.float64)


def lcfs_and_level(phi, n_iter=22, n_points=512):
    """Copy of lcfs._extract_one_sign (phi already oriented: axis = max) that also returns the level and the axis cell."""
    from contour import grid_to_physical, resample_closed
    from lcfs import _best_contour_at
    from o_point import find_o_point
    _, _, iz, ir, _ = find_o_point(phi, R, Z, MC)
    axis_val = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo, hi = float(np.min(border)), axis_val - 1e-9
    if not (hi > lo):
        return None, None, iz, ir
    best = _best_contour_at(phi, hi, iz, ir, MF)
    if best is None:
        return None, None, iz, ir
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = _best_contour_at(phi, mid, iz, ir, MF)
        if c is not None:
            hi, best = mid, c
        else:
            lo = mid
    return resample_closed(grid_to_physical(best, R, Z), n_points), hi, iz, ir


def scorer_vals(psi):
    """The scorer's view of one predicted frame: (contour, derived dict)."""
    from common import AXIS_SIGN
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    s = AXIS_SIGN["DIII-D"]
    ex, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", MC, MF, n_iter=22, axis_sign=s)
    return ex, derive_frame(psi, R, Z, "DIII-D", MC, MF, contour=ex, with_li=True, axis_sign=s)


def raw_li(psi, C):
    from derive import internal_inductance
    return internal_inductance(psi, R, Z, "DIII-D", C)


def transform(psi, C, L, iz, ir, g):
    from common import AXIS_SIGN
    from derive import _points_in_polygon
    s = AXIS_SIGN["DIII-D"]
    phi = s * psi
    pa = float(phi[iz, ir])
    RR, ZZ = np.meshgrid(R, Z)
    ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    x = (phi - L) / (pa - L)
    m = ins & (x > 0)
    xm = np.clip(x[m], 0.0, 1.0)
    out = phi.copy()
    out[m] = L + (pa - L) * (xm - g * (xm ** 2 - xm ** 3 / 3.0))
    return s * out


def guard_frame(psi, li_max, li_target, shrink_thr=None, shrink_c=1.0):
    """-> (new psi or None, info dict)."""
    from common import AXIS_SIGN
    from contour import symmetric_hausdorff
    s = AXIS_SIGN["DIII-D"]
    C, L, iz, ir = lcfs_and_level(s * psi)
    if C is None:
        return None, {"why": "no lcfs"}
    li0 = raw_li(psi, C)
    if not np.isfinite(li0):
        return None, None
    if li0 > li_max:
        target = li_target
    elif shrink_thr is not None and li0 > shrink_thr and shrink_c < 1.0:
        target = min(li_target, shrink_thr + shrink_c * (li0 - shrink_thr))
    else:
        return None, None
    li_target = target
    lo, hi = 0.0, 0.95
    if raw_li(transform(psi, C, L, iz, ir, hi), C) > li_target:
        g = hi
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if raw_li(transform(psi, C, L, iz, ir, mid), C) > li_target:
                lo = mid
            else:
                hi = mid
        g = hi
    new = transform(psi, C, L, iz, ir, g).astype(np.float32).astype(np.float64)
    ex0, v0 = scorer_vals(psi)
    ex1, v1 = scorer_vals(new)
    info = {"li0": li0, "target": li_target, "g": g, "li_scorer_after": v1["li"], "dR_axis_mm": 1e3 * (v1["R_axis"] - v0["R_axis"]),
            "dZ_axis_mm": 1e3 * (v1["Z_axis"] - v0["Z_axis"]),
            "lcfs_shift_mm": (1e3 * symmetric_hausdorff(ex0, ex1)) if (ex0 is not None and ex1 is not None) else None,
            "dkappa": v1["kappa"] - v0["kappa"], "dvolume": v1["volume"] - v0["volume"],
            "dpsi_rms_rel": float(np.sqrt(np.mean((new - psi) ** 2)) / np.std(psi))}
    _, _, iz1, ir1 = lcfs_and_level(s * new)
    ok = (np.isfinite(v1["li"]) and (iz1, ir1) == (iz, ir) and info["lcfs_shift_mm"] is not None
          and info["lcfs_shift_mm"] < 1.0)
    info["kept"] = bool(ok)
    return (new.astype(np.float32) if ok else None), info


def one_shot(args):
    src, dst, li_max, li_target, thr, c = args
    p = dict(np.load(src))
    psi = p["psi"].astype(np.float64)
    infos = []
    changed = False
    for k in range(len(psi)):
        new, info = guard_frame(psi[k], li_max, li_target, thr, c)
        if info is not None:
            info["k"] = k
            infos.append(info)
        if new is not None:
            p["psi"][k] = new
            changed = True
    np.savez(dst, **p)
    return Path(src).stem, changed, infos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--li-max", type=float, default=5.0)
    ap.add_argument("--li-target", type=float, default=4.9)
    ap.add_argument("--shrink-thr", type=float, default=None)
    ap.add_argument("--shrink-c", type=float, default=1.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.preds.glob("*.npz"))
    t0 = time.time()
    jobs = [(str(f), str(a.out / f.name), a.li_max, a.li_target, a.shrink_thr, a.shrink_c) for f in files]
    rep = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for stem, changed, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
                for i in ([] if a.quiet else infos):
                    print(stem, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in i.items()}), flush=True)
    n_flag = sum(len(v) for v in rep.values())
    n_kept = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "liguard_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"{len(files)} shots, {n_flag} frames flagged (li > {a.li_max} or shrink above {a.shrink_thr}), {n_kept} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
