"""li calibration: move the scorer's li of each predicted map to a target value without moving its flux surfaces.

Same transform as liguard.py (ml/ch1), applied inside the predicted LCFS:
    phi' = L + (phi_a - L) h(x),  x = (phi - L) / (phi_a - L),  h(x) = x - g (x^2 - x^3 / 3)
h(0) = 0 and h'(0) = 1 (flux and gradient continuous at the LCFS, boundary field and LCFS unchanged), h''(1) = 0 (affine at
the axis to second order). g > 0 lowers the interior poloidal field (li down), g < 0 raises it (li up). g is found by
bisection so that the scorer's li of the new map equals the target. The change is kept only if li is finite, the axis peak
cell is unchanged and the LCFS moved < 1 mm (checked with the scorer's own extraction).

Targets:
  --li-src DIR [DIR ...]   prediction dirs whose files carry a predicted "li" (scalar head); mean over the dirs;
                           target = li_map + alpha (li_src - li_map), only where |li_src / li_map - 1| > --min-rel
  --oracle SEED            target = the scorer's li of the TRUE map (holdout split SEED; diagnostic upper bound)
Frames whose map li is > --li-max are always pulled to --li-guard (the li range guard).

    python lical.py --preds runs/<ens>/preds --out runs/<ens>_lc/preds --li-src runs/<a>/preds runs/<b>/preds --alpha 1.0
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
_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = _M["grid_R"].astype(np.float64), _M["grid_Z"].astype(np.float64)
MC, MF = _M["mask_coarse"].astype(bool), _M["mask_coarse"].astype(np.float64)
RR, ZZ = np.meshgrid(R, Z)
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"


class FastLi:
    """li of maps sharing one contour C (inside mask and boundary sampling precomputed), = derive.internal_inductance."""

    def __init__(self, C):
        from derive import _points_in_polygon
        self.C = C
        self.ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
        dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
        self.dV = 2.0 * np.pi * RR * dR * dZ
        Cc = np.vstack([C, C[0]])
        self.seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1))

    def __call__(self, psi):
        from derive import _bilinear, _poloidal_field
        Bp = _poloidal_field(psi, R, Z, "DIII-D")
        Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
        if self.ins.sum() < 4:
            return np.nan
        num = float(np.sum(Bp2[self.ins] * self.dV[self.ins]))
        V = float(np.sum(self.dV[self.ins]))
        b = _bilinear(Bp, R, Z, self.C[:, 0], self.C[:, 1])
        G = float(np.sum(0.5 * (b + np.roll(b, -1)) * self.seg))
        lp = float(np.sum(self.seg))
        if G <= 0 or lp <= 0 or V <= 0:
            return np.nan
        return (num / V) / (G / lp) ** 2


def lcfs_and_level(phi, n_iter=22, n_points=512):
    """Copy of lcfs._extract_one_sign (phi oriented: axis = max) that also returns the level and the axis cell."""
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


def transform(psi, ins, L, pa, g, s):
    phi = s * psi
    x = (phi - L) / (pa - L)
    m = ins & (x > 0)
    xm = np.clip(x[m], 0.0, 1.0)
    out = phi.copy()
    out[m] = L + (pa - L) * (xm - g * (xm ** 2 - xm ** 3 / 3.0))
    return s * out


def calib_frame(psi, target_fn, g_lo=-1.5, g_hi=0.95):
    """-> (new psi or None, info). target_fn(li_map) -> target li or None (no change)."""
    from common import AXIS_SIGN
    from contour import symmetric_hausdorff
    s = AXIS_SIGN["DIII-D"]
    C, L, iz, ir = lcfs_and_level(s * psi)
    if C is None:
        return None, {"why": "no lcfs"}
    fl = FastLi(C)
    li0 = fl(psi)
    if not np.isfinite(li0):
        return None, {"why": "li0 nan"}
    tgt = target_fn(li0)
    if tgt is None:
        return None, {"li0": li0}
    pa = float(s * psi[iz, ir])
    lo, hi = g_lo, g_hi
    f_lo = fl(transform(psi, fl.ins, L, pa, lo, s))
    f_hi = fl(transform(psi, fl.ins, L, pa, hi, s))
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None, {"li0": li0, "why": "bracket nan"}
    if tgt >= f_lo:
        g = lo
    elif tgt <= f_hi:
        g = hi
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if fl(transform(psi, fl.ins, L, pa, mid, s)) > tgt:
                lo = mid
            else:
                hi = mid
        g = 0.5 * (lo + hi)
    new = transform(psi, fl.ins, L, pa, g, s).astype(np.float32).astype(np.float64)
    C1, L1, iz1, ir1 = lcfs_and_level(s * new)
    info = {"li0": li0, "target": tgt, "g": g}
    if C1 is None or (iz1, ir1) != (iz, ir):
        info["kept"] = False
        return None, info
    sh = 1e3 * symmetric_hausdorff(C, C1)
    li1 = FastLi(C1)(new)
    info.update({"li1": li1, "lcfs_shift_mm": sh, "dpsi_rms": float(np.sqrt(np.mean((new - psi) ** 2)))})
    ok = np.isfinite(li1) and 0.1 < li1 < 5.0 and sh < 1.0
    info["kept"] = bool(ok)
    return (new.astype(np.float32) if ok else None), info


def one_shot(args):
    src, dst, li_src, li_true, alpha, min_rel, li_max, li_guard, max_rel = args
    p = dict(np.load(src))
    psi = p["psi"].astype(np.float64)
    infos = []
    for k in range(len(psi)):
        def target_fn(li0, k=k):
            t = None
            if li_true is not None:
                t = li_true[k] if np.isfinite(li_true[k]) else None
            elif li_src is not None and np.isfinite(li_src[k]):
                rel = li_src[k] / li0 - 1.0
                if abs(rel) >= min_rel:
                    t = li0 * (1.0 + alpha * float(np.clip(rel, -max_rel, max_rel)))
            if t is None:
                return li_guard if li0 > li_max else None
            return min(t, li_guard) if t > li_max else t
        new, info = calib_frame(psi[k], target_fn)
        if info is not None and "target" in info:
            info["k"] = k
            infos.append(info)
        if new is not None:
            p["psi"][k] = new
    np.savez(dst, **p)
    return Path(src).stem, infos


def map_li(src):
    """Scorer li of every frame of a prediction file (NaN where the LCFS extraction fails)."""
    from common import AXIS_SIGN
    s = AXIS_SIGN["DIII-D"]
    psi = np.load(src)["psi"].astype(np.float64)
    out = np.full(len(psi), np.nan)
    for k in range(len(psi)):
        C, _, _, _ = lcfs_and_level(s * psi[k])
        if C is not None:
            out[k] = FastLi(C)(psi[k])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--li-src", type=Path, nargs="*", default=[])
    ap.add_argument("--oracle", type=int, default=None, help="holdout split seed: target = true scorer li")
    ap.add_argument("--targets", type=Path, default=None, help="(ch1k) dir of per-shot target files with an absolute 'li' (T,) "
                    "array (NaN = no target; the li guard still applies)")
    ap.add_argument("--li-ref", type=Path, default=None, help="reference preds dir whose map li anchors the blend targets")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-rel", type=float, default=0.0)
    ap.add_argument("--max-rel", type=float, default=0.5)
    ap.add_argument("--li-max", type=float, default=5.0)
    ap.add_argument("--li-guard", type=float, default=4.9)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.preds.glob("*.npz"))
    true_li = {}
    if a.oracle is not None:
        with open(f"{FEQ_CACHE}/nn/holdout_refs_holdout10_seed{a.oracle}.pkl", "rb") as fh:
            c = pickle.load(fh)
        true_li = {s: np.where(r[2][:, 6], r[1][:, 6], np.nan) for s, r in zip(c["stems"], c["refs"])}
    t0 = time.time()
    rep = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        ref = {}
        if a.li_ref is not None:
            # blend targets from the REFERENCE map's li (e.g. the ensemble map before shape / axis warps), applied as absolute
            # targets to the current map
            for f, v in zip(files, pool.map(map_li, [str(a.li_ref / f.name) for f in files], chunksize=2)):
                ref[f.stem] = v
        jobs = []
        for f in files:
            ls_ = None
            if a.li_src:
                ls_ = np.mean([np.load(d / f.name)["li"].astype(np.float64) for d in a.li_src], 0)
            tl = true_li.get(f.stem)
            if a.targets is not None:
                tl = np.load(a.targets / f.name)["li"].astype(np.float64)
            if f.stem in ref and ls_ is not None:
                lr = ref[f.stem]
                tl = lr * (1.0 + a.alpha * np.clip(ls_ / lr - 1.0, -a.max_rel, a.max_rel))
            jobs.append((str(f), str(a.out / f.name), ls_, tl, a.alpha, a.min_rel, a.li_max, a.li_guard, a.max_rel))
        for stem, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
    n_t = sum(len(v) for v in rep.values())
    n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "lical_report.json").write_text(json.dumps(rep, default=float))
    print(f"{len(files)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
