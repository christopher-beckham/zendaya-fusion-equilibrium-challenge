"""Label-free temporal post-processing of per-shot predictions (psi (T, nZ, nR), q95 (T,), betaN (T,)).

    python postproc.py --src <preds dir> --dst <out dir> [--window 3] [--outlier 0.15] [--machine MAST|DIII-D] [--workers 16]

Per shot, in time order:
  1. outlier frames: the magnetic axis (scorer's O-point search on the geometry mask) jumps more than --outlier metres from the
     median axis of its +-2 neighbours -> the frame's psi is replaced by the mean of its nearest non-outlier neighbours;
  2. pixel-wise running median of psi over --window frames (odd; 1 = off), same for q95 / betaN.
The median of a linear trend is its centre value, so slow ramps are unbiased; single-frame excursions are removed.
Other keys of the npz are copied. Writes <dst>/preds/<stem>.npz and <dst>/stems.txt, prints the outlier count.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

ROOT = Path(__file__).resolve().parents[4]   # the starter folder of this package
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from o_point import find_o_point          # noqa: E402
from derive import derive_frame            # noqa: E402

GRID = {"MAST": Path(f"{R1_ROOT}/cache/mast/mast_public_test/basis.npz"),
        "DIII-D": Path(f"{R1_ROOT}/cache/nn/diii_d_train/basis.npz")}
AXIS_SIGN = {"MAST": 1.0, "DIII-D": -1.0}
_G = {}


def grid(machine):
    if machine not in _G:
        b = np.load(GRID[machine]); R, Z = b["grid_R"].astype(float), b["grid_Z"].astype(float)
        RR, ZZ = np.meshgrid(R, Z)
        if machine == "MAST":
            sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "predict" / "more_data_code"))
            import mast_machine as mm
            from matplotlib.path import Path as MPath
            mask = MPath(mm.limiter_polygon()).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
        else:
            mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")["mask_coarse"].astype(bool)
        _G[machine] = (R, Z, mask)
    return _G[machine]


def axes(psi, machine):
    R, Z, mask = grid(machine)
    out = np.zeros((len(psi), 2))
    for k in range(len(psi)):
        phi = AXIS_SIGN[machine] * psi[k].astype(np.float64)
        phi = np.nan_to_num(phi, nan=np.nanmin(phi) - 1.0)
        Ra, Za, _, _, _ = find_o_point(phi, R, Z, mask)
        out[k] = Ra, Za
    return out


def process(args):
    f, dst, machine, window, thr, kmax, vlo, vhi, fpdir, fp_res_max, flux_jump = args
    d = dict(np.load(f))
    psi = d["psi"].astype(np.float32); T = len(psi)
    n_out = 0
    if thr > 0 and T >= 3:
        ax = axes(psi, machine)
        bad = np.zeros(T, bool)
        # block-robust guards (m2, 2026-10-07): (1) first-pass failure flags from the cache (the models warm-start from the first
        # pass; a collapsed first pass -> collapsed prediction for whole blocks of frames); (2) flux-magnitude jump: the in-mask
        # |psi| range of a frame vs the shot median of the other frames (a plasma collapsed onto the wall has ~10x the flux swing)
        if fpdir is not None:
            c = np.load(Path(fpdir) / f.name)
            if len(c["fp_res"]) == T:
                bad |= (c["fp_res"] > fp_res_max) | ~np.isfinite(c["fp_res"])
        if flux_jump > 0:
            R, Z, mask = grid(machine)
            swing = np.array([np.nanmax(psi[k][mask]) - np.nanmin(psi[k][mask]) for k in range(T)])
            ref = np.nanmedian(swing)
            bad |= (swing > flux_jump * ref) | (swing < ref / flux_jump)
        for k in range(T):
            nb = [j for j in range(max(0, k - 2), min(T, k + 3)) if j != k and not bad[j]]
            if not nb:
                nb = [j for j in range(T) if j != k and not bad[j]] or [j for j in range(T) if j != k]
            med = np.nanmedian(ax[nb], 0)
            if not np.isfinite(ax[k]).all() or np.hypot(*(ax[k] - med)) > thr:
                bad[k] = True
        if kmax > 0:                                   # shape-plausibility guard (machine-size bounds, no equilibrium data)
            R, Z, mask = grid(machine)
            for k in range(T):
                if bad[k]:
                    continue
                o = derive_frame(psi[k], R, Z, machine, mask_coarse=mask, with_li=False, axis_sign=AXIS_SIGN[machine])
                if not (np.isfinite(o["kappa"]) and o["kappa"] <= kmax and vlo <= o["volume"] <= vhi):
                    bad[k] = True
        good = np.where(~bad)[0]
        if len(good) >= 2:
            for k in np.where(bad)[0]:
                j = good[np.argmin(np.abs(good - k))]       # nearest good frame (copy, no blending)
                psi[k] = psi[j]
                for key in ("q95", "betaN"):
                    if key in d:
                        d[key][k] = d[key][j]
            n_out = int(bad.sum())
    if window > 1 and T >= window:
        psi = median_filter(psi, size=(window, 1, 1), mode="nearest")
        for key in ("q95", "betaN"):
            if key in d:
                d[key] = median_filter(d[key].astype(np.float32), size=window, mode="nearest")
    d["psi"] = psi.astype(np.float32)
    np.savez(dst / "preds" / f.name, **d)
    return n_out, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--outlier", type=float, default=0.15, help="metres; 0 = off")
    ap.add_argument("--machine", default="MAST")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--kappa-max", type=float, default=0.0, help="replace frames whose LCFS elongation exceeds this (0 = off)")
    ap.add_argument("--vol-range", type=float, nargs=2, default=(0.0, 1e9), help="replace frames whose plasma volume (m^3) is outside this range")
    ap.add_argument("--fp-cache", type=Path, default=None, help="per-shot cache with first-pass flags (fp_res); frames with fp_res > --fp-res-max are replaced")
    ap.add_argument("--fp-res-max", type=float, default=0.05)
    ap.add_argument("--flux-jump", type=float, default=0.0, help="replace frames whose in-mask psi swing differs from the shot median by more than this factor (0 = off)")
    a = ap.parse_args()
    (a.dst / "preds").mkdir(parents=True, exist_ok=True)
    if (a.src / "stems.txt").exists():
        shutil.copy(a.src / "stems.txt", a.dst / "stems.txt")
    files = sorted((a.src / "preds").glob("*.npz"))
    with Pool(a.workers) as p:
        res = p.map(process, [(f, a.dst, a.machine, a.window, a.outlier, a.kappa_max, a.vol_range[0], a.vol_range[1], a.fp_cache, a.fp_res_max, a.flux_jump) for f in files], chunksize=4)
    n_out = sum(r[0] for r in res); n = sum(r[1] for r in res)
    print(f"{a.src} -> {a.dst}: {len(files)} shots, {n} frames, outliers replaced {n_out} ({n_out / max(n, 1):.3%}), window {a.window}")


if __name__ == "__main__":
    main()
