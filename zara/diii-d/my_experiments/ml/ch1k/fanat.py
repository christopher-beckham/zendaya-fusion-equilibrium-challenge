"""Per-frame scorer functionals, LCFS extremes and LCFS distance of a holdout prediction set (one LCFS extraction per frame).

    python fanat.py --seed 0 --preds DIR [DIR ...] --names N [N ...] --outdir $FEQ_CACHE/ml/ch1k/fa_s0

For every held-out frame of split SEED and every prediction dir: the seven consistency scalars of the map (scorer code:
extract_lcfs_with_sign + derive_frame, li always derived), the LCFS extremes (Zmax, Zmin, Rmax, Rmin, R at Zmax, R at Zmin)
of the same contour, and the per-frame symmetric Hausdorff distance to the reference contour divided by the shot's rgeo
(capped at 1, as the scorer). Also written once per split (meta.npz): frame metadata (shot, k, T, t, Ip, B_T), the cached
reference functionals and the reference extremes. Labels are used only to score (truth side), never to change a map.
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
CACHE = Path(f"{FEQ_CACHE}/nn/diii_d_train")
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"
_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GRID = (_M["grid_R"].astype(np.float64), _M["grid_Z"].astype(np.float64), _M["mask_coarse"].astype(bool),
        _M["mask_coarse"].astype(np.float64))


def split_stems(frac, seed):
    stems = [f.stem for f in sorted(DATA_DIR.glob("*.parquet"))]
    perm = np.random.RandomState(seed).permutation(len(stems))
    hold = set(perm[: int(round(frac * len(stems)))].tolist())
    return [s for i, s in enumerate(stems) if i in hold]


def geo(C):
    return np.array([C[:, 1].max(), C[:, 1].min(), C[:, 0].max(), C[:, 0].min(),
                     C[np.argmax(C[:, 1]), 0], C[np.argmin(C[:, 1]), 0]])


def one_shot(args):
    path, ref = args
    from common import AXIS_SIGN, CONS_SCALARS
    from contour import symmetric_hausdorff
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    psi = np.load(path)["psi"].astype(np.float64)
    contours, cons_gt, cmask, rgeo = ref
    T = psi.shape[0]
    cp = np.full((T, 7), np.nan)
    gp = np.full((T, 6), np.nan)
    dl = np.full(T, np.nan)
    s = AXIS_SIGN["DIII-D"]
    for k in range(T):
        ex, _ = extract_lcfs_with_sign(psi[k], GRID[0], GRID[1], "DIII-D", GRID[2], GRID[3], n_iter=22, axis_sign=s)
        if ex is not None:
            gp[k] = geo(ex)
        if contours[k] is not None:
            dl[k] = 1.0 if ex is None else min(1.0, symmetric_hausdorff(ex, np.asarray(contours[k], np.float64)) / rgeo)
        v = derive_frame(psi[k], GRID[0], GRID[1], "DIII-D", GRID[2], GRID[3], contour=ex, with_li=True, axis_sign=s)
        cp[k] = [v[c] for c in CONS_SCALARS]
    return cp, gp, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    assert len(a.preds) == len(a.names)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.outdir.mkdir(parents=True, exist_ok=True)
    stems = split_stems(0.10, a.seed)
    with open(f"{FEQ_CACHE}/nn/holdout_refs_holdout10_seed{a.seed}.pkl", "rb") as fh:
        c = pickle.load(fh)
    assert c["stems"] == stems
    refs = c["refs"]
    meta = a.outdir / "meta.npz"
    if not meta.exists():
        rec = {k: [] for k in ("shot", "k", "T", "t", "ip", "bt", "q95", "betaN")}
        for i, s in enumerate(stems):
            d = np.load(CACHE / f"{s}.npz")
            T = len(d["t"])
            rec["shot"].append(np.full(T, i)); rec["k"].append(np.arange(T)); rec["T"].append(np.full(T, T))
            for key in ("t", "ip", "bt", "q95", "betaN"):
                rec[key].append(d[key])
        out = {k: np.concatenate(v) for k, v in rec.items()}
        out["cons_gt"] = np.concatenate([np.where(r[2], r[1], np.nan) for r in refs])
        out["geo_gt"] = np.concatenate([np.stack([geo(np.asarray(C)) if C is not None else np.full(6, np.nan) for C in r[0]])
                                        for r in refs])
        out["rgeo"] = np.concatenate([np.full(len(r[1]), r[3]) for r in refs])
        out["stems"] = np.array(stems)
        tmp = meta.with_name(f"meta.{os.getpid()}.tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, meta)
        print(f"meta: {len(out['shot']):,} frames -> {meta}", flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for p, n in zip(a.preds, a.names):
            o = a.outdir / f"{n}.npz"
            if o.exists():
                print(f"skip {o}", flush=True)
                continue
            t0 = time.time()
            res = list(pool.map(one_shot, [(str(p / f"{s}.npz"), refs[i]) for i, s in enumerate(stems)], chunksize=2))
            np.savez(o, cons_p=np.concatenate([r[0] for r in res]), geo_p=np.concatenate([r[1] for r in res]),
                     dl=np.concatenate([r[2] for r in res]), src=str(p))
            print(f"{n}: {time.time() - t0:.0f}s -> {o}", flush=True)


if __name__ == "__main__":
    main()
