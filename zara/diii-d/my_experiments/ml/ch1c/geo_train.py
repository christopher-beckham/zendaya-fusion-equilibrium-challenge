"""LCFS bounding geometry of the TRUE maps (scorer's extraction) for every DIII-D training shot (label-side, official data).

Per frame: Zmax, Zmin, Rmax, Rmin, R at Zmax, R at Zmin of the scorer's LCFS (local_score: extract_lcfs on the true map,
512 points) -> $FEQ_CACHE/ml/ch1c/geo_train/<stem>.npz (geo (T, 6), NaN where the extraction fails).

    python geo_train.py --shard 0 --n-shards 4
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
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
CACHE = Path(f"{FEQ_CACHE}/nn/diii_d_train")
OUT = Path(f"{FEQ_CACHE}/ml/ch1c/geo_train")
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"
GEO = ["Zmax", "Zmin", "Rmax", "Rmin", "R_Zmax", "R_Zmin"]


def geo(C):
    if C is None or len(C) < 8:
        return np.full(6, np.nan)
    return np.array([C[:, 1].max(), C[:, 1].min(), C[:, 0].max(), C[:, 0].min(), C[np.argmax(C[:, 1]), 0], C[np.argmin(C[:, 1]), 0]])


def one(stem):
    from lcfs import extract_lcfs
    _M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z, mc, mf = _M["grid_R"], _M["grid_Z"], _M["mask_coarse"].astype(bool), _M["mask_coarse"].astype(np.float64)
    out = OUT / f"{stem}.npz"
    if out.exists():
        return stem, 0
    psi = np.load(CACHE / f"{stem}.npz")["psi"].astype(np.float32)
    g = np.stack([geo(extract_lcfs(p, R, Z, "DIII-D", mc, mf, n_points=512)) for p in psi])
    np.savez(out.with_suffix(".tmp.npz"), geo=g)
    os.replace(out.with_suffix(".tmp.npz"), out)
    return stem, len(psi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stems = [f.stem for f in sorted(DATA_DIR.glob("*.parquet"))][a.shard::a.n_shards]
    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (s, k) in enumerate(pool.map(one, stems, chunksize=2)):
            n += k
            if i % 400 == 0:
                print(f"{i}/{len(stems)} shots, {n:,} frames, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(stems)} shots, {n:,} new frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
