"""Scorer functionals of the TRUE flux maps of every DIII-D training shot (label-side only, official data).

For each shot: the seven consistency scalars exactly as the scorer derives them from the ground-truth map
(local_score.build_reference: extract_lcfs + derive_frame; NaN where the derivation fails) and rgeo.
Output: $FEQ_CACHE/ml/ch1c/cons_train/<stem>.npz with cons (T, 7) float64, rgeo.

    python ref_train.py --shard 0 --n-shards 4 [--workers 8]
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
CACHE = Path(f"{FEQ_CACHE}/nn/diii_d_train")
OUT = Path(f"{FEQ_CACHE}/ml/ch1c/cons_train")
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"


def one(stem):
    import local_score as ls
    _M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = (_M["grid_R"], _M["grid_Z"], _M["mask_coarse"].astype(bool), _M["mask_coarse"].astype(np.float64))
    out = OUT / f"{stem}.npz"
    if out.exists():
        return stem, 0
    psi = np.load(CACHE / f"{stem}.npz")["psi"].astype(np.float32)
    _, cons, cmask, rgeo = ls.build_reference(psi, *grid)
    np.savez(out.with_suffix(".tmp.npz"), cons=np.where(cmask, cons, np.nan), rgeo=rgeo)
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
            if i % 200 == 0:
                print(f"{i}/{len(stems)} shots, {n:,} frames, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(stems)} shots, {n:,} new frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
