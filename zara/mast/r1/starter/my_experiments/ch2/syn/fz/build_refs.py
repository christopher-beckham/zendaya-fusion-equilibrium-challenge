"""Scorer functionals on the TRUE DIII-D flux maps (the targets of the soft scorer-functional losses).

    python build_refs.py --shard 0 --n-shards 2 --workers 4

Runs the competition scorer's own build_reference (fusion_scoring: O-point, LCFS bisection, derive_frame) on every
frame of every DIII-D training shot -> $R1_ROOT/cache/pegs/refs_d3d/<stem>.npz with
    cons (T, 7) R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li (NaN where the scorer fails)
    lcfs (T, 64, 2) the scorer's LCFS resampled to 64 points (NaN rows where it fails)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = Path(f"{R1_ROOT}/cache/nn/diii_d_train")
OUT = Path(f"{R1_ROOT}/cache/pegs/refs_d3d")


def one(stem):
    sys.path.insert(0, str(ROOT))
    import local_score as ls
    M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = (M["grid_R"], M["grid_Z"], M["mask_coarse"].astype(bool), M["mask_coarse"].astype(np.float64))
    d = np.load(CACHE / f"{stem}.npz")
    contours, cons, ok, _ = ls.build_reference(d["psi"].astype(np.float32), *grid)
    lc = np.full((len(contours), 64, 2), np.nan, np.float32)
    for k, c in enumerate(contours):
        if c is not None:
            idx = np.linspace(0, len(c), 64, endpoint=False).astype(int)
            lc[k] = c[idx]
    tmp = OUT / f"{stem}.tmp.npz"
    np.savez(tmp, cons=np.where(ok, cons, np.nan).astype(np.float32), lcfs=lc)
    os.replace(tmp, OUT / f"{stem}.npz")
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-shots", type=int, default=0)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in CACHE.glob("*.npz") if p.stem not in ("basis", "meta"))[: args.max_shots or None]
    stems = [s for s in stems[args.shard::args.n_shards] if not (OUT / f"{s}.npz").exists()]
    print(f"{len(stems)} shots to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, _ in enumerate(ex.map(one, stems, chunksize=4)):
            if i % 200 == 0:
                print(f"  {i}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
