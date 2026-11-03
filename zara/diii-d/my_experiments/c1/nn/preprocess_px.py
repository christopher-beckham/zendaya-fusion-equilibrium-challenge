"""Build the Group A physics-feature cache (physics_plus.group_a) for a split, one shard at a time.

    python preprocess_px.py --split diii_d_train --shard 0 --nshards 8 --workers 4
→ $FEQ_CACHE/nn/px/<split>/<stem>.npz  (Xp (T, F) float32, pos (72, 2), te (T, 72), ne (T, 72))
   $FEQ_CACHE/nn/px/<split>/meta.json (feature names; written by shard 0)
"""
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root
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
DATA = HERE.parents[2] / "challenge_data" / "data"
OUT = Path(f"{FEQ_CACHE}/nn/px")


def one(args):
    f, out_dir = args
    sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
    import pandas as pd
    from features import shot_inputs
    from physics_plus import group_a
    out = Path(out_dir) / (Path(f).stem + ".npz")
    if out.exists():
        return None
    row = pd.read_parquet(f).iloc[0].to_dict()
    inp = shot_inputs(row)
    X, names, ts = group_a(row, inp)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, Xp=X.astype(np.float32), pos=ts["pos"].astype(np.float32), te=ts["te"].astype(np.float32),
             ne=ts["ne"].astype(np.float32))
    os.replace(tmp, out)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="diii_d_train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    files = [str(f) for f in sorted((DATA / a.split).glob("*.parquet"))][a.shard::a.nshards]
    out_dir = OUT / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = None
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, nm in enumerate(pool.map(one, [(f, str(out_dir)) for f in files], chunksize=4), 1):
            names = nm or names
            if i % 200 == 0 or i == len(files):
                print(f"  shard {a.shard}: {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    if names is not None and not (out_dir / "meta.json").exists():
        (out_dir / "meta.json").write_text(json.dumps({"names": names}, indent=1))
    print(f"shard {a.shard} done: {len(files)} shots in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
