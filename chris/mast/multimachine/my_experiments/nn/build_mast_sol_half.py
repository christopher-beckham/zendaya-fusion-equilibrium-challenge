"""MAST inputs with the solenoid current halved, written next to the existing ones (nothing is overwritten).

The organisers' data notes: magnetics_sol_current reads exactly 2x the FAIR-MAST level-2 pf_active solenoid current,
and which convention is right is open. Halving the solenoid picture is the same as halving that current, since the
picture is linear in it. Writes:
  <cache>/features_v2/pictures_mast_sol0.5.npz
  <cache>/features_v5_mast_public_test_sol0.5.npy and .parquet (same rows as features_v5_mast_public_test)
With FEC_MAST_MASK=limiter (mast_region.py) the features use the limiter region's Z0 and go to
  <cache>/features_v5_mast_public_test_sol0.5_limiter.npy and .parquet (shards in .../mast_public_test_sol0.5_limiter);
the pictures do not depend on the region and are left as they are.
Inputs only; no MAST label is read.

usage: python build_mast_sol_half.py [--workers 4]
       FEC_MAST_MASK=limiter python build_mast_sol_half.py [--workers 4]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import _join, _worker

CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
MAST = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data" / SPLIT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import mast_region
    sfx = mast_region.suffix()
    print(f"MAST region: {mast_region.name()}", flush=True)
    src = CACHE / "features_v2" / "pictures_mast.npz"
    dst = CACHE / "features_v2" / "pictures_mast_sol0.5.npz"
    if not (sfx and dst.exists()):
        with np.load(src) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["magnetics_sol_current"] = arrays["magnetics_sol_current"] * 0.5
        np.savez(dst, **arrays)
    paths = sorted(MAST.glob("*.parquet"))
    shard_dir = CACHE / "features_v5" / "shards" / f"{SPLIT}_sol0.5{sfx}"
    jobs = [(str(p), str(dst), str(shard_dir / f"{p.stem}.npy"), "v5") for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, err) in enumerate(ex.map(_worker, jobs, chunksize=4), start=1):
            if err:
                raise SystemExit(f"{name}: {err}")
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    feats, ns = _join(paths, None, shard_dir, CACHE / f"features_v5_{SPLIT}_sol0.5{sfx}.npy")
    ref = pd.read_parquet(CACHE / f"features_v5_{SPLIT}.parquet")
    starts = np.concatenate([[0], np.cumsum(ns)[:-1]])
    out = pd.DataFrame({"shot": [p.stem for p in paths], "filename": [p.name for p in paths], "start": starts, "n": ns})
    if not (out[["filename", "start", "n"]].values == ref[["filename", "start", "n"]].values).all():
        raise SystemExit(f"row layout differs from features_v5_{SPLIT}")
    out.to_parquet(CACHE / f"features_v5_{SPLIT}_sol0.5{sfx}.parquet")
    old = np.load(CACHE / (f"features_v5_{SPLIT}_sol0.5.npy" if sfx else f"features_v5_{SPLIT}.npy"))
    import features_v5 as f5
    d = np.nanmedian(np.abs(feats - old), axis=0)
    top = np.argsort(-np.nan_to_num(d))[:8]
    print(f"wrote {len(feats)} rows. Largest median |change| per column: "
          + ", ".join(f"{f5.FEATURES[i]} {d[i]:.3g}" for i in top), flush=True)
