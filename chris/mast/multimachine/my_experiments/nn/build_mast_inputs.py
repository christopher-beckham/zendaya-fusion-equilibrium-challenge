"""Cache MAST public-test decoder inputs for training-time physics losses: mirrored coil currents and Ip per frame.

Rows line up with <cache>/features_v5_mast_public_test.npy (same start and n per shot). Inputs only: no efit_* label
column is read (features_v2.shot_features reads magnetics and Thomson inputs and the EFIT time stamps).

usage: python build_mast_inputs.py [--workers 2]
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import features_v2 as fv

CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
MAST = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data" / SPLIT
OUT = CACHE / ("mast_inputs.npz" if SPLIT == "mast_public_test" else f"mast_inputs_{SPLIT}.npz")
PICS = fv.load_pictures(CACHE / "features_v2/pictures_mast.npz")


def one(name):
    _, x = fv.shot_features(MAST / name, PICS, return_inputs=True)
    return name, np.asarray(x["cur"], np.float32), np.asarray(x["ip"], np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    index = pd.read_parquet(CACHE / f"features_v5_{SPLIT}.parquet")
    n = int((index.start + index.n).max())
    cur = np.full((n, len(PICS["columns"])), np.nan, np.float32)
    ip = np.full(n, np.nan, np.float32)
    rows = index.set_index("filename")
    with Pool(args.workers) as pool:
        for k, (name, c, i) in enumerate(pool.imap_unordered(one, list(index.filename)), start=1):
            r = rows.loc[name]
            if len(c) != int(r.n):
                raise SystemExit(f"{name}: {len(c)} input rows, {int(r.n)} feature rows")
            cur[int(r.start):int(r.start) + len(c)] = c
            ip[int(r.start):int(r.start) + len(c)] = i
            if k % 100 == 0:
                print(f"  {k}/{len(index)}", flush=True)
    if not (np.isfinite(cur).all() and np.isfinite(ip).all()):
        raise SystemExit("missing rows")
    np.savez(OUT, cur=cur, ip=ip, columns=np.array(PICS["columns"]))
    print(f"wrote {OUT}: {n} frames", flush=True)
