"""Write a MAST submission file from a per-shot predictions folder, shots in sorted local parquet order (= Hugging Face
stream order), psi / q95 / betaN float32. Same content rule as my_experiments/m2/post/build_sub.py, any split.

    python write_mast.py --parquet <split parquet dir> --preds <preds dir> --out <mast_<split>.npz>
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", type=Path, required=True)
ap.add_argument("--preds", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
arrays = {}
files = sorted(a.parquet.glob("*.parquet"))
for i, f in enumerate(files):
    T = len(pq.read_table(f, columns=["efit_times"]).column(0).to_pylist()[0])
    p = np.load(a.preds / f"{f.stem}.npz")
    psi = np.asarray(p["psi"], np.float32); q95 = np.asarray(p["q95"], np.float32); bn = np.asarray(p["betaN"], np.float32)
    assert psi.shape == (T, 65, 65), (f.stem, psi.shape, T)
    assert np.isfinite(psi).all() and np.isfinite(q95).all() and np.isfinite(bn).all(), f"{f.stem}: non-finite values"
    arrays[f"shot_{i:04d}_psirz"] = psi; arrays[f"shot_{i:04d}_q95"] = q95; arrays[f"shot_{i:04d}_betaN"] = bn
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **arrays)
print(f"{len(files)} shots -> {a.out}")
