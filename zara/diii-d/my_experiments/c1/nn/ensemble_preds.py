"""Average the predictions of several runs (psi, q95, betaN per shot) into a new preds directory.

    python ensemble_preds.py --out runs/ens/preds runs/a/preds runs/b/preds ...
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--scal-from", nargs="*", type=Path, default=None,
                help="take q95/betaN from the average of these runs instead of the psi runs")
ap.add_argument("runs", nargs="+", type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
stems = sorted(p.stem for p in a.runs[0].glob("*.npz"))
for s in stems:
    ds = [np.load(r / f"{s}.npz") for r in a.runs]
    out = {k: np.mean([d[k].astype(np.float64) for d in ds], 0).astype(np.float32) for k in ("psi", "q95", "betaN")}
    if a.scal_from:
        sd = [np.load(r / f"{s}.npz") for r in a.scal_from]
        for k in ("q95", "betaN"):
            out[k] = np.mean([d[k].astype(np.float64) for d in sd], 0).astype(np.float32)
    if "t" in ds[0].files:
        out["t"] = ds[0]["t"]
    np.savez(a.out / f"{s}.npz", **out)
print(f"averaged {len(a.runs)} runs over {len(stems)} shots -> {a.out}")
