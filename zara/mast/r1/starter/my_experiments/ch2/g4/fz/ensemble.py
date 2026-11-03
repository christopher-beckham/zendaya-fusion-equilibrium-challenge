"""Average the predictions of several runs (same shots, same frames): psi mean, q95 / betaN mean.

    python ensemble.py --out runs/ens_a/ho --preds runs/u1_phys/ho/preds runs/u1_aux/ho/preds ...
Writes <out>/preds/<stem>.npz and <out>/stems.txt. The same averaging rule applies on every machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    stems = sorted(p.stem for p in args.preds[0].glob("*.npz"))
    od = args.out / "preds"; od.mkdir(parents=True, exist_ok=True)
    for s in stems:
        ds = [np.load(p / f"{s}.npz") for p in args.preds]
        psi = np.mean([d["psi"].astype(np.float64) for d in ds], 0).astype(np.float32)
        q95 = np.mean([d["q95"].astype(np.float64) for d in ds], 0).astype(np.float32)
        bn = np.mean([d["betaN"].astype(np.float64) for d in ds], 0).astype(np.float32)
        assert np.isfinite(psi).all() and np.isfinite(q95).all() and np.isfinite(bn).all(), s
        extra = {k: ds[0][k] for k in ds[0].files if k not in ("psi", "q95", "betaN")}
        np.savez(od / f"{s}.npz", psi=psi, q95=q95, betaN=bn, **extra)
    (args.out / "stems.txt").write_text("\n".join(stems))
    print(f"{len(stems)} shots averaged over {len(args.preds)} runs -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
