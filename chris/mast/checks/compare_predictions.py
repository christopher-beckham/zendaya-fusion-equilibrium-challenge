"""Compare two MAST prediction files (same split, same shot keys): pooled R² of B against A, and the largest absolute
difference, for ψ, q95 and βN. No labels are used: A is the reference prediction.

    python chris/mast/checks/compare_predictions.py A.npz B.npz [--name NAME]
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--name", default="")
    x = ap.parse_args()
    za, zb = np.load(x.a), np.load(x.b)
    if za.files != zb.files:
        raise SystemExit("different keys")
    cells = []
    for q in ("psirz", "q95", "betaN"):
        ss_res = ss_tot = 0.0
        n = 0
        s = s2 = 0.0
        worst = 0.0
        keys = [k for k in za.files if k.endswith(f"_{q}")]
        for k in keys:
            a, b = za[k].astype(np.float64).ravel(), zb[k].astype(np.float64).ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            ss_res += float(((b - a) ** 2).sum()); s += float(a.sum()); s2 += float((a ** 2).sum()); n += a.size
            if a.size:
                worst = max(worst, float(np.abs(b - a).max()))
        ss_tot = s2 - s * s / n
        cells.append(f"{q} R² {1 - ss_res / ss_tot:.6f} max|Δ| {worst:.3g}")
    print(f"{x.name or x.b}: " + " | ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
