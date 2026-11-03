"""Collect scores of upegs / pegs runs into one table.

    python collect.py runs/u1_phys ../pegs/runs/v3 ...     (default: every runs/* here)
Columns: DIII-D holdout (704 or ho100) and the synthetic test sets (SPARCS / ITERS / MAST / D3DS) x harmonics variant.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load(f):
    d = json.load(open(f))
    if "results" in d:                      # score_holdout format
        r = list(d["results"].values())[0]
    else:
        r = d
    return r


def fmt(r):
    if r is None:
        return "-"
    return f"{r['S']:.4f} (psi {r['r2_psi']:.4f} qb {max(r['r2_qb'], -9):.3f} 1-D {1 - min(1, r['dlcfs']):.3f} C {r['consistency']:.3f})"


def main():
    runs = [Path(a) for a in sys.argv[1:]] or sorted(Path("runs").glob("*"))
    for run in runs:
        if not run.is_dir():
            continue
        rows = []
        for sub in sorted(run.glob("*/")):
            fs = sorted(glob.glob(str(sub / "score_*.json")))
            if fs:
                rows.append((sub.name, load(fs[-1])))
        if rows:
            print(f"== {run}")
            for name, r in rows:
                print(f"   {name:22s} {fmt(r)}")
                if "r2_cons_each" in r:
                    print("   " + " " * 22 + "  ".join(f"{k} {('n/a' if v is None else f'{v:.2f}')}" for k, v in r["r2_cons_each"].items()))


if __name__ == "__main__":
    main()
