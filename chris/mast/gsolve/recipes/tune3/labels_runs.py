#!/usr/bin/env python3
"""Synthetic MAST label medians (5/50/95) per critic run (tuning panel, kept equilibria): python labels_runs.py TAG..."""
import glob
import sys

import numpy as np
import pandas as pd

rows = []
for t in sys.argv[1:]:
    Z = [np.load(f) for f in sorted(glob.glob(f"runs/{t}/gen_*.npz"))]
    r = {"run": t}
    for k in ("li", "gen_K", "betaN", "q95", "kappa", "R_ax", "Z_ax", "gen_bv"):
        v = np.concatenate([z[k] for z in Z]); v = v[np.isfinite(v)]
        r[k] = " / ".join(f"{q:.3g}" for q in np.percentile(v, [5, 50, 95]))
    rows.append(r)
print(pd.DataFrame(rows).set_index("run").T.to_string())
