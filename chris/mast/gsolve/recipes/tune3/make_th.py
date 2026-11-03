#!/usr/bin/env python3
"""Realistic-edge DIII-D tables: d3d/<split>_th.parquet = <split>3 with the oracle EFIT edges replaced by the Thomson
edges (R_in/R_out <- R_in_th/R_out_th, z_* <- zth_*, te_q/ne_q <- te_q_th/ne_q_th). Rows without Thomson edges keep the
oracle values (val: 0.1-0.3% of frames) and are flagged th_ok = False; the teacher fit drops them."""
from pathlib import Path
import numpy as np, pandas as pd
import zfeat
HERE = Path(__file__).resolve().parent
for sp in ("train", "val"):
    d = pd.read_parquet(HERE / "d3d" / f"{sp}3.parquet")
    ok = np.isfinite(d.R_out_th) & np.isfinite(d.zth_qstar) & d.te_q_th.notna()
    d["th_ok"] = ok
    d["R_in_or"], d["R_out_or"] = d.R_in, d.R_out
    d.loc[ok, "R_in"], d.loc[ok, "R_out"] = d.R_in_th[ok], d.R_out_th[ok]
    for k in zfeat.FEATS:
        d.loc[ok, f"z_{k}"] = d.loc[ok, f"zth_{k}"]
    d["te_q"] = [t if o else q for t, q, o in zip(d.te_q_th, d.te_q, ok)]
    d["ne_q"] = [t if o else q for t, q, o in zip(d.ne_q_th, d.ne_q, ok)]
    d.to_parquet(HERE / "d3d" / f"{sp}_th.parquet"); print(sp, len(d), "th_ok", ok.mean().round(4))
