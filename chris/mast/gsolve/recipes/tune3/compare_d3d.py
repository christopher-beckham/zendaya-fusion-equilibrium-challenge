#!/usr/bin/env python3
"""D_label table on the frames that every listed recipe kept:  python compare_d3d.py def x8 cond ... -> d3dchk/compare.csv
Per label pooled R2 / bias / spread vs EFIT, offset-corrected R2psi, D_LCFS, S_off, and D_label = 1 - mean R2 over the
eight labels (q95, betaN, li, R_axis, Z_axis, kappa, tri_top, tri_bot)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAB = {"q95": ("q95", "e_q95"), "betaN": ("betaN", "e_betaN"), "li": ("p_li", "t_li"), "R_axis": ("p_R_axis", "t_R_axis"),
       "Z_axis": ("p_Z_axis", "t_Z_axis"), "kappa": ("p_kappa", "t_kappa"), "tri_top": ("p_tri_top", "t_tri_top"),
       "tri_bot": ("p_tri_bot", "t_tri_bot")}
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def r2f(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    ok = np.isfinite(t)
    p = np.where(np.isfinite(p), p, np.nanmean(t))
    return 1 - np.sum((t[ok] - p[ok]) ** 2) / np.sum((t[ok] - t[ok].mean()) ** 2)


tags = sys.argv[1:]
F = {t: pd.read_parquet(HERE / "d3dchk" / f"{t}_frames.parquet").set_index("row") for t in tags}
common = sorted(set.intersection(*[set(f.index) for f in F.values()]))
rows = []
for t, f in F.items():
    g = f.loc[common]
    r = {"recipe": t, "frames": len(g), "kept_frac_own": len(f) / 1932}
    for k, (p, q) in LAB.items():
        r[f"R2 {k}"] = r2f(g[q], g[p])
        r[f"bias {k}"] = float(np.nanmedian(g[p] - g[q]))
        r[f"sd {k}"] = float(np.nanstd(g[p] - g[q]))
    r["D_label"] = 1 - np.mean([r[f"R2 {k}"] for k in LAB])
    r["R2psi_off"] = 1 - g.ssr_off.sum() / g.sst.sum()
    r["dlcfs"] = float(g.groupby("e_shot").dlcfs.mean().mean())
    cons = np.mean([max(0, r2f(g[f"t_{k}"], g[f"p_{k}"])) for k in CONS])
    qb = np.mean([r2f(g.e_q95, g.q95), r2f(g.e_betaN, g.betaN)])
    r["S_off"] = 0.55 * max(0, r["R2psi_off"]) + 0.15 * max(0, qb) + 0.10 * (1 - min(1, r["dlcfs"])) + 0.20 * cons
    if "ssr_cal" in g:
        r["R2psi_cal"] = 1 - g.ssr_cal.sum() / g.sst.sum()
        r["S_D"] = 0.55 * max(0, r["R2psi_cal"]) + 0.15 * max(0, qb) + 0.10 * (1 - min(1, r["dlcfs"])) + 0.20 * cons
    r["median li gen/efit"] = f"{np.nanmedian(g.p_li):.3f}/{np.nanmedian(g.t_li):.3f}"
    r["median K gen/efit"] = f"{np.nanmedian(g.K):.2f}/{np.nanmedian(g.e_K):.2f}"
    rows.append(r)
# shot bootstrap (200 resamples of held-out shots, same resamples for every recipe): sd of D_label and of the
# paired difference to the reference recipe (REF env var, default 'cond')
import os
ref = os.environ.get("REF", "cond")
def dlab(g):
    return 1 - np.mean([r2f(g[q], g[p]) for p, q in LAB.values()])
def sd_(g):
    cons = np.mean([max(0, r2f(g[f"t_{k}"], g[f"p_{k}"])) for k in CONS])
    qb = np.mean([r2f(g.e_q95, g.q95), r2f(g.e_betaN, g.betaN)])
    ss = g.ssr_cal if "ssr_cal" in g else g.ssr_off
    return 0.55 * max(0, 1 - ss.sum() / g.sst.sum()) + 0.15 * max(0, qb) + 0.10 * (1 - min(1, g.groupby("e_shot").dlcfs.mean().mean())) + 0.20 * cons
shots = F[tags[0]].loc[common].e_shot.values
us = np.unique(shots)
idx_by = {u: np.nonzero(shots == u)[0] for u in us}
rng = np.random.default_rng(0)
B = {t: [] for t in tags}
BS = {t: [] for t in tags}
BZ = {t: [] for t in tags}
FC = {t: F[t].loc[common] for t in tags}
for b in range(200):
    pick = np.concatenate([idx_by[u] for u in rng.choice(us, len(us))])
    for t in tags:
        gg = FC[t].iloc[pick]
        B[t].append(dlab(gg)); BS[t].append(sd_(gg)); BZ[t].append(r2f(gg.t_Z_axis, gg.p_Z_axis))
for r in rows:
    t = r["recipe"]
    r["D_label boot sd"] = float(np.std(B[t]))
    if ref in B:
        d = np.array(B[t]) - np.array(B[ref])
        r[f"D_label - {ref} (boot mean +- sd)"] = f"{d.mean():+.4f} +- {d.std():.4f}"
        d2 = np.array(BS[t]) - np.array(BS[ref])
        r[f"S_D - {ref} (boot mean +- sd)"] = f"{d2.mean():+.4f} +- {d2.std():.4f}"
        r["dS_D_z"] = float(d2.mean() / max(d2.std(), 1e-9)); r["dD_label_z"] = float(d.mean() / max(d.std(), 1e-9))
        d3 = np.array(BZ[t]) - np.array(BZ[ref])
        r[f"Z_axis R2 - {ref}"] = f"{d3.mean():+.4f} +- {d3.std():.4f}"
D = pd.DataFrame(rows).set_index("recipe").T
D.to_csv(HERE / "d3dchk" / ("compare_" + "_".join(tags)[:150] + ".csv"))
print(D.to_string())
