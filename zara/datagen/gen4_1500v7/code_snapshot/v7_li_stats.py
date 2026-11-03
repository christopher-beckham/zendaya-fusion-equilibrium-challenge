"""Recipe v7 checks: achieved li against the per-frame target, current-shape exponent, solver cost, hidden field and
acceptance. Synthetic data only (the target comes from the li* table, i.e. MAST inputs and a DIII-D-trained model).

    python v7_li_stats.py <set dir or assembled folder> [--v6 <v6 frames_metadata.parquet>]
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq


def pct(v, q=(5, 25, 50, 75, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.3g}" for x in np.percentile(v, q)) if len(v) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.path, "raw", "*.parquet")) or glob.glob(os.path.join(a.path, "mast_synth_*.parquet")))
    rows, fails, fr_all = [], Counter(), []
    for f in files:
        t = pq.read_table(f, columns=["synth_frames", "efit_li", "synth_meta"])
        fr = json.loads(t.column("synth_frames")[0].as_py())
        li = t.column("efit_li")[0].as_py()
        ok = [x for x in fr if x["status"] == "ok"]
        for x, l in zip(ok, li):
            rows.append(dict(li=l, **{k: (np.nan if x.get(k) is None else x.get(k)) for k in ("li_target", "li_star", "pk", "an", "am", "eb", "dbz_mT",
                                                                     "n_solves", "sec", "edge_err", "limited", "lk_star", "cp")},
                             start=x.get("phase") == "start"))
        for x in fr:
            fr_all.append(x)
            if x["status"] not in ("ok", "outside"):
                fails[x["status"]] += 1
    g = {k: np.array([r[k] for r in rows], float) for k in rows[0]} if rows else {}
    print(f"files {len(files)}  labelled frames {len(rows)}")
    if not rows:
        return
    m = np.isfinite(g["li_target"])
    d = g["li"] - g["li_target"]
    print(f"frames with a target {m.mean():.3f}")
    print(f"li achieved      {pct(g['li'][m])}")
    print(f"li target        {pct(g['li_target'][m])}")
    print(f"li* (no resid.)  {pct(g['li_star'][m])}")
    print(f"li - target      {pct(d[m])}   |d| < 0.03: {np.mean(np.abs(d[m]) < 0.03):.3f}  < 0.05: {np.mean(np.abs(d[m]) < 0.05):.3f}"
          f"  < 0.10: {np.mean(np.abs(d[m]) < 0.10):.3f}")
    print(f"corr(li, target) {np.corrcoef(g['li'][m], g['li_target'][m])[0, 1]:.3f}")
    print(f"pk               {pct(g['pk'][m])}   at bounds: lo {np.mean(g['pk'][m] <= 0.2001):.3f} hi {np.mean(g['pk'][m] >= 39.99):.3f}")
    print(f"an x pk          {pct((g['an'] * g['pk'])[m])}")
    print(f"hidden field mT  {pct(g['dbz_mT'])}   below 0: {np.mean(g['dbz_mT'] < -0.01):.3f}  at -10: {np.mean(g['dbz_mT'] <= -9.99):.3f}  at 20: {np.mean(g['dbz_mT'] >= 19.99):.3f}")
    print(f"edge err (m)     {pct(g['edge_err'])}")
    print(f"limited share    {np.nanmean(g['limited']):.3f}")
    print(f"solves / frame   {pct(g['n_solves'])}   sec / frame {pct(g['sec'])}")
    for nm, sel in (("start frames", g["start"] > 0), ("sweep frames", g["start"] == 0)):
        mm = sel & m
        if mm.any():
            print(f"  {nm}: n {int(mm.sum())}  |li - target| median {np.median(np.abs(d[mm])):.3f}  solves {pct(g['n_solves'][mm], (50, 90))}")
    big = np.abs(d) > 0.10
    if (big & m).any():
        print(f"|d| > 0.10: n {int((big & m).sum())}; their targets {pct(g['li_target'][big & m])}; pk {pct(g['pk'][big & m])}")
    print("frame failures:", dict(fails.most_common(12)))
    st = Counter(x["status"] for x in fr_all if x["status"] != "outside")
    print("frame status:", dict(st.most_common(12)))
    dn = glob.glob(os.path.join(a.path, "done", "*.json"))
    if dn:
        recs = [json.load(open(p)) for p in dn]
        acc = sum(1 for r in recs if r.get("ok", 0) >= 3)
        print(f"candidates finished {len(recs)}; cases with >= 3 labelled frames {acc} ({acc / max(len(recs), 1):.3f}); "
              f"wall s/candidate {pct([r.get('wall', np.nan) for r in recs])}")


if __name__ == "__main__":
    main()
