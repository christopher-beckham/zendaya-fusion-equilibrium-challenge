"""Label distributions (5 / 50 / 95 %): gsolve defaults (s1000) vs tuned (tuned1000) vs Zara v4 / v6 training sets.

    python compare_labels.py [--tuned DIR_WITH_NPZ] [--out labels_compare.csv]
Synthetic labels only (no MAST label is read).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Z = Path("/fsx/zara.ercan/more_data/m2")
COLS = {"q95": ("q95", "q95"), "betaN": ("betaN", "beta_n"), "li": ("li", "li"), "kappa": ("kappa", "elongation"),
        "R_axis": ("R_ax", "r_axis"), "Z_axis": ("Z_ax", "z_axis"), "vessel field (mT)": ("gen_bv", "dbz_mT"),
        "pressure ratio K": ("gen_K", "cp")}


def npz_labels(paths):
    out = {k: [] for k in COLS}
    for p in paths:
        z = np.load(p)
        for k, (a, _) in COLS.items():
            out[k].append(z[a] if a in z else np.full(len(z["shot"]), np.nan))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default=str(Path("/fsx/chris.beckham/gsolve_data/tuned1000")))
    ap.add_argument("--cond", default=None, help="folder with the conditional-prior set's *part*.npz")
    ap.add_argument("--out", default=str(HERE / "labels_compare.csv"))
    a = ap.parse_args()
    sets = {"gsolve defaults (s1000)": npz_labels(sorted(glob.glob("/fsx/chris.beckham/gsolve_data/s1000/s1000_part*.npz")))}
    tp = sorted(glob.glob(f"{a.tuned}/*part*.npz"))
    if tp:
        sets["gsolve x8 (tuned1000)"] = npz_labels(tp)
    if a.cond:
        sets[f"gsolve conditional ({Path(a.cond).name})"] = npz_labels(sorted(glob.glob(f"{a.cond}/*part*.npz")))
    for name, f in (("Zara v4", Z / "gen4_set5000v4/mast_g4s5000v4_train/frames_metadata.parquet"),
                    ("Zara v6", Z / "gen4_set1500v6/mast_g4s1500v6_train/frames_metadata.parquet")):
        d = pd.read_parquet(f)
        sets[name] = {k: d[b].values.astype(float) if b in d else np.full(len(d), np.nan) for k, (_, b) in COLS.items()}
    rows = []
    for k in COLS:
        r = {"label": k}
        for s, v in sets.items():
            x = v[k][np.isfinite(v[k])]
            r[s] = " / ".join(f"{q:.3g}" for q in np.percentile(x, [5, 50, 95])) if len(x) else "-"
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
