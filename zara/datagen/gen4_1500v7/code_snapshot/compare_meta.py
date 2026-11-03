"""Label and hidden-parameter percentiles of two assembled sets side by side (synthetic data only).

    python compare_meta.py <frames_metadata.parquet A> <frames_metadata.parquet B> [name A] [name B]
"""
import sys

import numpy as np
import pyarrow.parquet as pq

COLS = ["li", "li_target", "li_star", "pk", "q95", "beta_n", "beta_p", "elongation", "delta_upper", "delta_lower",
        "r_axis", "z_axis", "dbz_mT", "cp", "an"]


def pct(t, c):
    if c not in t.column_names:
        return "-"
    v = np.array([np.nan if x is None else x for x in t.column(c).to_pylist()], float)
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.3g}" for x in np.percentile(v, [5, 50, 95])) + f" (sd {v.std():.3g})" if len(v) else "-"


def main():
    a, b = pq.read_table(sys.argv[1]), pq.read_table(sys.argv[2])
    na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
    print(f"| 5 / 50 / 95 % | {na} ({a.num_rows} frames) | {nb} ({b.num_rows} frames) |")
    print("|---|---|---|")
    for c in COLS:
        print(f"| {c} | {pct(a, c)} | {pct(b, c)} |")
    for t, n in ((a, na), (b, nb)):
        lim = np.array([bool(x) for x in t.column("limited").to_pylist()])
        print(f"{n}: limited share {lim.mean():.3f}")


if __name__ == "__main__":
    main()
