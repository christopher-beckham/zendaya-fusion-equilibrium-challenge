"""Load every assembled case as a loader would (numpy arrays per column) and report shapes, NaN/Inf and label ranges."""
import glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq

bad = 0
n_fr = 0
stats = {k: [] for k in ("q95", "beta_n", "li", "r_axis", "z_axis")}
for d in sys.argv[1:]:
    files = sorted(f for f in glob.glob(os.path.join(d, "*.parquet")) if not f.endswith("frames_metadata.parquet"))
    for f in files:
        t = pq.read_table(f).to_pylist()[0]
        T = len(t["efit_times"])
        psi = np.asarray(t["efit_psirz"], float)
        lr, lz, ln = np.asarray(t["efit_lcfs_r"], float), np.asarray(t["efit_lcfs_z"], float), np.asarray(t["efit_lcfs_n"], float)
        te = np.asarray(t["thomson_core_Te"], float)
        tm, tk = np.asarray(t["magnetics_time"], float), np.asarray(t["efit_times"], float)
        if not np.all((tk >= tm.min() + 2.0) & (tk <= tm.max() - 2.0)):
            bad += 1; print("FRAME OUTSIDE INPUT WINDOW", f, tk.tolist(), tm.min(), tm.max()); continue
        ok = (psi.shape == (T, 65, 65) and np.isfinite(psi).all() and lr.shape == lz.shape and lr.shape[0] == T
              and np.all(ln <= lr.shape[1]) and te.ndim == 2 and te.shape[1] == 131 and len(t["magnetics_time"]) > 100
              and all(len(t[c]) == T for c in ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis")))
        for k in stats:
            v = np.asarray(t["efit_" + k], float); ok &= bool(np.isfinite(v).all()); stats[k] += v.tolist()
        n_fr += T
        if not ok:
            bad += 1; print("BAD", f, psi.shape, lr.shape, te.shape)
    print(f"{d}: {len(files)} files")
print(f"labelled frames {n_fr}; files failing the checks: {bad}")
for k, v in stats.items():
    print(f"  {k}: min {np.min(v):.3g} 5% {np.percentile(v, 5):.3g} median {np.median(v):.3g} 95% {np.percentile(v, 95):.3g} max {np.max(v):.3g}")
