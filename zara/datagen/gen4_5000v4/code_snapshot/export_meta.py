"""One row per labelled frame of an assembled set folder: origin shot and centre time, coil factors and blend partner,
profile parameters, pressure ratio, hidden field offset, limited flag, Thomson-edge diagnostics and the main labels
(including upper / lower triangularity from the stored boundary). Writes <folder>/frames_metadata.parquet and prints a
summary. Synthetic data only.

    python export_meta.py <assembled folder> [--procs 8]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LAB = ["q95", "beta_n", "li", "elongation", "r_axis", "z_axis", "beta_p"]
FR = ["tite", "ff", "nf", "w", "am", "an", "tite_d", "eb", "ex1", "paxis", "pe_axis", "cp", "ibz", "dbz_mT", "limited",
      "inner_hot", "edge_err", "target", "R_lcfs_in", "R_lcfs_out", "R_ts_in", "R_ts_out", "start", "phase"]


def triangularity(r, z, n):
    r, z = np.asarray(r[:int(n)], float), np.asarray(z[:int(n)], float)
    rmin, rmax = r.min(), r.max()
    a, rgeo = 0.5 * (rmax - rmin), 0.5 * (rmax + rmin)
    return (rgeo - r[np.argmax(z)]) / a, (rgeo - r[np.argmin(z)]) / a


def one(f):
    t = pq.read_table(f)
    col = lambda c: t.column(c)[0].as_py()
    meta = json.loads(col("synth_meta"))
    case = meta["case"]
    frames = [x for x in json.loads(col("synth_frames")) if x["status"] == "ok"]
    jit = meta.get("jitter", {})
    cols = jit.get("cols", {})
    blend = case.get("blend") or [None, None, None]
    rows = []
    for k, tk in enumerate(col("efit_times")):
        x = next((y for y in frames if abs(y["t"] - tk) < 1e-3), {})
        du, dl = triangularity(col("efit_lcfs_r")[k], col("efit_lcfs_z")[k], col("efit_lcfs_n")[k])
        row = dict(file=os.path.basename(f), frame=k, t=float(tk), origin_file=case["file"], centre_t=case["t"], split=case["split"],
                   stratum=case["stratum"], seed=case.get("seed"), group=case.get("group", -1), blend_file=blend[0],
                   blend_t=blend[1], blend_alpha=blend[2], recipe=meta.get("recipe"), version=meta.get("version"),
                   sol_jitter=jit.get("sol"), ip_jitter=jit.get("ip"), tf_jitter=jit.get("tf"), delta_upper=du, delta_lower=dl)
        for c, v in cols.items():
            row["jit_" + c.replace("magnetics_", "").replace("_current", "")] = v
        for q in FR:
            v = x.get(q)
            row[q] = None if v is None else (bool(v) if q in ("limited", "inner_hot") else (v if isinstance(v, str) else float(v)))
        for q in LAB:
            row[q] = float(col("efit_" + q)[k])
        rows.append(row)
    return rows


def pct(v, q=(5, 50, 95)):
    v = np.asarray([x for x in v if x is not None], float)
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.3g}" for x in np.percentile(v, q)) if len(v) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, "*.parquet")))
    with Pool(a.procs) as p:
        rows = [r for rr in p.map(one, files, chunksize=20) for r in rr]
    keys = sorted({k for r in rows for k in r})
    table = pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])
    out = os.path.join(a.folder, "frames_metadata.parquet")
    pq.write_table(table, out, compression="zstd")
    g = lambda k: [r.get(k) for r in rows]
    lim = np.array([bool(r.get("limited")) for r in rows])
    print(f"{out}: {len(rows)} frames, {len(keys)} columns, {len(files)} cases")
    print(f"  limited frames {lim.sum()} ({100 * lim.mean():.1f} %); cases blended {sum(1 for f in set((r['file'], r['blend_file']) for r in rows) if f[1])}; "
          f"frames in profile groups {sum(1 for r in rows if r['group'] >= 0)}")
    for k in ("cp", "dbz_mT", "delta_upper", "delta_lower", "li", "elongation", "beta_n", "q95"):
        v = np.asarray([x for x in g(k) if x is not None], float)
        print(f"  {k:12s} 5/50/95 %: {pct(v)}   sd {np.nanstd(v):.3g}")
    d = np.asarray(g("delta_upper"), float) - np.asarray(g("delta_lower"), float)
    print(f"  upper - lower triangularity sd {np.nanstd(d):.3g}; share |upper - lower| > 0.1: {np.mean(np.abs(d) > 0.1):.3f}")


if __name__ == "__main__":
    main()
