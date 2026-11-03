"""Recipe v6 checks against real MAST inputs (no MAST label is read): coil-pair imbalance, P4 / P5 balance, TF spread,
pair factors, hidden field, labels, and Thomson channels reading below 20 eV inside the plasma.

    python v6_stats.py <folder> [--procs 8] [--max N]

Real reference: the challenge input pool (coil currents at every EFIT time of every mast_public_test shot, |Ip| >= 150 kA)
and, for Thomson, the origin shot's real slices at the same times as the synthetic ones.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen4  # noqa: E402

POOL = "/fsx/zara.ercan/more_data/generated/cache/mast_input_pool.npz"
PAIRS = ("p2", "p4", "p5", "p3", "p6")


def pct(v, q=(5, 50, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return "n/a" if not len(v) else " / ".join(f"{x:.3g}" for x in np.percentile(v, q))


def low_share(R, TE):
    """Share of channels inside the 30 eV edges (smoothed profile) that read below 20 eV, pooled over slices."""
    n_in = n_low = 0
    for row in TE:
        te = np.asarray(row, float)[1:]
        lo, hi, _ = gen4.te_edges(R, te, 30.0)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        m = (R > lo) & (R < hi) & np.isfinite(te)
        n_in += m.sum()
        n_low += (te[m] < 20.0).sum()
    return n_in, n_low


def one(f):
    t = pq.read_table(f)
    col = lambda c: t.column(c)[0].as_py()
    meta = json.loads(col("synth_meta"))
    tk = np.asarray(col("efit_times"), float)
    tm = np.asarray(col("magnetics_time"), float)
    at = lambda c: np.interp(tk, tm, np.asarray(col(f"magnetics_{c}_current"), float))
    out = {f"d_{p}": (at(p + "u") - at(p + "l")).tolist() for p in PAIRS}
    p4, p5, ip = 0.5 * (at("p4u") + at("p4l")), 0.5 * (at("p5u") + at("p5l")), np.abs(at("plasma"))
    out.update(r45=(p4 / p5).tolist(), p4ip=(p4 / ip).tolist(), p5ip=(p5 / ip).tolist(), tf=at("tf").tolist())
    jit = meta.get("jitter", {})
    out.update(f2=jit.get("f2"), f4=jit.get("f4"), f5=jit.get("f5"), bz0=meta.get("bz0_mT"))
    fr = [x for x in json.loads(col("synth_frames")) if x["status"] == "ok"]
    out["dbz"] = [x.get("dbz_mT") for x in fr]
    for k in ("q95", "beta_n", "li", "elongation", "z_axis"):
        out[k] = col("efit_" + k)
    R = np.asarray(col("thomson_core_R"), float)
    out["syn_low"] = low_share(R, col("thomson_core_Te"))
    src = pq.read_table(os.path.join(gen4.DATA, meta["case"]["file"]), columns=["thomson_core_times", "thomson_core_Te"])
    ts_all = np.asarray(src.column("thomson_core_times")[0].as_py(), float)
    te_all = src.column("thomson_core_Te")[0].as_py()
    ts_syn = np.asarray(col("thomson_core_times"), float)
    out["real_low"] = low_share(R, [te_all[int(np.argmin(np.abs(ts_all - x)))] for x in ts_syn])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--max", type=int, default=10 ** 9)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.folder, "*.parquet")) if not f.endswith("frames_metadata.parquet"))[: a.max]
    with Pool(a.procs) as p:
        res = p.map(one, files, chunksize=8)
    A = lambda k: [v for r in res for v in (r[k] if isinstance(r[k], list) else [r[k]]) if v is not None]
    p = np.load(POOL)
    cols = list(p["columns"])
    V = p["values"]
    V = V[np.abs(V[:, cols.index("magnetics_plasma_current")]) >= 150]
    rc = lambda c: V[:, cols.index(f"magnetics_{c}_current")]
    print(f"{a.folder}: {len(files)} cases, {len(A('q95'))} labelled frames")
    print("upper - lower current, kA (5 / 50 / 95 %): synthetic | real MAST")
    for pr in PAIRS:
        print(f"  {pr}: {pct(A('d_' + pr))} | {pct(rc(pr + 'u') - rc(pr + 'l'))}")
    r4, r5 = 0.5 * (rc("p4u") + rc("p4l")), 0.5 * (rc("p5u") + rc("p5l"))
    ipr = np.abs(rc("plasma"))
    print(f"P4 / P5 current ratio: {pct(A('r45'))} | {pct(r4 / r5)}")
    print(f"P4 / Ip: {pct(A('p4ip'))} | {pct(r4 / ipr)};  P5 / Ip: {pct(A('p5ip'))} | {pct(r5 / ipr)}")
    tf_s, tf_r = np.asarray(A("tf")), rc("tf")
    print(f"TF current, kA: {pct(tf_s)} | {pct(tf_r)};  share in [-85.0, -84.7] kA: {np.mean((tf_s >= -85.0) & (tf_s <= -84.7)):.2f} | "
          f"{np.mean((tf_r >= -85.0) & (tf_r <= -84.7)):.2f}")
    print(f"pair factors: P2 {pct(A('f2'))}, P4 {pct(A('f4'))} (mean {np.mean(A('f4')):.3f}), P5 {pct(A('f5'))}")
    print(f"hidden field per frame, mT: {pct(A('dbz'))}; at 0 mT {np.mean(np.asarray(A('dbz')) < 0.05):.3f}, at 20 mT "
          f"{np.mean(np.asarray(A('dbz')) > 19.95):.3f}; per-case start {pct(A('bz0'))}")
    for k in ("q95", "beta_n", "li", "elongation", "z_axis"):
        print(f"  {k}: {pct(A(k))}")
    s = np.sum([r["syn_low"] for r in res], axis=0)
    rl = np.sum([r["real_low"] for r in res], axis=0)
    print(f"Thomson channels inside the 30 eV edges reading < 20 eV: {s[1] / max(s[0], 1):.4f} | {rl[1] / max(rl[0], 1):.4f}")


if __name__ == "__main__":
    main()
