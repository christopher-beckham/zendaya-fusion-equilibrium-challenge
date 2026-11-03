"""Inputs-only realism checks for gen4 parquet files (no MAST label is read: only our synthetic labels and real INPUTS).

    python check.py <dir or files...> [--max N] [--procs 4]

Pooled over files: frame status counts, time per frame, label percentiles, synthetic boundary vs the REAL Thomson 30 eV
edges of the origin frame, axis vs the real Te peak radius, synthetic vs real Thomson at the same times (Te / ne peaks,
missing channels, slice-to-slice jitter of the pressure peak radius and level, as in Chris's feedback), case sizes.
"""
from __future__ import annotations

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

LAB = ["li", "beta_n", "q95", "elongation", "r_axis", "z_axis", "beta_p"]
KEYS = ["secs", "d_in", "d_out", "d_pk", "cp", "volume", "syn_te", "real_te", "syn_ne", "real_ne", "syn_nan", "real_nan",
        "syn_pk", "real_pk", "syn_lv", "real_lv", "n_ok", "centre_ip", "centre_didt", "li_te_pk"] + LAB


def pct(v, q=(5, 50, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return "n/a" if not len(v) else " / ".join(f"{x:.3g}" for x in np.percentile(v, q))


def peak_level(TE, NE, R):
    pk, lv = [], []
    for a, b in zip(TE, NE):
        a, b = np.asarray(a, float)[1:], np.asarray(b, float)[1:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            pk.append(np.nan); lv.append(np.nan); continue
        pe = a[m] * b[m]; r = R[m]; o = np.argsort(r); pe, r = pe[o], r[o]
        sm = np.array([np.median(pe[max(0, i - 3):i + 4]) for i in range(len(pe))])
        pk.append(r[int(np.argmax(sm))]); lv.append(np.mean(sm))
    return np.array(pk), np.array(lv)


def one(f):
    out = {k: [] for k in KEYS}
    out["status"], out["starts"] = {}, {}
    t = pq.read_table(f)
    meta = json.loads(t.column("synth_meta")[0].as_py())
    fr = json.loads(t.column("synth_frames")[0].as_py())
    for x in fr:
        out["status"][x["status"]] = out["status"].get(x["status"], 0) + 1
        if "sec" in x:
            out["secs"].append(x["sec"])
        if x["status"] == "ok":
            out["starts"][x["start"]] = out["starts"].get(x["start"], 0) + 1
            out["d_in"].append(x["R_lcfs_in"] - x["R_ts_in"]); out["d_out"].append(x["R_lcfs_out"] - x["R_ts_out"])
            out["cp"].append(x["paxis"] / x["pe_axis"]); out["volume"].append(x.get("volume", np.nan))
    for k in LAB:
        out[k] += t.column("efit_" + k)[0].as_py()
    okf = [x for x in fr if x["status"] == "ok"]
    out["n_ok"].append(len(okf))
    out["d_pk"] += list(np.asarray(t.column("efit_r_axis")[0].as_py()) - np.array([x["R_ts_pk"] for x in okf]))
    if "case" in meta:
        out["centre_ip"].append(meta["case"]["ip"]); out["centre_didt"].append(meta["case"].get("didt", np.nan))
    origin = meta.get("origin_file") or meta["case"]["file"]
    src = pq.read_table(os.path.join(gen4.DATA, origin), columns=["thomson_core_times", "thomson_core_Te", "thomson_core_ne"])
    ts_all = np.asarray(src.column("thomson_core_times")[0].as_py(), float)
    te_r = src.column("thomson_core_Te")[0].as_py(); ne_r = src.column("thomson_core_ne")[0].as_py()
    te_s = t.column("thomson_core_Te")[0].as_py(); ne_s = t.column("thomson_core_ne")[0].as_py()
    R = np.asarray(t.column("thomson_core_R")[0].as_py(), float)
    ts_syn = np.asarray(t.column("thomson_core_times")[0].as_py(), float)
    idx = [int(np.argmin(np.abs(ts_all - x))) for x in ts_syn]
    te_r = [te_r[i] for i in idx]; ne_r = [ne_r[i] for i in idx]
    # li against the peaking of the synthetic Te at the nearest slice (Chris: Te shape should not give away the current)
    li = t.column("efit_li")[0].as_py()
    for tk, lk in zip(t.column("efit_times")[0].as_py(), li):
        if not len(ts_syn):
            break
        a = np.asarray(te_s[int(np.argmin(np.abs(ts_syn - tk)))], float)[1:]
        a = a[np.isfinite(a) & (a > 30)]
        if len(a) >= 20:
            out["li_te_pk"].append((lk, np.percentile(a, 97) / np.median(a)))
    for which, TE, NE in (("syn", te_s, ne_s), ("real", te_r, ne_r)):
        pk, lv = peak_level(TE, NE, R)
        if len(pk) >= 3:
            out[which + "_pk"] += list(np.abs(pk[1:-1] - 0.5 * (pk[:-2] + pk[2:])))
            out[which + "_lv"] += list(np.abs(lv[1:-1] - 0.5 * (lv[:-2] + lv[2:])) / lv[1:-1])
    for a1, b1, a2, b2 in zip(te_s, ne_s, te_r, ne_r):
        a1, b1, a2, b2 = (np.asarray(v, float)[1:] for v in (a1, b1, a2, b2))
        if np.isfinite(a1).sum() < 20 or np.isfinite(a2).sum() < 20:
            continue
        out["syn_te"].append(np.nanpercentile(a1, 97)); out["real_te"].append(np.nanpercentile(a2, 97))
        out["syn_ne"].append(np.nanpercentile(b1, 97)); out["real_ne"].append(np.nanpercentile(b2, 97))
        out["syn_nan"].append(np.isnan(a1).mean()); out["real_nan"].append(np.isnan(a2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(f for f in glob.glob(os.path.join(p, "*.parquet")) if not f.endswith("frames_metadata.parquet")) if os.path.isdir(p) else [p]
    files = files[: a.max]
    with Pool(a.procs) as pool:
        res = pool.map(one, files, chunksize=4)
    A = {k: [] for k in KEYS}
    st, starts = {}, {}
    for r in res:
        for k in KEYS:
            A[k] += r[k]
        for s, v in r["status"].items():
            st[s] = st.get(s, 0) + v
        for s, v in r["starts"].items():
            starts[s] = starts.get(s, 0) + v
    n = sum(st.values())
    print(f"{len(files)} files, {n} frames; status {json.dumps(dict(sorted(st.items(), key=lambda kv: -kv[1])))}")
    print(f"accepted frames {st.get('ok', 0)} ({st.get('ok', 0) / max(n, 1):.2f}); labelled frames per case "
          f"{np.mean(A['n_ok']):.1f} (histogram {np.bincount(A['n_ok']).tolist()}); starts {starts}; s/frame p50/p90 {pct(A['secs'], (50, 90))}")
    for k in LAB + ["volume"]:
        print(f"  {k:10s} 5/50/95 %: {pct(A[k])}")
    print(f"  cp (total/electron axis pressure): {pct(A['cp'])}")
    print(f"  LCFS inner - real Thomson 30 eV inner edge (cm): {pct(100 * np.asarray(A['d_in']))}")
    print(f"  LCFS outer - real Thomson 30 eV outer edge (cm): {pct(100 * np.asarray(A['d_out']))}")
    print(f"  R_axis - real Te peak radius (cm):              {pct(100 * np.asarray(A['d_pk']))}")
    if A["centre_ip"]:
        print(f"  case centre Ip (kA) 5/25/50/75/95: {pct(A['centre_ip'], (5, 25, 50, 75, 95))}; "
              f"dIp/dt per 100 ms (fraction of Ip): {pct(A['centre_didt'], (5, 25, 50, 75, 95))}")
    print("  Thomson, synthetic | real at the same times:")
    print(f"    Te peak (eV): {pct(A['syn_te'])}  |  {pct(A['real_te'])}")
    print(f"    ne peak (1e19 m-3): {pct(np.asarray(A['syn_ne']) / 1e19)}  |  {pct(np.asarray(A['real_ne']) / 1e19)}")
    print(f"    missing channel fraction (mean): {np.mean(A['syn_nan']):.3f}  |  {np.mean(A['real_nan']):.3f} ({len(A['syn_nan'])} slices)")
    print(f"    slice-to-slice jitter of the p_e peak radius (cm), median / 90th: "
          f"{pct(100 * np.asarray(A['syn_pk']), (50, 90))}  |  {pct(100 * np.asarray(A['real_pk']), (50, 90))}")
    print(f"    slice-to-slice relative jitter of the mean p_e, median / 90th: {pct(A['syn_lv'], (50, 90))}  |  {pct(A['real_lv'], (50, 90))}")
    if len(A["li_te_pk"]) > 10:
        from scipy.stats import spearmanr
        x = np.asarray(A["li_te_pk"], float)
        print(f"  li vs synthetic Te peaking (97th pct / median of channels > 30 eV): Spearman {spearmanr(x[:, 0], x[:, 1])[0]:.2f} "
              f"over {len(x)} frames")


if __name__ == "__main__":
    main()
