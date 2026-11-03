"""Why do v5 cases with a high P4 factor fail? (synthetic diagnostics only)

Each v5 candidate's P4 factor f4 is recomputed from its coil seed (the second draw of its coil stream). Per f4 bin: share of
candidates accepted, share whose start frame solved, failure reasons of the frames, and, over the solved frames, the hidden
field, the share at its 9 mT cap, the pressure raise, the boundary-vs-Thomson-edge error and elongation.

    python ana_v5_p4.py <v5 set dir> [--procs 16]
"""
import argparse
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen4 as G  # noqa: E402

BINS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def f4_of(c):
    rc = G.shot_rng(c["coil_seed"], c["file"] + "|coil")
    rc.uniform(0.97, 1.03)
    return float(rc.uniform(0.6, 1.4))


def frames_of(path):
    try:
        fr = json.loads(pq.read_table(path, columns=["synth_frames"]).column(0)[0].as_py())
    except Exception:  # noqa: BLE001
        return []
    return [(x.get("dbz_mT"), x.get("bz_capped"), x.get("lms", 0.0), x.get("edge_err"), x["labels"]["elongation"] if "labels" in x else None)
            for x in fr if x["status"] == "ok"]


def one(args):
    d, c = args
    rec = None
    p = os.path.join(d, "done", f"{c['split']}_s{c['stratum']:02d}_{c['id']:05d}.json")
    try:
        rec = json.load(open(p))
    except Exception:  # noqa: BLE001
        return None
    fr = []
    raw = os.path.join(d, "raw", f"cand_{c['id']:05d}.parquet")
    if rec.get("ok", 0) > 0 and os.path.exists(raw):
        fr = frames_of(raw)
    return f4_of(c), rec, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    C = json.load(open(os.path.join(a.d, "candidates.json")))["candidates"]
    with Pool(a.procs) as p:
        res = [r for r in p.map(one, [(a.d, c) for c in C], chunksize=200) if r is not None]
    print(f"{len(res)} finished candidates")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r = [x for x in res if lo <= x[0] < hi]
        if not r:
            continue
        acc = np.mean([x[1].get("ok", 0) >= 20 and x[1].get("span", 0) >= 100 for x in r])
        started = np.mean([x[1].get("ok", 0) > 0 for x in r])
        st = collections.Counter()
        for x in r:
            for k, v in x[1].get("status", {}).items():
                if k not in ("outside", "no_start", "abandoned", "low_ip"):
                    st[k] += v
        tot = max(sum(st.values()), 1)
        fr = [f for x in r for f in x[2]]
        dbz = np.array([f[0] for f in fr if f[0] is not None], float)
        cap = np.mean([bool(f[1]) for f in fr]) if fr else np.nan
        lms = np.mean([(f[2] or 0) > 1e-3 for f in fr]) if fr else np.nan
        err = np.array([f[3] for f in fr if f[3] is not None], float)
        kap = np.array([f[4] for f in fr if f[4] is not None], float)
        print(f"f4 {lo:.1f}-{hi:.1f}: {len(r)} cands, accepted {acc:.3f}, started {started:.3f} | frame outcomes "
              + ", ".join(f"{k} {v / tot:.2f}" for k, v in st.most_common(5))
              + f" | solved frames {len(fr)}: field median {np.median(dbz) if len(dbz) else np.nan:.1f} mT, at cap {cap:.2f}, "
              f"pressure raised {lms:.2f}, edge err median {100 * np.median(err) if len(err) else np.nan:+.1f} cm, kappa {np.median(kap) if len(kap) else np.nan:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
