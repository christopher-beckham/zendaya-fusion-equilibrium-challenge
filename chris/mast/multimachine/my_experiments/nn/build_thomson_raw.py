"""Cache raw Thomson Te per channel at the EFIT times, with each channel's (R, Z), for the Te-on-flux-surfaces loss.

Inputs only (Thomson and the EFIT time stamps); no efit_* label is read. Two files under the cache:
  thomson_raw_d3d.npz   rows aligned with psi.npy (shots.parquet order, first n EFIT frames of each shot)
  thomson_raw_mast.npz  rows aligned with features_v5_mast_public_test.npy
each with te (N, C) float16 in eV (NaN = no value), r and z (N, C) float16 in m. Channels are the core and edge
systems (DIII-D: vertical core chord and tangential midplane chords; MAST: midplane core and edge). Te is linearly
interpolated to each EFIT time from the two nearest Thomson times, and left NaN when the nearest one is more than
MAX_GAP_MS away.

With --quantity ne, the same for electron density (m^-3, stored as ne / 1e19 in float16) into
thomson_raw_ne_{d3d,mast}.npz.

usage: python build_thomson_raw.py [--workers 4] [--quantity Te|ne]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import _align_channels, _group_key, _rect_profile
from features_v2 import _cell_py

CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
DATA = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data"
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
SSFX = "" if SPLIT == "mast_public_test" else f"_{SPLIT}"   # file-name suffix for caches without the split in their name
MAX_GAP_MS = 25.0
SYSTEMS = {"core": ("TS_core",), "edge": ("TS_tangential", "TS_edge")}


def shot(job):
    path, n = Path(job[0]), job[1]
    q = job[2] if len(job) > 2 else "Te"
    cols = ["efit_times", "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z"]
    for s in SYSTEMS:
        cols += [f"thomson_{s}_times", f"thomson_{s}_{q}"]
    t = pq.read_table(path, columns=cols)
    et = np.asarray(_cell_py(t, "efit_times"), np.float64)
    et = et if n is None else et[:n]
    names = [str(x) for x in _cell_py(t, "thomson_chord_name")]
    cr = np.asarray(_cell_py(t, "thomson_chord_R"), np.float64)
    cz = np.asarray(_cell_py(t, "thomson_chord_Z"), np.float64)
    groups = np.array([_group_key(x) for x in names])
    te_all, r_all, z_all = [], [], []
    for s, keys in SYSTEMS.items():
        chosen = np.isin(groups, keys)
        rows = _cell_py(t, f"thomson_{s}_{q}")
        times = np.asarray(_cell_py(t, f"thomson_{s}_times"), np.float64)
        if not chosen.any() or not rows or times.size == 0:
            continue
        te = _rect_profile(rows)
        te, _ = _align_channels(te, te, int(chosen.sum()), path.name)
        order = np.argsort(times, kind="mergesort")
        times, te = times[order], te[order]
        out = np.full((len(et), te.shape[1]), np.nan)
        j = np.clip(np.searchsorted(times, et), 1, len(times) - 1) if len(times) > 1 else np.zeros(len(et), int)
        for k, e in enumerate(et):
            if len(times) == 1:
                if abs(times[0] - e) <= MAX_GAP_MS:
                    out[k] = te[0]
                continue
            lo, hi = j[k] - 1, j[k]
            if min(abs(times[lo] - e), abs(times[hi] - e)) > MAX_GAP_MS:
                continue
            w = np.clip((e - times[lo]) / max(times[hi] - times[lo], 1e-9), 0.0, 1.0)
            out[k] = (1 - w) * te[lo] + w * te[hi]
        out[~np.isfinite(out) | (out <= 0)] = np.nan
        if q == "ne":
            out = out / 1e19
        te_all.append(out); r_all.append(np.broadcast_to(cr[chosen], out.shape)); z_all.append(np.broadcast_to(cz[chosen], out.shape))
    if not te_all:
        return np.full((len(et), 0), np.nan), np.zeros((len(et), 0)), np.zeros((len(et), 0))
    return np.concatenate(te_all, 1), np.concatenate(r_all, 1), np.concatenate(z_all, 1)


def build(jobs, n_rows, out, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(shot, jobs, chunksize=8), start=1):
            results.append(res)
            if i % 500 == 0:
                print(f"  {out.name}: {i}/{len(jobs)}", flush=True)
    width = max(r[0].shape[1] for r in results)
    te = np.full((n_rows, width), np.nan, np.float16)
    r = np.zeros((n_rows, width), np.float16); z = np.zeros((n_rows, width), np.float16)
    cursor = 0
    for (a, b, c), (_, n, *_) in zip(results, jobs):
        k = len(a)
        if n is not None and k != n:
            raise SystemExit(f"row count {k} != {n}")
        te[cursor:cursor + k, :a.shape[1]] = a; r[cursor:cursor + k, :a.shape[1]] = b; z[cursor:cursor + k, :a.shape[1]] = c
        cursor += k
    if cursor != n_rows:
        raise SystemExit(f"{out.name}: {cursor} rows written, expected {n_rows}")
    np.savez(out, te=te, r=r, z=z)          # 'te' holds ne / 1e19 in the _ne files
    have = np.isfinite(te.astype(np.float32)).sum(1)
    print(f"wrote {out}: {n_rows} rows, {width} channels, channels with Te per frame median {np.median(have):.0f}, "
          f"frames with none {np.mean(have == 0):.1%}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quantity", choices=["Te", "ne"], default="Te")
    ap.add_argument("--mast-only", action="store_true", help="build only the MAST file for FEC_MAST_SPLIT")
    args = ap.parse_args()
    tag = "" if args.quantity == "Te" else "_ne"
    if not args.mast_only:
        shots = pd.read_parquet(CACHE / "shots.parquet").sort_values("start")
        jobs = [(str(DATA / "diii_d_train" / f), int(n), args.quantity) for f, n in zip(shots.filename, shots.n)]
        build(jobs, int(shots.n.sum()), CACHE / f"thomson_raw{tag}_d3d.npz", args.workers)
    mi = pd.read_parquet(CACHE / f"features_v5_{SPLIT}.parquet").sort_values("start")
    jobs = [(str(DATA / SPLIT / f), int(n), args.quantity) for f, n in zip(mi.filename, mi.n)]
    build(jobs, int(mi.n.sum()), CACHE / f"thomson_raw{tag}_mast{SSFX}.npz", args.workers)
