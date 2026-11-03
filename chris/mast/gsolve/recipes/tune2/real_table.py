"""Real MAST frame table for the critic (inputs only): every EFIT frame of the 1,206 public-test shots.

Model inputs come from the exact features_v5 cache (limiter variant, solenoid x0.5); history rows are i-4 / i+4 (the
model's K=1 window). Relation / noise / outside-window columns are computed per Thomson slice (critic.slice_features)
and interpolated to the EFIT times the same way the model interpolates its Thomson columns. As a check of the
synthetic feature path, the model Thomson columns are recomputed the same way and compared with the cache.

Reads efit_times / magnetics / thomson columns only (no efit label column).
usage: OMP_NUM_THREADS=1 python real_table.py [--workers 16]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import critic as cr

C = Path("/fsx/chris.beckham/fusion_cache")
REAL = Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/mast_public_test")
OUT = Path(__file__).resolve().parent / "real"
FEATS = json.load(open(C / "features_v5.json"))["features"]
SL = [c for c in cr.SLICE_COLS]


def _interp_cols(tt, rows, et, max_gap=np.inf):
    """Per column: linear interpolation over finite slices, NaN outside the slice range (no gap limit: the model's rule)."""
    out = np.full((len(et), rows.shape[1]), np.nan)
    for k in range(rows.shape[1]):
        ok = np.isfinite(rows[:, k]) & np.isfinite(tt)
        if ok.sum() < 2:
            continue
        t, v = tt[ok], rows[ok, k]
        o = np.argsort(t, kind="mergesort"); t, v = t[o], v[o]
        y = np.interp(et, t, v, left=np.nan, right=np.nan)
        j = np.clip(np.searchsorted(t, et), 1, len(t) - 1)
        gap = t[j] - t[j - 1]
        y[gap > max_gap] = np.nan
        out[:, k] = y
    return out


def one(fname):
    cols = ["efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_tf_current", "thomson_core_times",
            "thomson_core_Te", "thomson_core_ne", "thomson_core_R"]
    r = pq.read_table(REAL / fname, columns=cols).to_pylist()[0]
    et = np.asarray(r["efit_times"], float)
    tm = np.asarray(r["magnetics_time"], float)
    ipr = np.asarray(r["magnetics_plasma_current"], float)
    def fin(v):
        v = np.asarray(v, float); ok = np.isfinite(v) & np.isfinite(tm)
        return np.interp(et, tm[ok], v[ok]) if ok.sum() >= 2 else np.full(len(et), np.nan)
    ip = fin(ipr)
    tf = fin(r["magnetics_tf_current"])
    rows = np.full((0, len(SL)), np.nan)
    tt = np.zeros(0)
    if r["thomson_core_Te"]:
        Rc = np.asarray(r["thomson_core_R"], float)
        Te = np.array([np.asarray(x, float) for x in r["thomson_core_Te"]])
        Ne = np.array([np.asarray(x, float) for x in r["thomson_core_ne"]])
        if Te.shape[1] == len(Rc) + 1:
            Te, Ne = Te[:, 1:], Ne[:, 1:]
        tt = np.asarray(r["thomson_core_times"], float)
        rows = np.array([[d[c] for c in SL] for d in (cr.slice_features(Rc, Te[k], Ne[k], 0.0) for k in range(len(tt)))])
    # Every column is interpolated over the slices the model itself uses (all model Thomson columns finite), between the
    # two bracketing slices only: a column that is NaN on either bracketing slice stays NaN at the frame. A synthetic frame
    # copies one profile to its bracketing slices, so the same rule gives it the single-profile value.
    mcols = [SL.index(c) for c in ["pe_pa"] + cr.MODEL_TH[1:]]
    fr = np.full((len(et), len(SL)), np.nan)
    if len(tt):
        good = np.isfinite(rows[:, mcols]).all(1) & np.isfinite(tt)
        if good.sum() >= 2:
            t, v = tt[good], rows[good]
            o = np.argsort(t, kind="mergesort"); t, v = t[o], v[o]
            j = np.clip(np.searchsorted(t, et), 1, len(t) - 1)
            w = ((et - t[j - 1]) / np.where(t[j] > t[j - 1], t[j] - t[j - 1], 1.0))[:, None]
            inside = (et >= t[0]) & (et <= t[-1])
            fr = np.where(inside[:, None], v[j - 1] * (1 - w) + v[j] * w, np.nan)
            ex = np.isin(et, t)                       # exactly on a slice: that slice's values
            if ex.any():
                fr[ex] = v[np.searchsorted(t, et[ex])]
    d = pd.DataFrame(fr, columns=SL)
    ipm = np.maximum(np.abs(ip), 100.0) / 1e3
    d["rf_te0_ip"] *= (100.0 / 1e3) / ipm          # slices were computed with Ip floor 100 kA
    d["rf_ne0_gw"] *= (100.0 / 1e3) / ipm
    d["betaN_e"] = cr.betaN_e(d["pe_pa"].values, ip, tf)
    d.insert(0, "frame", np.arange(len(et)))
    d.insert(0, "t", et)
    d.insert(0, "filename", fname)
    d["ip_kA"], d["tf_kA"] = ip, tf
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    idx = pq.read_table(C / "features_v5_mast_public_test_sol0.5_limiter.parquet").to_pandas()
    if a.limit:
        idx = idx.head(a.limit)
    X = np.load(C / "features_v5_mast_public_test_sol0.5_limiter.npy", mmap_mode="r")
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(one, idx.filename, chunksize=4))
    keep_model = cr.COIL + cr.MODEL_TH + ["thomson_missing"]
    jj = [FEATS.index(c) for c in keep_model]
    out = []
    for r, d in zip(idx.itertuples(), parts):
        assert len(d) == r.n, r.filename
        b = np.asarray(X[r.start:r.start + r.n])[:, jj]
        i = np.arange(r.n)
        ip_, in_ = np.clip(i - 4, 0, r.n - 1), np.clip(i + 4, 0, r.n - 1)
        et = d.t.values
        m = pd.DataFrame(b, columns=[f"m:{c}" for c in keep_model])
        mine = d[["betaN_e"] + cr.MODEL_TH[1:]].rename(columns=lambda c: f"chk:{c}")
        e = d[cr.REL + cr.NOISE + cr.OUTWIN].reset_index(drop=True)
        now = pd.concat([m, e], axis=1)
        fr = pd.DataFrame({"shot": r.filename.replace(".parquet", "").replace("mast_shot_", ""), "frame": i, "t": et,
                           "lag_prev": et - et[ip_], "lag_next": et[in_] - et, "ip_kA": d.ip_kA.values, "tf_kA": d.tf_kA.values})
        prev = now.iloc[ip_].reset_index(drop=True).add_prefix("prev:")
        nxt = now.iloc[in_].reset_index(drop=True).add_prefix("next:")
        out.append(pd.concat([fr, now, prev, nxt, mine.reset_index(drop=True)], axis=1))
    T = pd.concat(out, ignore_index=True)
    T.to_parquet(OUT / "real_frames.parquet")
    # check: recomputed model Thomson columns vs the cache
    rep = {}
    for c in cr.MODEL_TH:
        a_, b_ = T[f"m:{c}"].values, T[f"chk:{c}"].values
        ok = np.isfinite(a_) & np.isfinite(b_)
        rep[c] = dict(n=int(ok.sum()), both_nan=int((~np.isfinite(a_) & ~np.isfinite(b_)).sum()),
                      only_cache=int((np.isfinite(a_) & ~np.isfinite(b_)).sum()), only_mine=int((~np.isfinite(a_) & np.isfinite(b_)).sum()),
                      max_abs=float(np.nanmax(np.abs(a_[ok] - b_[ok]))) if ok.any() else None,
                      p99_abs=float(np.percentile(np.abs(a_[ok] - b_[ok]), 99)) if ok.any() else None)
    (OUT / "check_model_th.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(T.shape)


if __name__ == "__main__":
    main()
