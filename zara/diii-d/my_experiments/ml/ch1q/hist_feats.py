"""Long input-history features per EFIT frame, from test-available inputs only (magnetics + Thomson + EFIT time stamps).

Per shot (all signals in the shot's Ip > 0 frame, like c1/features.py):
  Ip (2 kHz, data_fixes time axis) and the 19 coil currents + bcoil (20 kHz) are averaged on a 1 ms grid; causal exponential
  moving averages (EWMA) with time constants tau are evaluated at every EFIT time t.
  ip_lag_<tau>      EWMA_tau(Ip)(t) - Ip(t)              [MA]   tau = 0.05, 0.15, 0.4, 1.0, 2.5 s
  ip_lead_<tau>     anti-causal EWMA (future) - Ip(t)    [MA]   tau = 0.1, 0.4 s
  ip_runmax, ip_over_runmax, t_since_runmax, t_since_bd (breakdown = first |Ip| > 0.1 MA), ip_shotmax, t_to_end (last time
  |Ip| > 0.1 MA minus t), ip_int (integral of Ip since breakdown, MA s), ip_int1s (last 1 s)
  coil_lag_<c>_<tau>  EWMA_tau(I_c)(t) - I_c(t)           [1e5 A]  tau = 0.3, 1.5 s, 19 coils
  bt_lag_1.0
  EFIT time stamps: dt_prev, dt_next (s, capped at 2 s; edges = 2), t_since_first, t_to_last (s), ip_prev_frame - ip
  Thomson (from the per-frame cache X; NaN frames skipped, time-weighted EWMA over EFIT frames, tau = 0.1, 0.4, 1.5 s):
  te0 (mean coreTe z 0.05-0.20, keV-ish units of the cache), te0^1.5, te peaking te0 / coreTe(z 0.5), core_pe, tan_pe,
  plus ts_age_core / ts_age_tan (s since the last frame with a profile, capped at 2 s)

    python hist_feats.py --part train|test --workers 16
-> $FEQ_CACHE/ml/ch1q/hist_<part>.npz: H (N, F) float32 in the frame order of the ml/ch1s input exports
   (sorted stems, frames in efit order), names, stems, t (checked against the export).
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data_fixes import fix_d3d_ip_times  # noqa: E402

DATA = {"train": ROOT / "challenge_data" / "data" / "diii_d_train", "test": ROOT / "challenge_data" / "data" / "diii_d_public_test"}
CACHE = {"train": Path(f"{FEQ_CACHE}/nn/diii_d_train"), "test": Path(f"{FEQ_CACHE}/nn/diii_d_public_test")}
OUT = Path(f"{FEQ_CACHE}/ml/ch1q")
COILS = [f"magnetics_F{i}{s}" for s in "AB" for i in range(1, 10)] + ["magnetics_ECOILA"]
TAU_IP = (0.05, 0.15, 0.4, 1.0, 2.5)
TAU_LEAD = (0.1, 0.4)
TAU_COIL = (0.3, 1.5)
TAU_TS = (0.1, 0.4, 1.5)
DT = 1.0  # ms grid


def names():
    n = [f"ip_lag_{t}" for t in TAU_IP] + [f"ip_lead_{t}" for t in TAU_LEAD]
    n += ["ip_runmax", "ip_over_runmax", "t_since_runmax", "t_since_bd", "ip_shotmax", "t_to_end", "ip_int", "ip_int1s"]
    n += [f"coil_lag_{c[10:]}_{t}" for t in TAU_COIL for c in COILS] + ["bt_lag_1.0"]
    n += ["dt_prev", "dt_next", "t_since_first", "t_to_last", "ip_prev_frame_diff"]
    for s in ("te0", "te0_15", "te_peak", "core_pe", "tan_pe"):
        n += [f"{s}_now"] + [f"{s}_ewma_{t}" for t in TAU_TS]
    n += ["ts_age_core", "ts_age_tan"]
    return n


def flat(tbl, c):
    return pc.list_flatten(tbl.column(c)).to_numpy(zero_copy_only=False).astype(np.float64)


def bin_mean(t, v, grid):
    """mean of finite samples in [g - DT/2, g + DT/2); empty bins -> linear interpolation of the bin means; outside -> edge."""
    ok = np.isfinite(v) & np.isfinite(t)
    t, v = t[ok], v[ok]
    if t.size < 2:
        return np.full(grid.shape, np.nan)
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    c = np.r_[0.0, np.cumsum(v)]
    lo = np.searchsorted(t, grid - DT / 2, "left"); hi = np.searchsorted(t, grid + DT / 2, "left")
    n = hi - lo
    m = n > 0
    out = np.full(grid.shape, np.nan)
    out[m] = (c[hi[m]] - c[lo[m]]) / n[m]
    if (~m).any() and m.sum() >= 2:
        out[~m] = np.interp(grid[~m], grid[m], out[m])
    elif m.sum() < 2:
        out = np.interp(grid, t, v)
    return out


def ewma(x, tau_ms, reverse=False):
    a = np.exp(-DT / tau_ms)
    xx = x[::-1] if reverse else x
    y, _ = lfilter([1 - a], [1, -a], xx, zi=[a * xx[0]])
    return y[::-1] if reverse else y


def ts_ewma(t_s, v, tau):
    """time-weighted EWMA over EFIT frames, skipping NaN values; NaN until the first finite value."""
    out = np.full(len(v), np.nan)
    s = w = 0.0
    tp = None
    for i in range(len(v)):
        if tp is not None:
            d = np.exp(-(t_s[i] - tp) / tau)
            s *= d; w *= d
        tp = t_s[i]
        if np.isfinite(v[i]):
            s += v[i]; w += 1.0
        if w > 1e-6:
            out[i] = s / w
    return out


def ts_age(t_s, v):
    out = np.full(len(v), 2.0)
    last = None
    for i in range(len(v)):
        if np.isfinite(v[i]):
            last = t_s[i]
        if last is not None:
            out[i] = min(2.0, t_s[i] - last)
    return out


def shot(args):
    part, stem, cnames = args
    tbl = pq.read_table(DATA[part] / f"{stem}.parquet",
                        columns=["efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
                                 "magnetics_bcoil"] + COILS)
    row = {"magnetics_plasma_current_times": flat(tbl, "magnetics_plasma_current_times"),
           "magnetics_plasma_current": flat(tbl, "magnetics_plasma_current"), "magnetics_time": flat(tbl, "magnetics_time")}
    te = flat(tbl, "efit_times")
    tm = row["magnetics_time"]
    ipt = np.asarray(fix_d3d_ip_times(row), np.float64)
    ipraw = row["magnetics_plasma_current"]
    t0 = max(min(np.nanmin(ipt), np.nanmin(tm)), te[0] - 8000.0)
    t1 = te[-1] + 1500.0
    grid = np.arange(np.floor(t0), np.ceil(t1) + DT, DT)
    ip = bin_mean(ipt, ipraw, grid) * 1e-3                       # kA -> MA
    a_ = np.abs(np.nan_to_num(ip))
    s = 1.0 if a_.max() == 0 else float(np.sign(np.median(ip[a_ > 0.2 * a_.max()])) or 1.0)
    ip = np.nan_to_num(s * ip)
    ie = np.clip(np.searchsorted(grid, te), 0, len(grid) - 1)    # grid index of each EFIT time
    ip_e = ip[ie]
    F = []
    for tau in TAU_IP:
        F.append(ewma(ip, tau * 1e3)[ie] - ip_e)
    for tau in TAU_LEAD:
        F.append(ewma(ip, tau * 1e3, reverse=True)[ie] - ip_e)
    runmax = np.maximum.accumulate(ip)
    # time of the running max: last index where ip == runmax up to i
    idx = np.arange(len(ip)); at = np.where(ip >= runmax - 1e-9, idx, 0); at = np.maximum.accumulate(at)
    above = np.abs(ip) > 0.1
    bd = int(np.argmax(above)) if above.any() else 0
    end = len(ip) - 1 - int(np.argmax(above[::-1])) if above.any() else len(ip) - 1
    cint = np.cumsum(np.where(idx >= bd, ip, 0.0)) * DT * 1e-3
    i1 = np.clip(ie - int(1000 / DT), 0, None)
    F += [runmax[ie], ip_e / np.maximum(runmax[ie], 0.05), (ie - at[ie]) * DT * 1e-3, (ie - bd) * DT * 1e-3,
          np.full(len(te), ip.max()), (end - ie) * DT * 1e-3, cint[ie], cint[ie] - cint[i1]]
    for tau in TAU_COIL:
        for c in COILS:
            v = bin_mean(tm, flat(tbl, c), grid) * s * 1e3 / 1e5     # kA(-turn) -> 1e5 A
            v = np.nan_to_num(v)
            F.append(ewma(v, tau * 1e3)[ie] - v[ie])
    bt = np.nan_to_num(np.abs(bin_mean(tm, flat(tbl, "magnetics_bcoil"), grid))) * 1e3 / 1e5
    F.append(ewma(bt, 1000.0)[ie] - bt[ie])
    ts = te * 1e-3
    dtp = np.r_[2.0, np.minimum(np.diff(ts), 2.0)]; dtn = np.r_[np.minimum(np.diff(ts), 2.0), 2.0]
    F += [dtp, dtn, ts - ts[0], ts[-1] - ts, np.r_[0.0, ip_e[:-1] - ip_e[1:]]]
    with np.load(CACHE[part] / f"{stem}.npz") as d:
        X = d["X"].astype(np.float64); tc = d["t"].astype(np.float64)
    assert len(tc) == len(te) and np.allclose(tc, te), f"{stem}: cache times differ from efit_times"
    col = {n: i for i, n in enumerate(cnames)}
    te0 = np.nanmean(X[:, [col[f"coreTe_z{z:.2f}"] for z in (0.05, 0.10, 0.15, 0.20)]], 1) if True else None
    te5 = X[:, col["coreTe_z0.50"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = {"te0": te0, "te0_15": np.where(te0 > 0, te0, 0.0) ** 1.5 * np.where(np.isfinite(te0), 1, np.nan),
               "te_peak": np.where(te5 > 0.02 * np.nanmax(np.r_[te5, 1e-9]), te0 / te5, np.nan),
               "core_pe": X[:, col["core_pe"]], "tan_pe": X[:, col["tan_pe"]]}
    for k in ("te0", "te0_15", "te_peak", "core_pe", "tan_pe"):
        v = sig[k]
        F.append(v)
        for tau in TAU_TS:
            F.append(ts_ewma(ts, v, tau))
    F += [ts_age(ts, te0), ts_age(ts, X[:, col["tan_pe"]])]
    H = np.column_stack(F).astype(np.float32)
    return stem, H, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="test", choices=["train", "test"])
    ap.add_argument("--split", default="", help="a new inputs-only split (challenge_data/data/<split>, cache/nn/<split>); "
                    "frame order of cache/ml/ch1q/inp_rfull_<split>.npz (export_inputs_q.py) -> hist_<split>.npz")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()
    cnames = json.loads((CACHE["train"] / "meta.json").read_text())["names"]
    if a.split:                                              # new fold: register its paths under its own name
        DATA[a.split] = ROOT / "challenge_data" / "data" / a.split
        CACHE[a.split] = Path(f"{FEQ_CACHE}/nn") / a.split
        a.part = a.split
        exp = np.load(OUT / f"inp_rfull_{a.split}.npz")
    else:
        exp = np.load(f"{FEQ_CACHE}/ml/ch1s/inp_rfull_{a.part}.npz")
    stems = [str(s) for s in exp["stems"]]
    if a.max_shots:
        stems = stems[: a.max_shots]
    t0 = time.time()
    res = {}
    with Pool(a.workers) as p:
        for i, (s, H, te) in enumerate(p.imap_unordered(shot, [(a.part, s, cnames) for s in stems], chunksize=4)):
            res[s] = (H, te)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    H = np.concatenate([res[s][0] for s in stems]); T = np.concatenate([res[s][1] for s in stems])
    n = names()
    assert H.shape[1] == len(n), (H.shape, len(n))
    if not a.max_shots:
        assert len(T) == len(exp["t"]) and np.allclose(T, exp["t"]), "frame order differs from the ch1s export"
    fin = np.isfinite(H).mean(0)
    for nm, f_ in zip(n, fin):
        if f_ < 0.999:
            print(f"  {nm}: finite fraction {f_:.4f}")
    out = OUT / (f"hist_{a.part}.npz" if not a.max_shots else f"hist_{a.part}_smoke.npz")
    np.savez(out, H=H, names=np.array(n), stems=np.array(stems), t=T)
    q = np.nanpercentile(H, [1, 50, 99], axis=0)
    for j, nm in enumerate(n):
        print(f"  {nm:28s} p1 {q[0, j]:10.4f} p50 {q[1, j]:10.4f} p99 {q[2, j]:10.4f}")
    print(f"{a.part}: {len(stems)} shots, {len(T):,} frames, {H.shape[1]} features, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
