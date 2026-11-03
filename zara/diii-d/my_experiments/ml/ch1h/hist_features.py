"""Long-history input features ("Group H") for DIII-D, computed from test-available input columns only.

    python hist_features.py --split diii_d_train --workers 16 [--max-shots N]
-> $FEQ_CACHE/ml/ch1h/hist/<split>/<stem>.npz   H (T, F) float32 (NaN = not computable), t (T,) ms
   $FEQ_CACHE/ml/ch1h/hist/<split>/meta.json    names, two_sided (names that use samples after the frame time)
   $FEQ_CACHE/ml/ch1h/hist/<split>/check.json   per-shot agreement with the nn cache (T, sign, Ip at the frames)

Inputs used: magnetics_plasma_current (+ the corrected time axis data_fixes.fix_d3d_ip_times), the 19 coil currents on
magnetics_time, Thomson core / tangential Te and ne, efit_times, coil geometry. No EFIT output column is read.
Every shot is put in its Ip > 0 frame with the sign rule of c1/features.py (currents multiplied by the same sign).

Signals on a uniform 1 ms grid from t = -1000 ms to the last frame + 1500 ms (Ip = 0 outside its trace; coil currents
held at their first / last value). EWMA = y[n] = a y[n-1] + (1 - a) x[n], a = exp(-1 ms / tau), started at x(-1000 ms).
Causal unless listed in `two_sided`:
  A  ip_d_ewm{tau}      Ip(t) - EWMA_tau(Ip) [MA] (= tau x the EWMA of dIp/dt), tau = 0.1, 0.3, 1, 3 s
     ip_runmax          running maximum of the 10 ms-smoothed Ip [MA]
     t_since_max98/90   time since Ip was last >= 0.98 / 0.90 of its running maximum [s] (0 in ramp-up / flat-top)
     t_since_up/down    time since the 20 ms-smoothed dIp/dt was last > +0.3 / < -0.3 MA/s [s, capped 5]
     ip_int             integral of Ip since breakdown (first Ip > 0.1 MA) [MA s]
  B  vl_ewm{tau}        EWMA of the coil loop voltage at (1.70, 0) m, -2 pi d(psi_coils)/dt [V] (as Group A's vloop_*)
     dpsi_bd            applied coil flux change at (1.70, 0) since breakdown, 2 pi (psi_c(t) - psi_c(t_bd)) [Wb]
     dpsi_bd_over_ip    dpsi_bd / max(Ip, 0.05 MA)
  C  <coil>_d_ewm{tau}  coil current minus its EWMA [1e5 A], tau = 0.3, 1 s, 19 coils
  D  Thomson per raw profile (channels with Te > 0; >= 5 valid; moments by trapezoid along the chord coordinate):
     core (vertical chord, coordinate Z): te_max (90th percentile, keV), zc15 / wz15 (Te^1.5-weighted centroid / rms width
     [m], Spitzer-conductivity proxy of the current profile), wzte (Te-weighted rms width), ne_avg (1e19 m^-3);
     tangential (coordinate R): te_max, rc15, wr15.
     For each: the value at the frame (mean of profiles within +-12 ms) and causal EWMAs with tau = 0.1, 0.3, 1 s over the
     raw profiles up to the frame time; core_age / tan_age = time since the last valid profile [s, capped 1].
  E  two-sided: t_to_last (last EFIT time - t) [s], shot_frac (position between the first and last EFIT time),
     t_to_ipend (last time Ip > 0.1 MA - t) [s], ipfut_d{tau} = anti-causal EWMA(Ip) - Ip [MA], tau = 0.1, 0.3 s.
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C1 = HERE.parents[1] / "c1"
DATA = ROOT / "challenge_data" / "data"
OUT = Path(f"{FEQ_CACHE}/ml/ch1h/hist")
NN_CACHE = Path(f"{FEQ_CACHE}/nn")

TAU_IP = (0.1, 0.3, 1.0, 3.0)
TAU_VL = (0.1, 0.3, 1.0, 3.0)
TAU_COIL = (0.3, 1.0)
TAU_TS = (0.1, 0.3, 1.0)
TAU_FUT = (0.1, 0.3)
DT = 1.0                       # grid step, ms
T_LO = -1000.0                 # grid start, ms
TS_WIN = 12.0                  # ms
A_E = -152.8                   # E-coil scale (as c1/nn/physics_plus.py)
R0, Z0 = 1.70, 0.0
CORE_STATS = ("te_max", "zc15", "wz15", "wzte", "ne_avg")
TAN_STATS = ("te_max", "rc15", "wr15")
COLUMNS = None                 # read every column (the parquet rows are small enough); EFIT columns are never used


def names_list(cols):
    n = [f"ip_d_ewm{int(t * 1000)}" for t in TAU_IP]
    n += ["ip_runmax", "t_since_max98", "t_since_max90", "t_since_up", "t_since_down", "ip_int"]
    n += [f"vl_ewm{int(t * 1000)}" for t in TAU_VL] + ["dpsi_bd", "dpsi_bd_over_ip"]
    n += [f"{c.replace('magnetics_', '')}_d_ewm{int(t * 1000)}" for t in TAU_COIL for c in cols]
    for pre, stats in (("core", CORE_STATS), ("tan", TAN_STATS)):
        for s in stats:
            n += [f"{pre}_{s}"] + [f"{pre}_{s}_ewm{int(t * 1000)}" for t in TAU_TS]
    n += ["core_age", "tan_age"]
    two = ["t_to_last", "shot_frac", "t_to_ipend"] + [f"ipfut_d{int(t * 1000)}" for t in TAU_FUT]
    return n + two, two


def ewm_uniform(x, tau_s, reverse=False):
    """EWMA along axis 0 of a uniformly sampled (N, ...) array (step DT ms), started at the first value."""
    from scipy.signal import lfilter
    a = np.exp(-DT / (tau_s * 1000.0))
    y = x[::-1] if reverse else x
    zi = a * y[:1]
    out, _ = lfilter([1.0 - a], [1.0, -a], y, axis=0, zi=zi)
    return out[::-1] if reverse else out


def ewm_irregular(ts, xs, tq, tau_s):
    """Causal EWMA of irregular samples (ts ms, xs) at query times tq: sum w x / sum w over ts <= tq, w = exp(-(tq-ts)/tau).
    NaN where no sample precedes tq."""
    ok = np.isfinite(xs)
    ts, xs = ts[ok], xs[ok]
    out = np.full(len(tq), np.nan)
    if ts.size == 0:
        return out
    tref = max(float(ts.max()), float(np.max(tq)))
    e = np.exp((ts - tref) / (tau_s * 1000.0))
    A = np.cumsum(xs * e)
    B = np.cumsum(e)
    k = np.searchsorted(ts, tq, side="right") - 1
    m = k >= 0
    with np.errstate(all="ignore"):
        out[m] = A[k[m]] / B[k[m]]
    return out


def window_mean_valid(ts, xs, tq, half):
    ok = np.isfinite(xs)
    ts, xs = ts[ok], xs[ok]
    out = np.full(len(tq), np.nan)
    if ts.size == 0:
        return out
    c = np.concatenate([[0.0], np.cumsum(xs)])
    lo = np.searchsorted(ts, tq - half, side="left")
    hi = np.searchsorted(ts, tq + half, side="right")
    n = hi - lo
    m = n > 0
    out[m] = (c[hi[m]] - c[lo[m]]) / n[m]
    return out


def last_true_age(cond, grid, tq, cap):
    """Time (s) since cond was last True on the grid, at the query times (cap if never)."""
    idx = np.where(cond, np.arange(len(cond)), -1)
    last = np.maximum.accumulate(idx)
    k = np.clip(np.searchsorted(grid, tq, side="right") - 1, 0, len(grid) - 1)
    lk = last[k]
    out = np.where(lk >= 0, (tq - grid[np.maximum(lk, 0)]) / 1000.0, cap)
    return np.minimum(out, cap)


def ts_stats(row, prefix, key, coord):
    """Per raw Thomson profile: statistics (Nt, S) and times (ms), sorted by time."""
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    sel = np.char.startswith(names, prefix)
    pos = np.asarray(row["thomson_chord_Z" if coord == "Z" else "thomson_chord_R"], float)[sel]
    times = np.asarray(row[f"thomson_{key}_times"], float)
    S = len(CORE_STATS) if coord == "Z" else len(TAN_STATS)
    try:
        Te = np.stack(row[f"thomson_{key}_Te"]).astype(float)
        ne = np.stack(row[f"thomson_{key}_ne"]).astype(float)
    except (ValueError, TypeError):
        return np.zeros(0), np.zeros((0, S))
    if Te.ndim != 2 or Te.shape[1] != pos.size or Te.shape[0] != times.size or pos.size == 0:
        return np.zeros(0), np.zeros((0, S))
    o = np.argsort(times)
    times, Te, ne = times[o], Te[o] / 1e3, ne[o] / 1e19
    oc = np.argsort(pos)
    pos, Te, ne = pos[oc], Te[:, oc], ne[:, oc]
    Te[~(Te > 0)] = np.nan
    ne[~(ne > 0)] = np.nan
    out = np.full((len(times), S), np.nan)
    with np.errstate(all="ignore"):
        for i in range(len(times)):
            v = np.isfinite(Te[i])
            if v.sum() < 5:
                continue
            x, te = pos[v], Te[i, v]
            if np.nanmax(te) < 0.02:
                continue
            w15 = te ** 1.5
            n15 = np.trapezoid(w15, x)
            nte = np.trapezoid(te, x)
            if not (n15 > 0 and nte > 0):
                continue
            c15 = np.trapezoid(x * w15, x) / n15
            s15 = np.sqrt(max(np.trapezoid((x - c15) ** 2 * w15, x) / n15, 0.0))
            cte = np.trapezoid(x * te, x) / nte
            ste = np.sqrt(max(np.trapezoid((x - cte) ** 2 * te, x) / nte, 0.0))
            tmax = np.percentile(te, 90)
            if coord == "Z":
                vn = np.isfinite(ne[i])
                navg = np.trapezoid(ne[i, vn], pos[vn]) / (pos[vn].max() - pos[vn].min()) if vn.sum() >= 5 else np.nan
                out[i] = [tmax, c15, s15, ste, navg]
            else:
                out[i] = [tmax, c15, s15]
    return times, out


def shot_hist(row, cols):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(C1))
    from data_fixes import fix_d3d_ip_times
    from features import KA, window_mean
    from greens import coil_filaments, filament_psi

    te = np.asarray(row["efit_times"], float)
    tm = np.asarray(row["magnetics_time"], float)
    ip_t = np.asarray(fix_d3d_ip_times(row), float)
    ip_raw = np.asarray(row["magnetics_plasma_current"], float) * KA          # A
    # sign rule of c1/features.py (window mean +-1 ms at the EFIT times)
    ip_fr = window_mean(ip_t, ip_raw, te, 1.0)
    a = np.abs(np.nan_to_num(ip_fr))
    s = 1.0 if a.max() == 0 else float(np.sign(np.median(ip_fr[a > 0.2 * a.max()])) or 1.0)
    grid = np.arange(T_LO, te[-1] + 1500.0 + DT / 2, DT)
    # Ip on the grid (MA), 0 outside the trace
    ok = np.isfinite(ip_raw)
    ipg = window_mean(ip_t[ok], s * ip_raw[ok] / 1e6, grid, 0.5)
    inside = (grid >= ip_t[ok][0]) & (grid <= ip_t[ok][-1])
    gap = ~np.isfinite(ipg)
    if gap.any():
        ipg[gap] = np.interp(grid[gap], ip_t[ok], s * ip_raw[ok] / 1e6)
    ipg[~inside] = 0.0
    # coil currents on the grid (A-turn, Ip > 0 frame), held at the ends
    Ig = np.empty((len(grid), len(cols)))
    for j, c in enumerate(cols):
        v = np.asarray(row[c], float) * KA * s
        okc = np.isfinite(v)
        g = window_mean(tm[okc], v[okc], grid, 0.5)
        bad = ~np.isfinite(g)
        if bad.any():
            g[bad] = np.interp(grid[bad], tm[okc], v[okc])
        Ig[:, j] = g
    T = len(te)
    kq = np.clip(np.round((te - T_LO) / DT).astype(int), 0, len(grid) - 1)
    feats = []
    ipq = ipg[kq]
    # A: Ip history
    for tau in TAU_IP:
        feats.append(ipq - ewm_uniform(ipg, tau)[kq])
    k10 = int(round(10 / DT))
    ips = np.convolve(ipg, np.ones(k10) / k10, mode="same")
    runmax = np.maximum.accumulate(ips)
    feats.append(runmax[kq])
    for f in (0.98, 0.90):
        feats.append(last_true_age(ips >= f * np.maximum(runmax, 1e-3), grid, te, 10.0))
    k20 = int(round(20 / DT))
    ips20 = np.convolve(ipg, np.ones(k20) / k20, mode="same")
    dip = np.gradient(ips20, DT / 1000.0)                                   # MA/s
    feats.append(last_true_age(dip > 0.3, grid, te, 5.0))
    feats.append(last_true_age(dip < -0.3, grid, te, 5.0))
    above = np.nonzero(ips > 0.1)[0]
    kbd = int(above[0]) if above.size else None
    ipint = np.cumsum(np.where(np.arange(len(grid)) >= (kbd if kbd is not None else len(grid)), ipg, 0.0)) * DT / 1000.0
    feats.append(ipint[kq] if kbd is not None else np.full(T, np.nan))
    # B: coil flux at (R0, Z0) and the loop voltage history
    fil = np.zeros(len(cols))
    for i in range(len(cols)):
        fr, fz = coil_filaments(float(row["coil_R"][i]), float(row["coil_Z"][i]), float(row["coil_width"][i]),
                                float(row["coil_height"][i]), float(row["coil_angle1"][i]), float(row["coil_angle2"][i]), 8)
        sc = A_E if cols[i] == "magnetics_ECOILA" else -1.0
        fil[i] = filament_psi(fr, fz, np.full_like(fr, R0), np.full_like(fz, Z0)).mean() * sc
    psic = Ig @ fil
    for tau in TAU_VL:
        feats.append(-2 * np.pi * (psic - ewm_uniform(psic, tau))[kq] / tau)
    if kbd is not None:
        dps = 2 * np.pi * (psic[kq] - psic[kbd])
        feats += [dps, dps / np.maximum(ipq, 0.05)]
    else:
        feats += [np.full(T, np.nan)] * 2
    # C: coil current deviations from their EWMAs
    for tau in TAU_COIL:
        feats += list(((Ig - ewm_uniform(Ig, tau))[kq] / 1e5).T)
    # D: Thomson profile shape history
    ages = []
    for prefix, key, coord, stats in (("TS_core", "core", "Z", CORE_STATS), ("TS_tangential", "edge", "R", TAN_STATS)):
        tt, st = ts_stats(row, prefix, key, coord)
        for j in range(len(stats)):
            if tt.size:
                feats.append(window_mean_valid(tt, st[:, j], te, TS_WIN))
                feats += [ewm_irregular(tt, st[:, j], te, tau) for tau in TAU_TS]
            else:
                feats += [np.full(T, np.nan)] * (1 + len(TAU_TS))
        good = np.isfinite(st).all(1) if tt.size else np.zeros(0, bool)
        if good.any():
            tg = tt[good]
            k = np.searchsorted(tg, te, side="right") - 1
            ages.append(np.where(k >= 0, np.minimum((te - tg[np.maximum(k, 0)]) / 1000.0, 1.0), 1.0))
        else:
            ages.append(np.full(T, 1.0))
    feats += ages
    # E: two-sided
    feats.append((te[-1] - te) / 1000.0)
    feats.append((te - te[0]) / max(te[-1] - te[0], 1.0))
    endk = np.nonzero(ips > 0.1)[0]
    feats.append((grid[endk[-1]] - te) / 1000.0 if endk.size else np.full(T, np.nan))
    for tau in TAU_FUT:
        feats.append(ewm_uniform(ipg, tau, reverse=True)[kq] - ipq)
    H = np.column_stack(feats).astype(np.float32)
    return H, s, ipq * 1e6


def one(args):
    f, out_dir, cache_dir = args
    import pandas as pd
    stem = Path(f).stem
    out = Path(out_dir) / f"{stem}.npz"
    row = pd.read_parquet(f).iloc[0].to_dict()
    cols = [str(c) for c in row["coil_input_column"]]
    H, s, ipq = shot_hist(row, cols)
    chk = {"stem": stem, "T": int(H.shape[0])}
    cf = Path(cache_dir) / f"{stem}.npz"
    if cf.exists():
        with np.load(cf) as d:
            chk["T_cache"] = int(len(d["ip"]))
            chk["sign_ok"] = bool(float(d["sign"]) == s)
            ipc = d["ip"].astype(np.float64)
            m = np.isfinite(ipc) & np.isfinite(ipq)
            chk["ip_maxdiff_kA"] = float(np.max(np.abs(ipc[m] - ipq[m])) / 1e3) if m.any() else None
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, H=H, t=np.asarray(row["efit_times"], np.float32))
    os.replace(tmp, out)
    return chk, cols, int(np.isfinite(H).all(0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="diii_d_train")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    files = [str(f) for f in sorted((DATA / a.split).glob("*.parquet"))]
    if a.max_shots:
        files = files[:: max(1, len(files) // a.max_shots)][: a.max_shots]
    out_dir = OUT / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks, cols = [], None
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (chk, c, nfin) in enumerate(pool.map(one, [(f, str(out_dir), str(NN_CACHE / a.split)) for f in files],
                                                    chunksize=4), 1):
            checks.append(chk)
            cols = c
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    names, two = names_list(cols)
    (out_dir / "meta.json").write_text(json.dumps({"names": names, "two_sided": two, "coils": cols,
                                                    "doc": __doc__}, indent=1))
    (out_dir / "check.json").write_text(json.dumps(checks))
    bad_T = [c["stem"] for c in checks if "T_cache" in c and c["T_cache"] != c["T"]]
    bad_s = [c["stem"] for c in checks if "sign_ok" in c and not c["sign_ok"]]
    dif = np.array([c["ip_maxdiff_kA"] for c in checks if c.get("ip_maxdiff_kA") is not None])
    print(f"{len(files)} shots in {time.time() - t0:.0f}s; {len(names)} features; T mismatches {len(bad_T)} {bad_T[:3]}; "
          f"sign mismatches {len(bad_s)} {bad_s[:3]}; Ip vs cache max diff kA: median {np.median(dif) if dif.size else -1:.3f} "
          f"p99 {np.percentile(dif, 99) if dif.size else -1:.3f} max {dif.max() if dif.size else -1:.3f}", flush=True)


if __name__ == "__main__":
    main()
