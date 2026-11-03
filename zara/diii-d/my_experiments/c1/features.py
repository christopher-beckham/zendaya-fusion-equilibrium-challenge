"""Per-frame inputs for DIII-D, aligned to `efit_times`. Uses only allowed (non-EFIT) columns.

`shot_inputs(row)` returns, in the shot's Ip > 0 frame (see `sign`):
    I      (T, 19)  PF-coil currents in A(-turn) x sign, ordered like `coil_input_column`
    ip     (T,)     plasma current x sign, A (Ip time axis corrected with data_fixes)
    sign   float    +1 or -1: the shot's Ip direction (the currents above are multiplied by it)
    X      (T, F)   feature matrix for the learned stage (NaN where an input is unavailable)
    names  (F,)     feature names
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_fixes import fix_d3d_ip_times  # noqa: E402

COIL_WINDOW_MS = 1.0          # coil currents: mean over +-1 ms (20 kHz samples are noisy)
DERIV_MS = (5.0, 20.0)        # dI/dt baselines, a handle on vessel eddy currents
TS_WINDOW_MS = 12.0           # Thomson: mean of profiles within +-12 ms of the EFIT time
# Optional (C1_TS_INTERP=1): frames without a profile in the window get the time interpolation between the nearest
# profiles within TS_INTERP_MAX_MS, plus one "profile age" feature per system. Off by default so that the pickled
# C1 models and the original nn cache keep their feature layout.
import os as _os
TS_INTERP = _os.environ.get("C1_TS_INTERP", "0") == "1"
TS_INTERP_MAX_MS = 40.0
CORE_Z = np.linspace(0.05, 0.8, 16)    # vertical core chord at R = 1.94 m, upper half only
TAN_R = np.linspace(1.7, 2.25, 12)      # tangential chords near the midplane (10-14 channels)
KA = 1e3


def window_mean(t: np.ndarray, v: np.ndarray, tq: np.ndarray, half: float) -> np.ndarray:
    """Mean of finite samples v(t) with |t - tq| <= half; NaN where the window is empty."""
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size == 0:
        return np.full(tq.shape, np.nan)
    c = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, tq - half, side="left")
    hi = np.searchsorted(t, tq + half, side="right")
    n = hi - lo
    out = np.full(tq.shape, np.nan)
    m = n > 0
    out[m] = (c[hi[m]] - c[lo[m]]) / n[m]
    # sparse signal (e.g. a 2 kHz shot with a tiny window): fall back to interpolation
    inside = (~m) & (tq >= t[0]) & (tq <= t[-1])
    out[inside] = np.interp(tq[inside], t, v)
    return out


def _interp_gaps(times, prof, tq, avg, max_ms):
    """Fill frames with no in-window profile by linear time interpolation of the raw profiles (per channel),
    when the nearest raw profiles on both sides (or one side within max_ms) are within max_ms. Returns (avg, age)."""
    age = np.zeros(len(tq))
    have = np.isfinite(avg).any(1)
    tf = times[np.isfinite(prof).any(1)]
    if tf.size == 0:
        age[:] = np.nan
        return avg, age
    k = np.searchsorted(tf, tq)
    k0 = np.clip(k - 1, 0, len(tf) - 1); k1 = np.clip(k, 0, len(tf) - 1)
    d = np.minimum(np.abs(tf[k0] - tq), np.abs(tf[k1] - tq))
    age[:] = d
    fill = (~have) & (d <= max_ms)
    for j in range(prof.shape[1]):
        ok = np.isfinite(prof[:, j])
        if ok.sum() >= 2:
            avg[fill, j] = np.interp(tq[fill], times[ok], prof[ok, j], left=np.nan, right=np.nan)
    return avg, age


def _profile_on_grid(times, prof, pos, tq, grid):
    """Time-window-average a (Nt, C) profile at tq, then resample in space onto `grid`."""
    prof = np.asarray(prof, dtype=np.float64)
    order = np.argsort(pos)
    pos, prof = np.asarray(pos)[order], prof[:, order]
    avg = np.stack([window_mean(times, prof[:, j], tq, TS_WINDOW_MS) for j in range(prof.shape[1])], 1)
    age = None
    if TS_INTERP:
        avg, age = _interp_gaps(times, prof, tq, avg, TS_INTERP_MAX_MS)
    out = np.full((len(tq), len(grid)), np.nan)
    for k in range(len(tq)):
        ok = np.isfinite(avg[k])
        if ok.sum() >= 2:
            out[k] = np.interp(grid, pos[ok], avg[k, ok], left=np.nan, right=np.nan)
    if TS_INTERP:
        return avg, pos, out, age
    return avg, pos, out


def _boundary(avg, pos, frac=0.3):
    """Position of the outermost channel whose ne is above frac * peak (plasma edge proxy)."""
    out = np.full(avg.shape[0], np.nan)
    for k in range(avg.shape[0]):
        a = avg[k]
        ok = np.isfinite(a)
        if ok.sum() < 3 or np.nanmax(a) <= 0:
            continue
        above = ok & (a > frac * np.nanmax(a))
        if above.any():
            out[k] = pos[np.nonzero(above)[0].max()]
    return out


def _thomson(row, prefix, key, grid, tq):
    """Te/ne on a fixed spatial grid, the ne-edge position and the pe line integral."""
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    sel = np.char.startswith(names, prefix)
    pos = np.asarray(row["thomson_chord_Z" if key == "core" else "thomson_chord_R"],
                     dtype=np.float64)[sel]
    times = np.asarray(row[f"thomson_{key}_times"], dtype=np.float64)
    nan_block = np.full((len(tq), 2 * len(grid) + 2 + (1 if TS_INTERP else 0)), np.nan)
    try:
        Te = np.stack(row[f"thomson_{key}_Te"]).astype(np.float64)
        ne = np.stack(row[f"thomson_{key}_ne"]).astype(np.float64)
    except (ValueError, TypeError):
        return nan_block
    if Te.ndim != 2 or Te.shape[1] != pos.size or Te.shape[0] != times.size:
        return nan_block
    order = np.argsort(times)
    times, Te, ne = times[order], Te[order], ne[order]
    Te[Te <= 0] = np.nan
    ne[ne <= 0] = np.nan
    if TS_INTERP:
        _, _, te_g, age = _profile_on_grid(times, Te / 1e3, pos, tq, grid)          # keV
        ne_avg, spos, ne_g, _ = _profile_on_grid(times, ne / 1e19, pos, tq, grid)  # 1e19 m^-3
    else:
        _, _, te_g = _profile_on_grid(times, Te / 1e3, pos, tq, grid)          # keV
        ne_avg, spos, ne_g = _profile_on_grid(times, ne / 1e19, pos, tq, grid)  # 1e19 m^-3
    edge = _boundary(ne_avg, spos)
    pe = np.nansum(te_g * ne_g, axis=1) * (grid[1] - grid[0])
    pe[~np.isfinite(te_g).any(axis=1)] = np.nan
    cols = [te_g, ne_g, edge, pe]
    if TS_INTERP:
        cols.append(np.minimum(np.nan_to_num(age, nan=200.0), 200.0) / 100.0)
    return np.column_stack(cols)


def shot_inputs(row: dict) -> dict:
    te = np.asarray(row["efit_times"], dtype=np.float64)
    tm = np.asarray(row["magnetics_time"], dtype=np.float64)
    cols = [str(c) for c in row["coil_input_column"]]

    def sig(col, times=tm, half=COIL_WINDOW_MS, at=te):
        return window_mean(times, np.asarray(row[col], dtype=np.float64), at, half)

    ip_t = np.asarray(fix_d3d_ip_times(row), dtype=np.float64)
    ip = sig("magnetics_plasma_current", times=ip_t) * KA
    # Reversed-Ip shots also reverse the coil currents, while the shipped psirz keeps one sign
    # convention, so put every shot in an Ip > 0 frame. B_T's direction does not enter psi: |B_T|.
    a = np.abs(np.nan_to_num(ip))
    s = 1.0 if a.max() == 0 else float(np.sign(np.median(ip[a > 0.2 * a.max()])) or 1.0)
    ip = s * ip
    I = s * np.column_stack([sig(c) for c in cols]) * KA
    bt = np.abs(sig("magnetics_bcoil")) * KA

    feats, names = [I / 1e5, ip[:, None] / 1e6, bt[:, None] / 1e5], \
        [*cols, "ip", "bcoil"]
    for d in DERIV_MS:                                  # backward difference over d ms
        prev_I = s * np.column_stack([sig(c, at=te - d) for c in cols]) * KA
        prev_ip = s * sig("magnetics_plasma_current", times=ip_t, at=te - d) * KA
        feats += [(I - prev_I) / d / 1e3, ((ip - prev_ip) / d / 1e3)[:, None]]
        names += [f"d{c}_{d:g}ms" for c in cols] + [f"dip_{d:g}ms"]

    core = _thomson(row, "TS_core", "core", CORE_Z, te)
    tan = _thomson(row, "TS_tangential", "edge", TAN_R, te)
    feats += [core, tan]
    names += [f"coreTe_z{z:.2f}" for z in CORE_Z] + [f"coreNe_z{z:.2f}" for z in CORE_Z] + \
             ["core_ne_top", "core_pe"] + (["core_ts_age"] if TS_INTERP else [])
    names += [f"tanTe_r{r:.2f}" for r in TAN_R] + [f"tanNe_r{r:.2f}" for r in TAN_R] + \
             ["tan_ne_outer", "tan_pe"] + (["tan_ts_age"] if TS_INTERP else [])

    base = np.column_stack(feats)
    # Plasma response is roughly bilinear in Ip and the shaping currents: add Ip x feature.
    ipn = (ip / 1e6)[:, None]
    X = np.column_stack([base, ipn * base, (te / 1e3)[:, None]])
    names = names + [f"ip*{n}" for n in names] + ["t_s"]
    return {"I": I, "ip": ip, "bt": bt, "sign": s, "cols": cols, "X": X,
            "names": np.asarray(names)}
