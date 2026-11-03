"""Extra physics-derived per-frame inputs ("Group A"), computed from the raw shot signals.

    feats, names, ts = group_a(row, inp)

`inp` is features.shot_inputs(row) (currents in the shot's Ip > 0 frame). Returns
    feats (T, F) float64   NaN where not computable
    names list[str]
    ts    dict of padded Thomson channel arrays for the first-pass mapping (Group B, train_v6.py):
          pos (C, 2) R, Z of each channel; te (T, C) keV; ne (T, C) 1e19 m^-3; C = NCH (NaN padded)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
from data_fixes import fix_d3d_ip_times  # noqa: E402
from features import COIL_WINDOW_MS, KA, window_mean  # noqa: E402
from greens import MU0, coil_filaments, filament_psi  # noqa: E402

A_E = -152.8                     # E-coil scale of the physics stage
R0, Z0 = 1.70, 0.0
N_TF = 144                       # DIII-D toroidal-field turns: B_T = mu0 N I / (2 pi R)
TS_WINDOW_MS = 12.0
NCH = 72                         # padded Thomson channels (core <= 56, tangential <= 16)
IP_FLOOR = 5e4
HIST_MS = (50.0, 100.0, 200.0, 500.0)
VLOOP_MS = (5.0, 20.0, 100.0)
_ENV = np.load(HERE.parents[2] / "fusion_scoring" / "masks" / "d3d_envelope.npz")
_GR, _GZ = _ENV["grid_R"].astype(float), _ENV["grid_Z"].astype(float)


def probe_points(n_edge: int = 32):
    """Points 2 cells inside the scoring envelope, evenly spaced in angle, plus 4 X-point-region points."""
    m = _ENV["mask_coarse"].astype(bool)
    e = m.copy()
    for _ in range(2):
        f = e.copy(); f[1:] &= e[:-1]; f[:-1] &= e[1:]; f[:, 1:] &= e[:, :-1]; f[:, :-1] &= e[:, 1:]; e = f
    inner = e.copy(); inner[1:] &= e[:-1]; inner[:-1] &= e[1:]; inner[:, 1:] &= e[:, :-1]; inner[:, :-1] &= e[:, 1:]
    iz, ir = np.nonzero(e & ~inner)
    R, Z = _GR[ir], _GZ[iz]
    ang = np.arctan2(Z - 0.0, R - 1.65)
    o = np.argsort(ang)
    pick = o[np.linspace(0, len(o) - 1, n_edge, endpoint=False).round().astype(int)]
    pts = np.column_stack([R[pick], Z[pick]])
    xpt = np.array([[1.45, -1.15], [1.45, 1.15], [1.30, -1.25], [1.30, 1.25]])
    return np.vstack([pts, xpt])


_PROBE_CACHE: dict = {}


def probe_greens(row, cols):
    """(C, P, 3): flux, dpsi/dR, dpsi/dZ per unit current of each coil at the probe points."""
    key = tuple(np.round(np.concatenate([np.asarray(row[k], float) for k in
                ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")]), 5))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    pts = probe_points()
    d = 1e-3
    out = np.zeros((len(cols), len(pts), 3))
    for i in range(len(cols)):
        fr, fz = coil_filaments(float(row["coil_R"][i]), float(row["coil_Z"][i]), float(row["coil_width"][i]),
                                float(row["coil_height"][i]), float(row["coil_angle1"][i]), float(row["coil_angle2"][i]), 8)
        def g(R, Z):
            return filament_psi(fr[:, None], fz[:, None], R[None, :], Z[None, :]).mean(0)
        R, Z = pts[:, 0], pts[:, 1]
        out[i, :, 0] = g(R, Z)
        out[i, :, 1] = (g(R + d, Z) - g(R - d, Z)) / (2 * d)
        out[i, :, 2] = (g(R, Z + d) - g(R, Z - d)) / (2 * d)
    _PROBE_CACHE[key] = (out, pts)
    return out, pts


def _thomson_channels(row, prefix, key, tq):
    """Window-mean Te (keV), ne (1e19) per channel at tq, channel positions, and profile age (ms)."""
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    sel = np.char.startswith(names, prefix)
    R = np.asarray(row["thomson_chord_R"], float)[sel]
    Z = np.asarray(row["thomson_chord_Z"], float)[sel]
    times = np.asarray(row[f"thomson_{key}_times"], float)
    C = int(sel.sum())
    empty = (np.full((len(tq), C), np.nan), np.full((len(tq), C), np.nan), R, Z, np.full(len(tq), np.nan))
    try:
        Te = np.stack(row[f"thomson_{key}_Te"]).astype(float); ne = np.stack(row[f"thomson_{key}_ne"]).astype(float)
    except (ValueError, TypeError):
        return empty
    if Te.ndim != 2 or Te.shape[1] != C or Te.shape[0] != times.size or C == 0:
        return empty
    o = np.argsort(times); times, Te, ne = times[o], Te[o], ne[o]
    Te[Te <= 0] = np.nan; ne[ne <= 0] = np.nan
    te = np.stack([window_mean(times, Te[:, j] / 1e3, tq, TS_WINDOW_MS) for j in range(C)], 1)
    nn = np.stack([window_mean(times, ne[:, j] / 1e19, tq, TS_WINDOW_MS) for j in range(C)], 1)
    good = np.isfinite(Te).any(1)
    tg = times[good]
    if tg.size:
        k = np.clip(np.searchsorted(tg, tq), 1, max(1, tg.size - 1))
        lo, hi = tg[np.clip(k - 1, 0, tg.size - 1)], tg[np.clip(k, 0, tg.size - 1)]
        age = np.minimum(np.abs(tq - lo), np.abs(tq - hi))
    else:
        age = np.full(len(tq), np.nan)
    return te, nn, R, Z, age


def _profile_stats(te, ne, s):
    """15 shape statistics of Te, ne profiles along chord coordinate s (sorted ascending)."""
    o = np.argsort(s); s = s[o]; te = te[:, o]; ne = ne[:, o]
    T = te.shape[0]
    out = np.full((T, 15), np.nan)
    with np.errstate(all="ignore"):
        for k in range(T):
            ok = np.isfinite(te[k]) & np.isfinite(ne[k])
            if ok.sum() < 3:
                continue
            ss, t, n = s[ok], te[k, ok], ne[k, ok]
            pe = t * n
            tmax, nmax = t.max(), n.max()
            tmean, nmean = t.mean(), n.mean()
            dt = np.abs(np.gradient(t, ss)); dn = np.abs(np.gradient(n, ss))
            above = np.nonzero(t > 0.3 * tmax)[0]
            out[k] = [ok.sum(), tmax, nmax, tmean, nmean, tmax / tmean, nmax / nmean,
                      np.trapezoid(n, ss), np.trapezoid(t ** 1.5, ss), np.trapezoid(pe, ss),
                      ss[np.argmax(dt)], dt.max(), ss[np.argmax(dn)], dn.max(), ss[above.max()] if above.size else np.nan]
    return out


STAT_NAMES = ["nvalid", "te_max", "ne_max", "te_mean", "ne_mean", "te_peak", "ne_peak", "ne_int", "te15_int", "pe_int",
              "s_maxdte", "maxdte", "s_maxdne", "maxdne", "s_te30"]


def group_a(row: dict, inp: dict):
    te_q = np.asarray(row["efit_times"], float)
    tm = np.asarray(row["magnetics_time"], float)
    s = inp["sign"]
    cols = inp["cols"]
    is_e = np.array([c == "magnetics_ECOILA" for c in cols])
    scale = np.where(is_e, A_E, -1.0)
    ip_t = np.asarray(fix_d3d_ip_times(row), float)
    ip_raw = np.asarray(row["magnetics_plasma_current"], float) * KA * s
    I, ip, bt = inp["I"], inp["ip"], inp["bt"]
    ipMA = np.where(np.abs(ip) >= IP_FLOOR, ip / 1e6, np.nan)
    feats, names = [], []

    def at(times, v, tq):
        return window_mean(times, v, tq, COIL_WINDOW_MS)

    # A1 long history: ramp rates of Ip, E-coil and B_T coil current
    ecol = [c for c in cols if c == "magnetics_ECOILA"][0]
    sigs = {"ip": (ip_t, ip_raw / 1e6), "ecoil": (tm, s * np.asarray(row[ecol], float) * KA / 1e5),
            "bt": (tm, np.abs(np.asarray(row["magnetics_bcoil"], float)) * KA / 1e5)}
    for nm, (tt, vv) in sigs.items():
        now = at(tt, vv, te_q)
        for d in HIST_MS:
            feats.append((now - at(tt, vv, te_q - d)) / (d / 1e3)); names.append(f"d{nm}_{d:g}ms")
    # A2 Ip history relative to now
    for d in (300.0, 1000.0):
        feats.append(window_mean(ip_t, ip_raw / 1e6, te_q - d / 2, d / 2) / ipMA); names.append(f"ipmean_{d:g}ms_rel")
    # A3 plasma age and Ip relative to its running maximum
    ok = np.isfinite(ip_raw)
    tt, vv = ip_t[ok], ip_raw[ok] / 1e6
    start = tt[np.argmax(vv > 0.1)] if (vv > 0.1).any() else np.nan
    feats.append(te_q - start); names.append("plasma_age_ms")
    runmax = np.maximum.accumulate(np.where(np.isfinite(vv), vv, -np.inf))
    k = np.clip(np.searchsorted(tt, te_q), 0, len(tt) - 1)
    feats.append((ip / 1e6) / np.where(runmax[k] > 0.05, runmax[k], np.nan)); names.append("ip_over_runmax")
    # probe Green's functions (coils only)
    Gp, pts = probe_greens(row, cols)                              # (C, P, 3)
    Gs = Gp * scale[:, None, None]
    # A4 loop-voltage proxy at the plasma centre: V = -2 pi dpsi_coils/dt
    fil = np.zeros(len(cols))
    for i in range(len(cols)):
        fr, fz = coil_filaments(float(row["coil_R"][i]), float(row["coil_Z"][i]), float(row["coil_width"][i]),
                                float(row["coil_height"][i]), float(row["coil_angle1"][i]), float(row["coil_angle2"][i]), 8)
        fil[i] = filament_psi(fr, fz, np.full_like(fr, R0), np.full_like(fz, Z0)).mean() * scale[i]
    psi_c = np.nan_to_num(I) @ fil
    for d in VLOOP_MS:
        I_prev = s * np.column_stack([at(tm, np.asarray(row[c], float), te_q - d) for c in cols]) * KA
        feats.append(-2 * np.pi * (psi_c - np.nan_to_num(I_prev) @ fil) / (d / 1e3)); names.append(f"vloop_{d:g}ms")
    # A5 synthetic magnetics from the coils, divided by Ip [MA]
    v = np.einsum("tc,cpk->tpk", np.nan_to_num(I), Gs)            # psi, dpsi/dR, dpsi/dZ
    Rp = pts[:, 0][None, :]
    psi_p = v[:, :, 0] / ipMA[:, None]
    bz = v[:, :, 1] / Rp / ipMA[:, None]
    br = -v[:, :, 2] / Rp / ipMA[:, None]
    for j in range(len(pts)):
        tag = f"p{j:02d}"
        feats += [psi_p[:, j], br[:, j], bz[:, j]]; names += [f"{tag}_psi_over_ip", f"{tag}_br_over_ip", f"{tag}_bz_over_ip"]
    # A6 coil up-down balance and totals / Ip
    for n in range(1, 10):
        a, b = cols.index(f"magnetics_F{n}A"), cols.index(f"magnetics_F{n}B")
        feats += [(I[:, a] - I[:, b]) / 1e6 / ipMA, (I[:, a] + I[:, b]) / 1e6 / ipMA]
        names += [f"F{n}_updown_over_ip", f"F{n}_sum_over_ip"]
    # A7 Thomson profile shape
    te_c, ne_c, Rc, Zc, age_c = _thomson_channels(row, "TS_core", "core", te_q)
    te_t, ne_t, Rt, Zt, age_t = _thomson_channels(row, "TS_tangential", "edge", te_q)
    st_c = _profile_stats(te_c, ne_c, Zc) if Zc.size else np.full((len(te_q), 15), np.nan)
    st_t = _profile_stats(te_t, ne_t, Rt) if Rt.size else np.full((len(te_q), 15), np.nan)
    feats += list(st_c.T) + list(st_t.T) + [age_c, age_t]
    names += [f"core_{n}" for n in STAT_NAMES] + [f"tan_{n}" for n in STAT_NAMES] + ["core_age_ms", "tan_age_ms"]
    # A8 dimensionless-style ratios (B_T in tesla at R0)
    btT = MU0 * N_TF * bt / (2 * np.pi * R0)
    btT = np.where(btT > 0.1, btT, np.nan)
    pe_c = st_c[:, 9]; nl_c = st_c[:, 7]
    with np.errstate(all="ignore"):
        feats += [nl_c / ipMA, pe_c / ipMA ** 2, pe_c / btT ** 2, pe_c / (ipMA * btT), ipMA / btT, btT / ipMA, st_c[:, 1] / ipMA]
    names += ["neint_over_ip", "pe_over_ip2", "pe_over_bt2", "pe_over_ip_bt", "ip_over_bt", "bt_over_ip", "temax_over_ip"]

    X = np.column_stack(feats)
    # padded channel arrays for the first-pass flux mapping
    T = len(te_q)
    pos = np.full((NCH, 2), np.nan); tep = np.full((T, NCH), np.nan); nep = np.full((T, NCH), np.nan)
    c0 = min(len(Rc), 56); c1 = min(len(Rt), NCH - c0)
    pos[:c0] = np.column_stack([Rc[:c0], Zc[:c0]]); pos[c0:c0 + c1] = np.column_stack([Rt[:c1], Zt[:c1]])
    tep[:, :c0] = te_c[:, :c0]; nep[:, :c0] = ne_c[:, :c0]
    tep[:, c0:c0 + c1] = te_t[:, :c1]; nep[:, c0:c0 + c1] = ne_t[:, :c1]
    return X, names, {"pos": pos, "te": tep, "ne": nep}
