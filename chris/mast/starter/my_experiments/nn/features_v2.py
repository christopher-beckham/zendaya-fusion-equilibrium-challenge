"""Input features that mean the same thing on DIII-D and MAST. One code path for both machines.

No coil current enters directly. Each machine's currents become the vacuum flux ψ_ext of its own coils
on its own grid (Green's functions from the shipped coil geometry). The features are dimensionless
numbers read off ψ_ext, Ip, the TF current and Thomson, normalised by the plasma size of that machine.

Plasma geometry is one constant pair per machine, not a label:
    DIII-D R0 1.68 m, a 0.65 m: the envelope eroded by its two-cell dilation, matching the Thomson
    pressure peak (median 1.76 m) and a Shafranov-balance radius from ψ_ext and Ip (1.73 m).
    MAST R0 0.84 m, a 0.56 m: the 5% Te foot of the midplane Thomson chord, 0.28 to 1.38 m.
Z0 is the envelope centroid. κ = 1.8 on both machines enters q_ITER only.

Columns, in order (FEATURES):
    Ip_MA           plasma current
    Bz_over_Bp_a    vacuum Bz at (R0, Z0) over the edge poloidal field μ0 Ip / (2π a)
    Br_over_Bp_a    vacuum Br at (R0, Z0), same scale. Up-down field asymmetry
    hexa_c3         ∂²Bz/∂R² · a² at (R0, Z0), same scale
    q_ITER          cylindrical q with the ITER shaping factor
    dIp_dt_norm     dIp/dt · t50 / Ip_max. t50 is how long |Ip| stays above half its peak
    V_loop_norm     loop voltage of ψ_ext at (R0, Z0) over Lp · Ip / t50
    log10_p_peak    peak Te·ne on the radial midplane chord, eV m⁻³
    betaN_e         electron βN from the peak pressure
    R_ax_th         (R of the pressure peak − R0) / a. A proxy for the outward shift of the axis
    p_rho*, Te_rho*, ne_rho*
                    profiles over their value at the peak, at ρ = (R − R_ax) / (R0 + a − R_ax)
                    on the outboard side of the radial midplane chord
    thomson_missing 1 where the Thomson columns could not be computed for the frame
    rate_missing    1 for a single-frame shot

The radial midplane chord is TS_tangential on DIII-D (10 or 14 channels, ρ up to about 0.55) and
TS_core on MAST (130 channels). The pressure peak is searched within |R − R0| < 0.4 a on both.
NaN marks a value that cannot be computed. The flags say which.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import (
    D3D_TF_TURNS, _MU0, _align_channels, _bilinear, _cell_py, _col, _frame_fields, _group_key,
    _profile_system, _rect_profile, fix_d3d_ip_times, radial_group_key,
)
from shift_report import GRID, MAST_TF_TURNS, coil_pictures, envelope, machine_of

GEOMETRY = {"d3d": (1.68, 0.65), "mast": (0.84, 0.56)}
# DIII-D ECOILA is shipped in kA, one rectangle, with no usable turn count. Least squares of DIII-D train ψ
# on the ECOILA picture, with the F-coil pictures held at 1 and an Ip filament free, gives 134-136 on three
# 200-shot samples. It likely also absorbs the unshipped co-located ECOILB group. MAST's solenoid needs no
# such factor: its 656 turns ship as separate elements and are summed.
D3D_ECOIL_SCALE = 135.0
DECODER_PICTURES = Path(__file__).resolve().parent / "baseline_pictures_ecoil135.npz"
KAPPA = 1.8
IP_FLOOR_KA = 100.0
CORE_HALF = 0.4
RHO_P = (0.1, 0.2, 0.3)
RHO_TN = (0.1, 0.2, 0.3, 0.4)
FLAGS = ["thomson_missing", "rate_missing"]
THOMSON = (
    ["log10_p_peak", "betaN_e", "R_ax_th"]
    + [f"p_rho{x}" for x in RHO_P] + [f"Te_rho{x}" for x in RHO_TN] + [f"ne_rho{x}" for x in RHO_TN]
)
FEATURES = (
    ["Ip_MA", "Bz_over_Bp_a", "Br_over_Bp_a", "hexa_c3", "q_ITER", "dIp_dt_norm", "V_loop_norm"]
    + THOMSON + FLAGS
)
N_FEATURES = len(FEATURES)
VERSION = "v2"
_E = 1.602176634e-19
_GEOM_COLS = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]


def geometry(machine: str) -> dict:
    env = envelope(machine)
    r0, a = GEOMETRY[machine]
    return {**env, "R0": r0, "a": a}


def coil_geometry(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path, columns=_GEOM_COLS)
    names = [str(c) for c in _cell_py(t, "coil_input_column")]
    rect = np.column_stack([np.asarray(_cell_py(t, c), dtype=np.float64) for c in _GEOM_COLS[1:]])
    return names, rect


def build_pictures(path: Path, out: Path) -> Path:
    """Per-column ψ_ext pictures for the machine of `path`, plus the geometry they came from.

    The sign follows baseline_pictures.npz, as in shift_report.build_pictures. DIII-D ECOILA is
    scaled by D3D_ECOIL_SCALE.
    """
    from shift_report import build_pictures as _build
    _build(path, out)
    names, rect = coil_geometry(path)
    with np.load(out) as z:
        pics = {k: z[k] for k in z.files}
    if machine_of(path) == "d3d":
        pics["magnetics_ECOILA"] = pics["magnetics_ECOILA"] * D3D_ECOIL_SCALE
    np.savez(out, **pics, _geom_names=np.array(names), _geom_rect=rect)
    return out


def write_decoder_pictures(out: Path = DECODER_PICTURES) -> Path:
    """baseline_pictures.npz with the ECOILA picture scaled by D3D_ECOIL_SCALE, for train.py --pictures."""
    here = Path(__file__).resolve().parent
    with np.load(here / "baseline_pictures.npz") as z:
        payload = {k: z[k] for k in z.files}
    payload["ecoil"] = (payload["ecoil"].astype(np.float64) * D3D_ECOIL_SCALE).astype(np.float32)
    np.savez(out, **payload)
    return out


def load_pictures(path: Path) -> dict:
    with np.load(path) as z:
        columns = [k for k in z.files if not k.startswith("_")]
        return {
            "columns": columns,
            "green": np.stack([z[c] for c in columns]),
            "geom_names": [str(n) for n in z["_geom_names"]],
            "geom_rect": np.asarray(z["_geom_rect"]),
        }


def _pulse(path: Path, machine: str):
    """Ip at the EFIT times and ±10 ms are not needed here. Returns Ip trace times, values (kA)."""
    cols = ["magnetics_time", "magnetics_plasma_current"]
    if machine == "d3d":
        cols.append("magnetics_plasma_current_times")
    t = pq.read_table(path, columns=cols)
    clock = _col(t, "magnetics_time").astype(np.float64)
    ip = _col(t, "magnetics_plasma_current").astype(np.float64)
    if machine == "d3d":
        ip_t = fix_d3d_ip_times({
            "magnetics_plasma_current_times": _col(t, "magnetics_plasma_current_times"),
            "magnetics_plasma_current": ip, "magnetics_time": clock,
        })
    else:
        ip_t = clock
    n = min(len(ip_t), len(ip))
    ip_t, ip = np.asarray(ip_t[:n], dtype=np.float64), ip[:n]
    ok = np.isfinite(ip_t) & np.isfinite(ip)
    return ip_t[ok], ip[ok]


def _at(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    ok = np.isfinite(times) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(query), np.nan)
    t, v = times[ok], values[ok]
    order = np.argsort(t, kind="mergesort")
    return np.interp(query, t[order], v[order])


def _thomson_rows(radial, te, ne, R0, a):
    """Per Thomson time: log10 p_peak, p_peak (Pa), R_ax, then p, Te, ne over their peak values on ρ."""
    n_prof = len(RHO_P) + 2 * len(RHO_TN)
    rows = np.full((len(te), 3 + n_prof), np.nan)
    for k in range(len(te)):
        T, N = te[k], ne[k]
        ok = np.isfinite(T) & np.isfinite(N) & np.isfinite(radial) & (T > 0) & (N > 0)
        if ok.sum() < 3:
            continue
        rr, T, N = radial[ok], T[ok], N[ok]
        order = np.argsort(rr)
        rr, T, N = rr[order], T[order], N[order]
        p = T * N
        core = np.abs(rr - R0) < CORE_HALF * a
        if not core.any():
            continue
        j = int(np.argmax(np.where(core, p, -np.inf)))
        r_ax = rr[j]
        rho = (rr - r_ax) / (R0 + a - r_ax)
        out = rho >= 0
        if out.sum() < 2:
            continue
        prof = []
        for vals, grid in ((p, RHO_P), (T, RHO_TN), (N, RHO_TN)):
            prof += list(np.interp(grid, rho[out], vals[out] / vals[j], left=np.nan, right=np.nan))
        rows[k] = [np.log10(p[j]), p[j] * _E, r_ax, *prof]
    return rows


def _thomson(path: Path, et: np.ndarray, R0: float, a: float,
             window: tuple[float, float] | None = None) -> np.ndarray:
    """(T, 3 + profile columns) on the EFIT times. Outside the Thomson time range is NaN.

    With `window`, only channels whose (R − R0)/a lies inside it are used, so two machines whose radial chords
    cover different parts of the plasma are measured over the same normalised span.
    """
    cols = ["thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
            "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
            "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne"]
    t = pq.read_table(path, columns=cols)
    names = [str(n) for n in _cell_py(t, "thomson_chord_name")]
    cr = np.asarray(_cell_py(t, "thomson_chord_R"), dtype=np.float64)
    cz = np.asarray(_cell_py(t, "thomson_chord_Z"), dtype=np.float64)
    group = radial_group_key(names, cr, cz)
    chosen = np.array([_group_key(n) == group for n in names])
    system = _profile_system(group)
    te = _rect_profile(_cell_py(t, f"thomson_{system}_Te"))
    ne = _rect_profile(_cell_py(t, f"thomson_{system}_ne"))
    te, ne = _align_channels(te, ne, int(chosen.sum()), path.name)
    times = np.asarray(_cell_py(t, f"thomson_{system}_times"), dtype=np.float64)
    radial = cr[chosen]
    if window is not None:
        keep = ((radial - R0) / a >= window[0]) & ((radial - R0) / a <= window[1])
        radial, te, ne = radial[keep], te[:, keep], ne[:, keep]
    rows = _thomson_rows(radial, te, ne, R0, a)
    valid = np.isfinite(rows).all(axis=1) & np.isfinite(times)
    out = np.full((len(et), rows.shape[1]), np.nan)
    if valid.sum() < 2:
        return out
    tt, rr = times[valid], rows[valid]
    order = np.argsort(tt, kind="mergesort")
    tt, rr = tt[order], rr[order]
    for k in range(rr.shape[1]):
        out[:, k] = np.interp(et, tt, rr[:, k], left=np.nan, right=np.nan)
    return out


def shot_features(path: Path, pictures: dict, return_inputs: bool = False,
                  thomson_window: tuple[float, float] | None = None):
    """(T, N_FEATURES) float32 at the shot's EFIT times. Reads inputs only.

    `thomson_window` restricts the Thomson channels to that span of (R − R0)/a (see _thomson).
    With `return_inputs`, also a dict of the mirrored currents `cur` (T, columns), mirrored Ip `ip` (kA),
    the machine, the mirror sign `s`, the EFIT times `et` (ms) and `t50` (s), for features built on top.
    """
    path = Path(path)
    machine = machine_of(path)
    geo = geometry(machine)
    r, z, R0, Z0, a = geo["r"], geo["z"], geo["R0"], geo["Z0"], geo["a"]
    names, rect = coil_geometry(path)
    if names != pictures["geom_names"] or not np.allclose(rect, pictures["geom_rect"], atol=1e-9):
        raise RuntimeError(f"{path.name}: coil geometry differs from the one the pictures were built from")
    head = pq.read_table(path, columns=["efit_grid_R", "efit_grid_Z", "efit_times"])
    if not (np.allclose(_cell_py(head, "efit_grid_R"), r, atol=1e-5)
            and np.allclose(_cell_py(head, "efit_grid_Z"), z, atol=1e-5)):
        raise RuntimeError(f"{path.name}: grid differs from the {machine} envelope grid")
    et = np.asarray(_cell_py(head, "efit_times"), dtype=np.float64)
    n = len(et)

    columns = pictures["columns"]
    tf_col = "magnetics_bcoil" if machine == "d3d" else "magnetics_tf_current"
    t = pq.read_table(path, columns=[*columns, tf_col, "magnetics_time"])
    clock = _col(t, "magnetics_time").astype(np.float64)
    cur = np.column_stack([_at(clock, _col(t, c).astype(np.float64), et) for c in columns])
    tf = _at(clock, _col(t, tf_col).astype(np.float64), et)
    ip_t, ip_trace = _pulse(path, machine)
    ip = _at(ip_t, ip_trace, et)
    ip_max = float(np.max(np.abs(ip_trace))) if ip_trace.size else np.nan
    flat = np.abs(ip) >= 0.8 * ip_max
    s = float(np.sign(np.median(ip[flat]))) if flat.any() else 1.0
    s = s if s in (-1.0, 1.0) else 1.0
    cur, ip = cur * s, ip * s
    above = ip_t[np.abs(ip_trace) > 0.5 * ip_max]
    t50 = (above.max() - above.min()) * 1e-3 if above.size > 1 else np.nan

    flux = (cur @ pictures["green"]).reshape(-1, GRID, GRID)
    ip_a = np.maximum(np.abs(ip), IP_FLOOR_KA) * 1000.0
    bp_a = _MU0 * ip_a / (2.0 * np.pi * a)
    bz = np.empty(n)
    br = np.empty(n)
    d2 = np.empty(n)
    center = np.empty(n)
    for k in range(n):
        f_br, f_bz, _dbz, f_d2 = _frame_fields(flux[k], r, z)
        bz[k] = _bilinear(f_bz, r, z, R0, Z0)
        br[k] = _bilinear(f_br, r, z, R0, Z0)
        d2[k] = _bilinear(f_d2, r, z, R0, Z0)
        center[k] = _bilinear(flux[k], r, z, R0, Z0)

    out = np.full((n, N_FEATURES), np.nan)
    out[:, 0] = ip / 1000.0
    out[:, 1] = bz / bp_a
    out[:, 2] = br / bp_a
    out[:, 3] = d2 * a ** 2 / bp_a
    turns = D3D_TF_TURNS if machine == "d3d" else MAST_TF_TURNS
    bt = np.abs(_MU0 * turns * tf * 1000.0 / (2.0 * np.pi * R0))
    eps = a / R0
    q_cyl = 2.0 * np.pi * a ** 2 * bt / (_MU0 * R0 * ip_a)
    out[:, 4] = q_cyl * (1.0 + KAPPA ** 2) / 2.0 * (1.17 - 0.65 * eps) / (1.0 - eps ** 2) ** 2
    if n > 1:
        t_s = et * 1e-3
        lp = _MU0 * R0 * (np.log(8.0 * R0 / a) - 1.5)
        out[:, 5] = np.gradient(ip, t_s, edge_order=1) * t50 / ip_max
        out[:, 6] = -2.0 * np.pi * np.gradient(center, t_s, edge_order=1) / (lp * ip_a / t50)

    th = _thomson(path, et, R0, a, thomson_window)
    log_p, pe_pa, r_ax, prof = th[:, 0], th[:, 1], th[:, 2], th[:, 3:]
    i0 = FEATURES.index("log10_p_peak")
    out[:, i0] = log_p
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, i0 + 1] = 100.0 * 2.0 * _MU0 * pe_pa * a / (bt * ip_a / 1e6)
    out[:, i0 + 2] = (r_ax - R0) / a
    out[:, i0 + 3:i0 + 3 + prof.shape[1]] = prof
    th_cols = slice(i0, i0 + len(THOMSON))
    out[:, FEATURES.index("thomson_missing")] = (~np.isfinite(out[:, th_cols]).all(axis=1)).astype(np.float64)
    out[:, FEATURES.index("rate_missing")] = 1.0 if n == 1 else 0.0
    if n == 1:
        out[:, 5:7] = np.nan
    out[~np.isfinite(out)] = np.nan
    if return_inputs:
        return out.astype(np.float32), {"cur": cur, "ip": ip, "machine": machine, "s": s, "et": et, "t50": t50}
    return out.astype(np.float32)
