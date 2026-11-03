"""Optional physics-derived inputs for C1Model, appended to the feature matrix of shot_inputs.

Each option is switched on by name (C1Model(physics=("field", "shaf", "eddy"))):

field   External (coil) field near the plasma centre, from the same Green's functions as the
        physics stage, divided by the plasma current. The coil vacuum flux is fitted with a cubic
        in (R - R0, Z - Z0) over a disk around the plasma filament; the coefficients give the
        vertical and radial field (radial / vertical force balance), their gradients (decay
        index, quadrupole -> elongation) and the cubic terms (hexapole -> triangularity). A
        linear model can form any fixed combination of coil currents itself; what it cannot
        form is the ratio to Ip, which is what sets the equilibrium.
shaf    Shafranov radial force balance solved for beta_p + li/2:
            Lambda = -4 pi R0 B_Z,phys / (mu0 Ip) - ln(8 R0 / a) + 3/2
        with B_Z from `field` and a fixed minor radius a = 0.6 m. The shipped DIII-D psi has
        the opposite sign to the physical flux, so B_Z,phys = -B_Z from the shipped convention.
        (A minor radius from the tangential Thomson density edge was tried and dropped: the
        tangential chords stop short of the outer edge, so it pinned at its lower bound.)
        With a fixed, Lambda is a shifted copy of B_Z / B_Ip; on a 40-shot sample it correlated
        -0.37 with EFIT li, not positively as force balance predicts, so treat it as a test.
eddy    Vessel eddy currents follow the flux change with an L/R lag. Each coil current and Ip
        is passed through a first-order high-pass filter (x - lowpass_tau(x)) on its native
        time base, for tau = 5 and 10 ms, then averaged over +-1 ms at each EFIT time.

Everything uses allowed inputs only (coil currents, Ip, coil geometry, Thomson), in the shot's
Ip > 0 frame like the rest of shot_inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from features import COIL_WINDOW_MS, KA, window_mean
from greens import MU0

OPTIONS = ("field", "shaf", "eddy")
R0, Z0 = 1.70, 0.0             # plasma filament of the physics stage
FIT_RADIUS = 0.35              # m, disk over which the coil flux is fitted with a cubic
A_E_NOMINAL = -152.8           # E-coil scale a_E fitted by the physics stage on all 7,041 shots
A_NOMINAL = 0.60               # m, nominal DIII-D minor radius
IP_FLOOR = 5e4                 # A; below this the field / Ip ratios are left missing
EDDY_TAU_MS = (5.0, 10.0)
# cubic basis in x = (R - R0) / FIT_RADIUS, z = (Z - Z0) / FIT_RADIUS
POWERS = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


def _multipole_matrix(G: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray) -> np.ndarray:
    """(C, 1 + len(POWERS)) cubic-fit coefficients of each coil's flux image around (R0, Z0)."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    x, z = (RR - R0) / FIT_RADIUS, (ZZ - Z0) / FIT_RADIUS
    disk = x ** 2 + z ** 2 <= 1.0
    P = np.column_stack([np.ones(disk.sum())] + [x[disk] ** i * z[disk] ** j for i, j in POWERS])
    return np.linalg.lstsq(P, G[:, disk].T, rcond=None)[0].T


def field_terms(inp: dict, G: np.ndarray, grid_R, grid_Z) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Field / Ip features, their names, and B_Z at (R0, Z0) in tesla (for `shaf`)."""
    M = _multipole_matrix(G, grid_R, grid_Z)
    is_e = np.array([c == "magnetics_ECOILA" for c in inp["cols"]])
    # same combination as the physics stage: -sum_F I_F G_F + a_E I_E G_E
    scale = np.where(is_e, A_E_NOMINAL, -1.0)
    c = (np.nan_to_num(inp["I"]) * scale) @ M                    # (T, 1 + 9) flux coefficients
    cx, cz, cxx = c[:, 1] / FIT_RADIUS, c[:, 2] / FIT_RADIUS, c[:, 3] / FIT_RADIUS ** 2
    bz = cx / R0                                                 # B_Z = (1/R) dpsi/dR
    br = -cz / R0                                                # B_R = -(1/R) dpsi/dZ
    dbz_dr = 2 * cxx / R0 - cx / R0 ** 2
    ip = inp["ip"]
    ok = np.abs(ip) >= IP_FLOOR
    b_ip = np.where(ok, MU0 * ip / (4 * np.pi * R0), np.nan)    # field scale of the plasma current
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.clip(-R0 * dbz_dr / bz, -10, 10)
        # each coefficient in field units (per FIT_RADIUS^order), over the Ip field scale
        # first-order terms are B_Z and B_R themselves, so start at the second order
        norm = np.column_stack([c[:, 1 + k] / (R0 * FIT_RADIUS) / b_ip for k in range(2, len(POWERS))])
        X = np.column_stack([bz / b_ip, br / b_ip, norm, decay])
    names = ["bz_over_bip", "br_over_bip"] + [f"psi_c{i}{j}_over_bip" for i, j in POWERS[2:]] + \
            ["decay_index"]
    return X, names, bz


def shafranov_terms(inp: dict, bz: np.ndarray) -> tuple[np.ndarray, list[str]]:
    ip = inp["ip"]
    with np.errstate(divide="ignore", invalid="ignore"):
        # bz is in the shipped psi convention; the physical vertical field is -bz
        lam = np.where(np.abs(ip) >= IP_FLOOR, 4 * np.pi * R0 * bz / (MU0 * ip), np.nan) \
            - np.log(8 * R0 / A_NOMINAL) + 1.5
    return lam[:, None], ["shafranov_lambda"]


def _highpass(t: np.ndarray, v: np.ndarray, tq: np.ndarray, tau_ms: float) -> np.ndarray:
    """x - lowpass_tau(x) on the native samples, averaged over +-1 ms at the query times."""
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < 3:
        return np.full(tq.shape, np.nan)
    lo, hi = np.searchsorted(t, [tq.min() - 12 * tau_ms - 25.0, tq.max() + 2 * COIL_WINDOW_MS])
    t, v = t[lo:hi], v[lo:hi]
    if t.size < 3:
        return np.full(tq.shape, np.nan)
    dt = float(np.median(np.diff(t)))
    alpha = dt / (tau_ms + dt)
    b, a = [alpha], [1.0, alpha - 1.0]
    lp, _ = lfilter(b, a, v, zi=lfilter_zi(b, a) * v[0])
    return window_mean(t, v - lp, tq, COIL_WINDOW_MS)


def eddy_terms(row: dict, inp: dict) -> tuple[np.ndarray, list[str]]:
    from data_fixes import fix_d3d_ip_times

    te = np.asarray(row["efit_times"], dtype=np.float64)
    tm = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_t = np.asarray(fix_d3d_ip_times(row), dtype=np.float64)
    s = inp["sign"]
    feats, names = [], []
    for tau in EDDY_TAU_MS:
        for c in inp["cols"]:
            feats.append(s * _highpass(tm, np.asarray(row[c], dtype=np.float64), te, tau) * KA / 1e5)
            names.append(f"hp{tau:g}ms_{c}")
        feats.append(s * _highpass(ip_t, np.asarray(row["magnetics_plasma_current"], dtype=np.float64),
                                   te, tau) * KA / 1e6)
        names.append(f"hp{tau:g}ms_ip")
    return np.column_stack(feats), names


def add_physics(row: dict, inp: dict, options, G: np.ndarray, grid_R, grid_Z) -> dict:
    """Return inp with the chosen physics features appended to X / names."""
    bad = set(options) - set(OPTIONS)
    if bad:
        raise ValueError(f"unknown physics options {sorted(bad)}; choose from {OPTIONS}")
    blocks, names = [inp["X"]], list(inp["names"])
    bz = None
    if "field" in options or "shaf" in options:
        Xf, nf, bz = field_terms(inp, G, grid_R, grid_Z)
        if "field" in options:
            blocks.append(Xf)
            names += nf
    if "shaf" in options:
        Xs, ns = shafranov_terms(inp, bz)
        blocks.append(Xs)
        names += ns
    if "eddy" in options:
        Xe, ne = eddy_terms(row, inp)
        blocks.append(Xe)
        names += ne
    return {**inp, "X": np.column_stack(blocks), "names": np.asarray(names)}
