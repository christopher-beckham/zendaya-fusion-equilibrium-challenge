"""Per-frame βN ingredients from one flux map plus that frame's inputs (Thomson Te, ne, chord R, Z; Ip; F = R0·B0).

Everything here is label-free: it reads only the map (true or predicted) and the frame's inputs.

frame_feats(m, psi, te, ne, tr, tz, ip_ka, bt0) -> dict or None
  pe_avg        volume average (R-weighted) of the Thomson electron pressure mapped onto the map's flux surfaces
                (exactly physics_suite.frame_checks' betaN_th ingredient), Pa
  a_map, rgeo   half-width and centre of the map's boundary (the label's a and R_geo definitions), m
  p_gs_avg      volume average of the TOTAL pressure recovered from the map itself: the map's current density
                J = Δ*ψ / (μ0 R) is split, pixel by pixel inside the boundary, into a part ∝ R (pressure-driven) and
                a part ∝ 1/R (the rest), each a quadratic in ψ_N; integrating the pressure-driven part across flux
                gives p(ψ_N). Pa (sign fixed once, see GS_SIGN)
  ip_map        ∫ J dA over the plasma from the map (check of units / sign), kA
  plus shape and Thomson summary features for the regression variant.
"""
import sys
from pathlib import Path

import numpy as np

NN = Path(__file__).resolve().parents[2]   # my_experiments/nn of the tree this file sits in
sys.path.insert(0, str(NN / "mast_diag"))
import physics_suite as ps  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

MU0 = 4e-7 * np.pi
DEG = 2            # polynomial degree in ψ_N of each current part
GS_SIGN = 1.0      # fixed convention, set from training frames (see analyse.py)


def _gs_pressure(m, psi, inside, psin, span):
    """Total-pressure profile from the map: J = R P(ψN) + G(ψN)/R, P = dp/dψ. Returns (p on inside pixels, ip_map)."""
    dr, dz = m.r[1] - m.r[0], m.z[1] - m.z[0]
    R = m.RR
    d_r = np.gradient(psi, dr, axis=1)
    lap = R * np.gradient(d_r / R, dr, axis=1) + np.gradient(np.gradient(psi, dz, axis=0), dz, axis=0)
    J = lap / (MU0 * R)                                  # A/m², sign of our ψ convention
    ip_map = float(np.sum(J[inside]) * dr * dz) / 1e3
    # interior pixels only (all four neighbours inside), so the stencil does not straddle the boundary
    core = inside.copy()
    core[1:-1, 1:-1] &= inside[:-2, 1:-1] & inside[2:, 1:-1] & inside[1:-1, :-2] & inside[1:-1, 2:]
    core[0, :] = core[-1, :] = core[:, 0] = core[:, -1] = False
    core &= psin < 0.97
    if core.sum() < 30:
        return None, ip_map
    x, r, j = psin[core], R[core], J[core]
    B = np.column_stack([r * x ** k for k in range(DEG + 1)] + [x ** k / r for k in range(DEG + 1)])
    c, *_ = np.linalg.lstsq(B, j, rcond=None)
    a_p = c[:DEG + 1]                                    # P(x) = Σ a_k x^k  (per unit ψ)
    # p(x) = |span| ∫_x^1 P(s) ds  (ψ grows from axis to boundary by span in our normalisation)
    def pint(xx):
        return sum(a_p[k] * (1 - xx ** (k + 1)) / (k + 1) for k in range(DEG + 1))
    p = GS_SIGN * np.sign(ip_map) * abs(span) * pint(np.clip(psin[inside], 0, 1))
    return p, ip_map


def frame_feats(m, psi, te, ne, tr, tz, ip_ka, bt0):
    psi = np.asarray(psi, np.float64).reshape(65, 65)
    S = ps.S
    phi = S * psi
    lc, _ = ps.extract_lcfs_with_sign(psi, m.r, m.z, m.name, m.mask, m.mask.astype(float), axis_sign=S)
    if lc is None:
        return None
    ra, za, iz, ir, _ = ps.find_o_point(phi, m.r, m.z, m.mask)
    sp = RectBivariateSpline(m.z, m.r, phi)
    ax = float(sp(za, ra)[0, 0]); bd = float(np.mean(sp(lc[:, 1], lc[:, 0], grid=False)))
    span = ax - bd
    inside = MPath(lc).contains_points(m.pts).reshape(65, 65)
    if inside.sum() <= 10:
        return None
    psin_grid = (ax - phi) / span
    rmin, rmax = lc[:, 0].min(), lc[:, 0].max(); zmin, zmax = lc[:, 1].min(), lc[:, 1].max()
    out = dict(a_map=0.5 * (rmax - rmin), rgeo=0.5 * (rmax + rmin), kappa_map=(zmax - zmin) / (rmax - rmin),
               r_axis_map=float(ra), vol_map=float(np.sum(2 * np.pi * m.RR[inside]) * m.dA),
               ip=float(ip_ka), bt0=float(bt0), span=float(span))
    # Thomson electron pressure, as frame_checks
    te = te.astype(np.float64); ne = ne.astype(np.float64); cr = tr.astype(np.float64); cz = tz.astype(np.float64)
    ok = np.isfinite(te) & (cr >= m.r[0]) & (cr <= m.r[-1]) & (cz >= m.z[0]) & (cz <= m.z[-1])
    if ok.sum() >= 4:
        psin = (ax - sp(cz[ok], cr[ok], grid=False)) / span
        pe = te[ok] * ne[ok] * 1e19 * 1.602e-19
        good = np.isfinite(pe) & (psin >= 0) & (psin <= 1.05)
        if good.sum() >= 4:
            o = np.argsort(psin[good]); pn, pv = psin[good][o], pe[good][o]
            pix = np.clip(psin_grid[inside], 0, 1)
            pp = np.interp(pix, pn, pv, left=pv[0], right=pv[-1])
            out["pe_avg"] = float(np.sum(pp * m.RR[inside]) / np.sum(m.RR[inside]))
            out["pe_core"] = float(np.median(pv[pn < 0.3])) if (pn < 0.3).any() else np.nan
            out["te_max"] = float(np.nanmax(te[ok])); out["ne_avg"] = float(np.nanmean(ne[ok][psin <= 1]))
            out["n_ts"] = int(good.sum())
    p, ip_map = _gs_pressure(m, psi, inside, psin_grid, span)
    out["ip_map"] = ip_map
    if p is not None:
        out["p_gs_avg"] = float(np.sum(p * m.RR[inside]) / np.sum(m.RR[inside]))
    return out


def betan(p_avg, a, b0, ip_ka):
    """βN = 100 β_t a B0 / Ip[MA], β_t = 2 μ0 <p> / B0² (the generator's and EFIT's definition)."""
    return 100 * (2 * MU0 * p_avg / b0 ** 2) * a * b0 / (np.abs(ip_ka) / 1000)
