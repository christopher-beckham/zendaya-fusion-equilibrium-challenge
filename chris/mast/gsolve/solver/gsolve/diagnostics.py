"""Equilibrium numbers from a solution: axis, X-point, plasma edge (LCFS), q95, betaN, li, elongation, triangularity.

Per frame (numpy/scipy, about 50 ms): a cubic spline of psi is evaluated on a 1 cm grid of the wall region.
  axis      maximum of the spline near the solver's axis (Newton steps)
  edge      lowest flux level whose region around the axis does not touch the wall or a coil (bisection);
            for a diverted plasma this is the X-point level, for a limited one the wall contact
  X-points  saddle points of the spline (zero gradient, Hessian determinant < 0); the primary one is the
            one whose flux is closest to the edge level
  surfaces  traced along 256 rays from the axis (assumes each surface is seen from the axis, true for
            ordinary tokamak shapes)
  q95       F/(2 pi) * loop integral of dl / (R |grad psi|) on psi_N = 0.95, with F from the solved FF'
  betaN     100 * 2 mu0 <p> a / (B0 Ip[MA]), <p> volume average, B0 = F_vac / R_geo, a = half the LCFS width
  li        <Bp^2>_V / (mu0 Ip / l_p)^2 with mu0 Ip = loop integral of Bp on the LCFS (EFIT's li, as the
            challenge scorer), l_p the LCFS length
  kappa, delta_upper, delta_lower, volume: from the LCFS polygon (as the challenge scorer)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from .geometry import MU0

_REG: dict = {}


def _region(m, step=0.01):
    if id(m) not in _REG:
        r0, r1, z0, z1 = m.wall_box
        Rf, Zf = np.arange(r0, r1 + 1e-9, step), np.arange(z0, z1 + 1e-9, step)
        RR, ZZ = np.meshgrid(Rf, Zf)
        ok = np.ones(RR.shape, bool)
        if m.coil_margin is not None:
            e = m.elements
            for rc, zc, w, h in zip(e["R"], e["Z"], e["width"], e["height"]):
                ok &= ~((np.abs(RR - rc) < 0.5 * w + m.coil_margin) & (np.abs(ZZ - zc) < 0.5 * h + m.coil_margin))
        _REG[id(m)] = (m, Rf, Zf, RR, ok, ok & ~ndimage.binary_erosion(ok, border_value=0))
    return _REG[id(m)][1:]


def _newton(spl, R, Z, n=8, maxstep=0.01):
    """Newton steps towards a zero of grad psi (axis or X-point)."""
    for _ in range(n):
        gz, gr = spl.ev(Z, R, dx=1), spl.ev(Z, R, dy=1)
        hzz, hrr, hrz = spl.ev(Z, R, dx=2), spl.ev(Z, R, dy=2), spl.ev(Z, R, dx=1, dy=1)
        det = hrr * hzz - hrz ** 2
        if det == 0:
            break
        dR, dZ = -(hzz * gr - hrz * gz) / det, -(hrr * gz - hrz * gr) / det
        s = min(1.0, maxstep / max(np.hypot(dR, dZ), 1e-30))
        R, Z = R + s * dR, Z + s * dZ
    return float(R), float(Z), float(hrr * hzz - hrz ** 2)


def _surface(spl, Ra, Za, phi_ax, dphi, level, box, n=256, ns=400):
    """(n, 2) points of the surface psi_N = level, first crossing along rays from the axis (NaN if open; for the
    LCFS a ray without a crossing ends on the wall, where the surface touches it between 1 cm grid points)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    with np.errstate(divide="ignore"):
        L = np.minimum(np.where(c > 0, (box[1] - Ra) / c, np.where(c < 0, (box[0] - Ra) / c, np.inf)),
                       np.where(s > 0, (box[3] - Za) / s, np.where(s < 0, (box[2] - Za) / s, np.inf)))
    t = np.linspace(0, 1, ns)[None] * L[:, None]
    x = (phi_ax - spl.ev(Za + t * s[:, None], Ra + t * c[:, None])) / dphi
    hit = x >= level
    j = np.where(hit.any(1), hit.argmax(1), 0).clip(1)
    x0, x1 = x[np.arange(n), j - 1], x[np.arange(n), j]
    tt = t[np.arange(n), j - 1] + (level - x0) / np.where(x1 > x0, x1 - x0, 1) * (t[np.arange(n), j] - t[np.arange(n), j - 1])
    tt = np.where(hit.any(1), tt, L if level >= 1 - 1e-3 else np.nan)   # LCFS: no crossing = wall contact
    return np.stack([Ra + tt * c, Za + tt * s], 1)


def _loop(C):
    D = np.roll(C, -1, 0) - C
    return 0.5 * (C + np.roll(C, -1, 0)), np.hypot(D[:, 0], D[:, 1])


def analyse(sol, k: int, i_tf_kA: float) -> dict:
    """Numbers of frame k of a Solution (forward or correction mode). i_tf_kA: TF coil current (kA)."""
    m = sol.machine
    nan = float("nan")
    out = dict(R_ax=nan, Z_ax=nan, psi_ax=nan, psi_b=nan, xpt_R=nan, xpt_Z=nan, diverted=False, R_geo=nan, a=nan,
               kappa=nan, delta_upper=nan, delta_lower=nan, volume=nan, q95=nan, betaN=nan, li=nan, lcfs=None)
    psi = sol.psi[k].double().numpy()
    if not np.isfinite(psi).all():
        return out
    spl = RectBivariateSpline(m.Z, m.R, psi)
    Rf, Zf, RR, ok, edge = _region(m)
    phi = spl(Zf, Rf)
    # axis
    Ra, Za, _ = _newton(spl, float(sol.info["R_ax"][k]), float(sol.info["Z_ax"][k]))
    iz, ir = np.argmin(np.abs(Zf - Za)), np.argmin(np.abs(Rf - Ra))
    if not ok[iz, ir]:
        return out
    phi_ax = float(spl.ev(Za, Ra))

    def region(level):
        lab, _ = ndimage.label((phi > level) & ok)
        comp = lab == lab[iz, ir]
        return None if lab[iz, ir] == 0 or (comp & edge).any() else comp
    lo, hi = float(phi[ok].min()), phi_ax
    for _ in range(40):                                  # edge level by bisection
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if region(mid) is not None else (mid, hi)
    phi_b, dphi = hi, phi_ax - hi
    comp = region(phi_b + 1e-6 * dphi)
    if not dphi > 0 or comp is None or comp.sum() < 20:
        return out
    out.update(R_ax=Ra, Z_ax=Za, psi_ax=phi_ax, psi_b=phi_b)
    # X-points: local minima of |grad psi| with a negative Hessian determinant, refined by Newton steps
    g2 = spl(Zf, Rf, dx=1) ** 2 + spl(Zf, Rf, dy=1) ** 2
    cand = (g2 == ndimage.minimum_filter(g2, 5)) & ndimage.binary_erosion(ok, iterations=2)
    best = None
    for jz, jr in zip(*np.nonzero(cand)):
        R, Z, det = _newton(spl, Rf[jr], Zf[jz])
        if det < 0 and m.wall_box[0] <= R <= m.wall_box[1] and m.wall_box[2] <= Z <= m.wall_box[3]:
            d = abs(float(spl.ev(Z, R)) - phi_b)
            if best is None or d < best[0]:
                best = (d, R, Z)
    if best is not None:
        psi_x = float(spl.ev(best[2], best[1]))
        if 0 < psi_x - phi_b < 1e-2 * dphi:               # the 1 cm grid missed the narrow gap at the X-point
            phi_b, dphi = psi_x, phi_ax - psi_x
            out["psi_b"] = phi_b
        out.update(xpt_R=best[1], xpt_Z=best[2], diverted=bool(abs(psi_x - phi_b) < 2e-3 * dphi))
    # plasma edge and shape
    C = _surface(spl, Ra, Za, phi_ax, dphi, 1 - 1e-6, m.wall_box)
    if not np.isfinite(C).all():
        return out
    Rmax, Rmin, Zmax, Zmin = C[:, 0].max(), C[:, 0].min(), C[:, 1].max(), C[:, 1].min()
    a, Rg = 0.5 * (Rmax - Rmin), 0.5 * (Rmax + Rmin)
    R2, Z2 = np.append(C[:, 0], C[0, 0]), np.append(C[:, 1], C[0, 1])
    out.update(lcfs=C, a=a, R_geo=Rg, kappa=(Zmax - Zmin) / (2 * a), delta_upper=(Rg - C[np.argmax(C[:, 1]), 0]) / a,
               delta_lower=(Rg - C[np.argmin(C[:, 1]), 0]) / a,
               volume=abs(np.pi / 3 * np.sum((R2[:-1] + R2[1:]) * (R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]))))
    # li
    gp = lambda P: np.hypot(spl.ev(P[:, 1], P[:, 0], dx=1), spl.ev(P[:, 1], P[:, 0], dy=1)) / P[:, 0]  # noqa: E731
    Rc = RR[comp]
    Bp2 = (spl.ev(Zf[np.nonzero(comp)[0]], Rc, dx=1) ** 2 + spl.ev(Zf[np.nonzero(comp)[0]], Rc, dy=1) ** 2) / Rc ** 2
    mid, dl = _loop(C)
    mu0ip = np.sum(gp(mid) * dl)
    out["li"] = float(np.sum(Bp2 * Rc) / np.sum(Rc) / (mu0ip / dl.sum()) ** 2)
    # q95 and betaN from the solved profiles
    fvac = float(m.fvac(i_tf_kA))
    if fvac > 0:
        xs = np.linspace(0, 1, 201)
        p, ffp = sol.profiles(xs, k)
        fr = float(sol.info["psi_ax"][k] - sol.info["psi_b"][k])
        tail = np.concatenate([np.cumsum(((ffp[1:] + ffp[:-1]) / 2 * np.diff(xs))[::-1])[::-1], [0.0]])
        F95 = np.sqrt(max(fvac ** 2 + 2 * fr * np.interp(0.95, xs, tail), 1e-12))
        C95 = _surface(spl, Ra, Za, phi_ax, dphi, 0.95, m.wall_box)
        if np.isfinite(C95).all():
            mid, dl = _loop(C95)
            out["q95"] = float(F95 / (2 * np.pi) * np.sum(dl / (mid[:, 0] ** 2 * gp(mid))))
        xN = np.clip((phi_ax - phi[comp]) / dphi, 0, 1)
        p_avg = np.sum(np.interp(xN, xs, p) * Rc) / np.sum(Rc)
        out["betaN"] = float(100 * 2 * MU0 * p_avg * a / (fvac / Rg * float(sol.ip[k]) / 1e6))
    return out


def analyse_all(sol, i_tf_kA, keys=("R_ax", "Z_ax", "xpt_R", "xpt_Z", "kappa", "delta_upper", "delta_lower",
                                     "volume", "q95", "betaN", "li")) -> dict:
    """Arrays of the numbers for every frame; NaN where the solver status is not 0 or the analysis fails."""
    i_tf = np.broadcast_to(np.asarray(i_tf_kA, np.float64), (len(sol.ip),))
    res = {k: np.full(len(sol.ip), np.nan) for k in keys}
    for j in range(len(sol.ip)):
        if int(sol.info["status"][j]) == 0:
            o = analyse(sol, j, i_tf[j])
            for k in keys:
                res[k][j] = o[k]
    return res
