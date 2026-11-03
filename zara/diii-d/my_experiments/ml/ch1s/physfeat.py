"""Physics functionals of one DIII-D flux map (65x65, shipped convention: axis = minimum of psi, Wb/rad) for q95 / betaN.

    feats = frame_feats(psi, ip, bt)      # dict of floats, NaN where undefined

  q-integrals   q_L = F0 / (2 pi) * loop-integral dl / (R |grad psi|) on the psi_N = L surface (L = 0.90, 0.95, 0.98), with the
                vacuum F0 = mu0 N_TF I_TF / (2 pi) (N_TF = 144, bt = TF coil current [A]); psi from a bicubic spline, surface points
                from marching squares refined by two Newton steps on the spline
  geometry      LCFS (scorer extraction): a, Rgeo, kappa, tri_top, tri_bot, area, volume, perimeter lp; Ampere G = loop-integral Bp dl
                (= mu0 Ip of the map), li (scorer form)
  GS fit        -Delta* psi = mu0 R^2 p'(x) + F F'(x) on grid points inside the LCFS (x = psi_N, cubic polynomials, least squares);
                p(x) from p' (p(1) = 0) -> <p>_V, beta_t (B0 = F0 / 1.6955 m), beta_p (Bpa = G / lp), betaN = 100 beta_t a B0 / Ip[MA]
  Uckan         q95 shape-factor estimate 5 a^2 B0 / (R Ip[MA]) (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)) / 2 * (1.17 - 0.65 eps) / (1 - eps^2)^2
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from contour import find_contours, encloses_point, grid_to_physical  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
from derive import internal_inductance  # noqa: E402
from o_point import find_o_point  # noqa: E402

_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GR, GZ = _M["grid_R"].astype(np.float64), _M["grid_Z"].astype(np.float64)
MASK = _M["mask_coarse"].astype(bool)
MASKF = MASK.astype(np.float64)
DR, DZ = float(GR[1] - GR[0]), float(GZ[1] - GZ[0])
RR, ZZ = np.meshgrid(GR, GZ)
MU0 = 4e-7 * math.pi
R0_EFIT = 1.6955
LEVELS = (0.90, 0.95, 0.98)
NAMES = (["q90", "q95", "q98", "a", "Rgeo", "kappa", "tri_top", "tri_bot", "area", "volume", "lp", "G_over_mu0ip", "li",
          "dpsi", "beta_t", "beta_p", "betaN_gs", "W_gs", "pp0", "ffp0", "gs_r2", "q95_uckan", "qcyl", "Rax", "Zax"])


def _poly_area(C):
    R, Z = C[:, 0], C[:, 1]
    return 0.5 * abs(np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z))


def _contour_at(phi, level, iz, ir):
    best = None
    for c in find_contours(phi, level):
        if len(c) < 8 or not np.allclose(c[0], c[-1], atol=1e-6):
            continue
        if not encloses_point(c, iz, ir):
            continue
        if best is None or len(c) > len(best):
            best = c
    return best


def frame_feats(psi, ip, bt, contour=None):
    out = dict.fromkeys(NAMES, np.nan)
    psi = np.asarray(psi, np.float64)
    if not np.isfinite(psi).all() or not np.isfinite(ip) or not np.isfinite(bt):
        return out
    phi = -psi                                            # axis = maximum of phi
    try:
        if contour is None:
            C, sgn = extract_lcfs_with_sign(psi, GR, GZ, "DIII-D", MASK, MASKF, n_iter=22, axis_sign=-1.0)
        else:
            C = contour
        if C is None or len(C) < 8:
            return out
        Ra, Za, iz, ir, _ = find_o_point(phi, GR, GZ, MASK)
        out["Rax"], out["Zax"] = Ra, Za
        sp = RectBivariateSpline(GZ, GR, psi, kx=3, ky=3)
        psi_ax = float(sp.ev(Za, Ra))
        psi_b = float(np.mean(sp.ev(C[:, 1], C[:, 0])))
        dpsi = psi_b - psi_ax                             # > 0 in the shipped convention
        out["dpsi"] = dpsi
        F0 = MU0 * 144 * float(bt) / (2 * math.pi)        # R B_T (vacuum), T m
        B0 = F0 / R0_EFIT
        ipa = abs(float(ip))
        # ---- LCFS geometry
        R, Z = C[:, 0], C[:, 1]
        Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
        a = 0.5 * (Rmax - Rmin); Rgeo = 0.5 * (Rmax + Rmin)
        kappa = (Zmax - Zmin) / (2 * a)
        tri_t = (Rgeo - R[np.argmax(Z)]) / a; tri_b = (Rgeo - R[np.argmin(Z)]) / a
        area = _poly_area(C)
        R2_, Z2_ = np.append(R, R[0]), np.append(Z, Z[0])
        vol = 2 * math.pi * abs(np.sum((R2_[:-1] + R2_[1:]) * (R2_[:-1] * Z2_[1:] - R2_[1:] * Z2_[:-1])) / 6.0)
        seg = np.hypot(np.diff(R2_), np.diff(Z2_)); lp = seg.sum()
        Rm, Zm = 0.5 * (R2_[:-1] + R2_[1:]), 0.5 * (Z2_[:-1] + Z2_[1:])
        gb = np.hypot(sp.ev(Zm, Rm, dx=1), sp.ev(Zm, Rm, dy=1))
        G = float(np.sum(gb / Rm * seg))
        out.update(a=a, Rgeo=Rgeo, kappa=kappa, tri_top=tri_t, tri_bot=tri_b, area=area, volume=vol, lp=lp,
                   G_over_mu0ip=G / (MU0 * ipa) if ipa > 0 else np.nan,
                   li=internal_inductance(psi, GR, GZ, "DIII-D", C))
        if ipa > 0:
            eps = a / Rgeo
            d = 0.5 * (tri_t + tri_b)
            out["qcyl"] = 5 * a * a * B0 * kappa / (Rgeo * ipa / 1e6)
            out["q95_uckan"] = (5 * a * a * B0 / (Rgeo * ipa / 1e6) * 0.5 * (1 + kappa ** 2 * (1 + 2 * d * d - 1.2 * d ** 3))
                                * (1.17 - 0.65 * eps) / (1 - eps * eps) ** 2)
        # ---- q on psi_N surfaces
        if dpsi > 0:
            for L, nm in zip(LEVELS, ("q90", "q95", "q98")):
                c = _contour_at(phi, -(psi_ax + L * dpsi), iz, ir)
                if c is None:
                    continue
                P = grid_to_physical(c, GR, GZ)
                lev = psi_ax + L * dpsi
                for _ in range(2):                        # Newton refinement onto the spline surface
                    v = sp.ev(P[:, 1], P[:, 0]) - lev
                    gR, gZ = sp.ev(P[:, 1], P[:, 0], dy=1), sp.ev(P[:, 1], P[:, 0], dx=1)
                    g2 = np.maximum(gR * gR + gZ * gZ, 1e-12)
                    P = P - np.column_stack([v * gR / g2, v * gZ / g2])
                P2 = np.vstack([P, P[:1]])
                sg = np.hypot(np.diff(P2[:, 0]), np.diff(P2[:, 1]))
                Pm = 0.5 * (P2[:-1] + P2[1:])
                g = np.hypot(sp.ev(Pm[:, 1], Pm[:, 0], dx=1), sp.ev(Pm[:, 1], Pm[:, 0], dy=1))
                out[nm] = F0 / (2 * math.pi) * float(np.sum(sg / (Pm[:, 0] * np.maximum(g, 1e-9))))
        # ---- Grad-Shafranov fit of p' and FF' inside the LCFS
        if dpsi > 0 and ipa > 0:
            Lap = (np.gradient(np.gradient(psi, DR, axis=1), DR, axis=1) - np.gradient(psi, DR, axis=1) / RR
                   + np.gradient(np.gradient(psi, DZ, axis=0), DZ, axis=0))
            Cc = np.vstack([C, C[:1]])
            from derive import _points_in_polygon
            ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
            ins[[0, -1], :] = False; ins[:, [0, -1]] = False
            x = (psi[ins] - psi_ax) / dpsi
            ok = (x >= 0) & (x <= 1.0)
            if ok.sum() >= 12:
                x = x[ok]; Rp = RR[ins][ok]; y = -Lap[ins][ok]
                # sign of the current: y has the sign of mu0 R J; normalise so that the plasma current is positive
                A = np.column_stack([MU0 * Rp ** 2 * x ** n for n in range(4)] + [x ** n for n in range(4)])
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                res = y - A @ coef
                out["gs_r2"] = 1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
                cp, cf = coef[:4], coef[4:]
                out["pp0"], out["ffp0"] = float(cp.sum()), float(cf[0])
                # p(x) = dpsi * int_1^x p'(s) ds  (dpsi > 0, p'(x) = dp/dpsi)
                xs = (psi - psi_ax) / dpsi
                Ip_int = lambda xx: sum(cp[n] * (xx ** (n + 1) - 1.0) / (n + 1) for n in range(4))   # int_1^x
                pgrid = dpsi * Ip_int(np.clip(xs, 0, 1))
                dV = 2 * math.pi * RR * DR * DZ
                insv = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
                V = float(np.sum(dV[insv]))
                pav = float(np.sum(pgrid[insv] * dV[insv]) / V)
                out["W_gs"] = 1.5 * pav * V / 1e6
                bt_ = 2 * MU0 * pav / B0 ** 2
                bpa = G / lp
                out["beta_t"], out["beta_p"] = bt_, 2 * MU0 * pav / bpa ** 2
                out["betaN_gs"] = 100 * bt_ * a * B0 / (ipa / 1e6)
    except Exception:
        return out
    return out


def shot_feats(args):
    """(psi (T,65,65), ip (T,), bt (T,)) -> (T, len(NAMES)) float64."""
    psi, ip, bt = args
    rows = []
    for k in range(len(ip)):
        f = frame_feats(psi[k], ip[k], bt[k])
        rows.append([f[n] for n in NAMES])
    return np.array(rows, np.float64)
