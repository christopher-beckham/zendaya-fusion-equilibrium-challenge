"""Per-frame q95 ingredients from one 65x65 flux map (our sign convention, as postproc_qb / physics_suite).

q(ψ_N) = R0·B0 / (2π) ∮ dl / (R |∇ψ|) on the ψ_N contour, with ψ_N normalised by the axis and by the boundary that
the scorer's procedure finds inside the selected MAST region (FEC_MAST_MASK). At ψ_N = 0.95 this is exactly
physics_suite.frame_checks "q95_map". Also returns boundary geometry and how close the boundary comes to the edge of
the allowed region (the limiter), which needs only the map and the machine geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NN = Path(__file__).resolve().parents[2]   # my_experiments/nn of the tree this file sits in
sys.path.insert(0, str(NN / "mast_diag"))
sys.path.insert(0, str(NN))
LEVELS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97)


def setup():
    import physics_suite as ps
    from scipy.ndimage import distance_transform_edt
    m = ps.Machine("MAST")
    # distance (in cells) from each in-region cell centre to the nearest out-of-region cell
    dist = distance_transform_edt(m.mask)
    return ps, m, dist


def frame(ps, m, dist, psi, bt):
    from scipy.interpolate import RectBivariateSpline
    from contour import encloses_point, find_contours, grid_to_physical, is_closed
    from lcfs import extract_lcfs_with_sign
    from o_point import find_o_point
    psi = np.asarray(psi, np.float64).reshape(65, 65)
    phi = ps.S * psi
    lc, _ = extract_lcfs_with_sign(psi, m.r, m.z, m.name, m.mask, m.mask.astype(float), axis_sign=ps.S)
    if lc is None:
        return None
    ra, za, iz, ir, _ = find_o_point(phi, m.r, m.z, m.mask)
    sp = RectBivariateSpline(m.z, m.r, phi)
    ax = float(sp(za, ra)[0, 0]); bd = float(np.mean(sp(lc[:, 1], lc[:, 0], grid=False)))
    span = ax - bd
    dr, dz = m.r[1] - m.r[0], m.z[1] - m.z[0]
    # boundary distance to the region edge, in cells (bilinear on the cell-centre distance map)
    from scipy.ndimage import map_coordinates
    rc = np.vstack([(lc[:, 1] - m.z[0]) / dz, (lc[:, 0] - m.r[0]) / dr])
    dmin = float(map_coordinates(dist, rc, order=1).min())
    o = dict(ax=ax, bd=bd, span=span, ra=float(ra), za=float(za),
             r_in=float(lc[:, 0].min()), r_out=float(lc[:, 0].max()), z_top=float(lc[:, 1].max()),
             z_bot=float(lc[:, 1].min()), edge_cells=dmin)
    o["kappa"] = (o["z_top"] - o["z_bot"]) / (o["r_out"] - o["r_in"])
    for lv in LEVELS:
        for cont in find_contours(phi, ax - lv * span):
            if len(cont) < 8 or not is_closed(cont) or not encloses_point(cont, iz, ir):
                continue
            p = grid_to_physical(cont, m.r, m.z)
            midp = 0.5 * (p[1:] + p[:-1]); dl = np.hypot(*np.diff(p, axis=0).T)
            gz = sp(midp[:, 1], midp[:, 0], dx=1, grid=False); gr = sp(midp[:, 1], midp[:, 0], dy=1, grid=False)
            o[f"q{int(round(lv * 100))}"] = float(m.r0 * bt / (2 * np.pi) * np.sum(dl / (midp[:, 0] * np.hypot(gr, gz))))
            if lv == 0.95:
                o["r_in95"] = float(p[:, 0].min())
            break
    # The same with the flux normalised to the separatrix found with no region (the X-point level, ignoring the wall):
    # the definition Zara's v3/v4 labels use on limited frames.
    try:
        lx, _ = extract_lcfs_with_sign(psi, m.r, m.z, m.name, None, None, axis_sign=ps.S)
    except Exception:
        lx = None
    if lx is not None:
        bx = float(np.mean(sp(lx[:, 1], lx[:, 0], grid=False)))
        o["psin_x"] = (ax - bx) / span
        o["r_in_x"] = float(lx[:, 0].min())
        for cont in find_contours(phi, ax - 0.95 * (ax - bx)):
            if len(cont) < 8 or not is_closed(cont) or not encloses_point(cont, iz, ir):
                continue
            p = grid_to_physical(cont, m.r, m.z)
            midp = 0.5 * (p[1:] + p[:-1]); dl = np.hypot(*np.diff(p, axis=0).T)
            gz = sp(midp[:, 1], midp[:, 0], dx=1, grid=False); gr = sp(midp[:, 1], midp[:, 0], dy=1, grid=False)
            o["q95x"] = float(m.r0 * bt / (2 * np.pi) * np.sum(dl / (midp[:, 0] * np.hypot(gr, gz))))
            break
    return o
