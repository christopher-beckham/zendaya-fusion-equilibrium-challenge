"""Boundary quality check: stored LCFS polygon vs the challenge scorer's extraction from the flux map.

The scorer extracts the boundary from psi itself, so it is an independent reference for the stored
efit_lcfs_* polygon (FreeGSNKE's separatrix routine occasionally drops part of the boundary and
joins the rest with a chord). Masks are geometry-only: the limiter outline for MAST, SPARC and
ITER, the scorer's DIII-D envelope for DIII-D.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

STARTER = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter"
sys.path.insert(0, os.path.join(STARTER, "fusion_scoring"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCORER_MACHINE = {"mast": "MAST", "sparc": "MAST", "iter": "MAST", "d3d": "DIII-D"}
THRESHOLD = 0.05  # boundary disagreement / minor radius above which a frame is rejected


def limiter_mask(ds, R, Z):
    if ds == "d3d":
        return np.load(os.path.join(STARTER, "fusion_scoring/masks/d3d_envelope.npz"))["mask_coarse"].astype(bool)
    if ds == "mast":
        import mast_machine as mm
        poly = mm.limiter_polygon()
    else:
        import pickle
        import gen_machine as gmach
        lim = pickle.load(open(gmach.cfg_paths(ds.upper())["limiter_path"], "rb"))
        poly = np.array([[q["R"], q["Z"]] for q in lim])
    RR, ZZ = np.meshgrid(R, Z)
    return Path(poly).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)


def densify(r, z, step=0.003):
    pts = []
    for k in range(len(r) - 1):
        n = max(2, int(np.hypot(r[k + 1] - r[k], z[k + 1] - z[k]) / step))
        t = np.linspace(0, 1, n, endpoint=False)
        pts.append(np.c_[r[k] + t * (r[k + 1] - r[k]), z[k] + t * (z[k + 1] - z[k])])
    return np.vstack(pts)


def boundary_disagreement(ds, psi_stored_convention, R, Z, lr, lz, minor_radius, mask):
    """max distance between stored polygon and scorer contour, divided by minor radius (nan if
    the scorer finds no boundary)."""
    from lcfs import extract_lcfs
    C = extract_lcfs(psi_stored_convention, R, Z, SCORER_MACHINE[ds], mask, mask.astype(float), n_points=512)
    if C is None:
        return np.nan
    a = densify(np.asarray(lr), np.asarray(lz))
    d = max(cKDTree(C).query(a)[0].max(), cKDTree(a).query(C)[0].max())
    return d / minor_radius
