"""features_v3 plus the coil picture sampled on a grid in plasma-normalised coordinates.

The v3 harmonics sample ψ_ext on circles of radius up to a, which miss the top and bottom of an elongated
plasma and the X-point regions. v4 adds ψ_ext − ψ_ext(R0, Z0) at the points
(R0 + u·a, Z0 + v·a) for u in PATCH_U and v in PATCH_V, divided by μ0 R0 Ip / (2π) as the harmonics are.

On 300 DIII-D train shots with 100 held out, gradient boosting on v3 plus these points predicts the true
boundary's elongation with R² 0.71 (v3 alone 0.64) and its volume with 0.73 (0.70).

Points listed in DROPPED fail the MAST overlap test on 60 MAST test shots (under 90% of MAST frames inside
DIII-D's 0.5–99.5% range, or MAST's median more than 2 DIII-D sd away). Most sit at the lower inboard corner,
where MAST's lower P-coils lie inside the plasma-normalised region. The centre point is 0 by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import features_v2 as fv
import features_v3 as f3
from common import _MU0

PATCH_U = np.linspace(-1.2, 1.2, 9)
PATCH_V = np.linspace(-2.2, 2.2, 13)
# (row in PATCH_V, column in PATCH_U), from proto_patch.py's pass map.
DROPPED = {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 8), (2, 2), (2, 3), (6, 4), (10, 3), (11, 0), (12, 1)}
KEPT = [(i, j) for i in range(len(PATCH_V)) for j in range(len(PATCH_U)) if (i, j) not in DROPPED]
PATCH = [f"psi_u{PATCH_U[j]:+.1f}_v{PATCH_V[i]:+.2f}" for i, j in KEPT]
FLAGS = fv.FLAGS
FEATURES = [n for n in f3.FEATURES if n not in FLAGS] + PATCH + FLAGS
N_FEATURES = len(FEATURES)
VERSION = "v4"


def _patch_weights(machine: str) -> np.ndarray:
    """Bilinear weights (1 + len(KEPT), 4225): the centre, then the kept points."""
    geo = fv.geometry(machine)
    r, z, R0, Z0, a = geo["r"], geo["z"], geo["R0"], geo["Z0"], geo["a"]
    pts = [(R0, Z0)] + [(R0 + PATCH_U[j] * a, Z0 + PATCH_V[i] * a) for i, j in KEPT]
    W = np.zeros((len(pts), len(r) * len(z)))
    for k, (pr, pz) in enumerate(pts):
        if not (r[0] <= pr <= r[-1] and z[0] <= pz <= z[-1]):
            raise RuntimeError(f"patch point ({pr:.3f}, {pz:.3f}) lies outside the {machine} grid")
        ir = min(max(int(np.searchsorted(r, pr, side="right") - 1), 0), len(r) - 2)
        iz = min(max(int(np.searchsorted(z, pz, side="right") - 1), 0), len(z) - 2)
        tr = (pr - r[ir]) / (r[ir + 1] - r[ir])
        tz = (pz - z[iz]) / (z[iz + 1] - z[iz])
        W[k, iz * len(r) + ir] += (1 - tz) * (1 - tr)
        W[k, iz * len(r) + ir + 1] += (1 - tz) * tr
        W[k, (iz + 1) * len(r) + ir] += tz * (1 - tr)
        W[k, (iz + 1) * len(r) + ir + 1] += tz * tr
    return W


def patch(cur: np.ndarray, ip: np.ndarray, pictures: dict, machine: str) -> np.ndarray:
    """(T, len(PATCH)) from mirrored currents (T, columns) and mirrored Ip (T,) in kA."""
    geo = fv.geometry(machine)
    samples = cur @ (pictures["green"] @ _patch_weights(machine).T)
    scale = _MU0 * geo["R0"] * np.maximum(np.abs(ip), fv.IP_FLOOR_KA) * 1000.0 / (2.0 * np.pi)
    return (samples[:, 1:] - samples[:, :1]) / scale[:, None]


def shot_features(path: Path, pictures: dict) -> np.ndarray:
    """(T, N_FEATURES) float32: the v3 columns without flags, the patch, then the flags."""
    path = Path(path)
    base, extra = fv.shot_features(path, pictures, return_inputs=True)
    harm = f3.harmonics(extra["cur"], extra["ip"], pictures, extra["machine"])
    grid = patch(extra["cur"], extra["ip"], pictures, extra["machine"])
    n_flag = len(FLAGS)
    out = np.concatenate([base[:, :-n_flag], harm, grid, base[:, -n_flag:]], axis=1)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)
