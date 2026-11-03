"""features_v2 plus poloidal harmonics of the coil picture. One code path for both machines.

v2 describes the coils' field by Bz, Br and ∂²Bz/∂R² at one point. v3 adds the shape of ψ_ext around
the plasma: on circles of radius ρ·a about (R0, Z0), for ρ in HARMONIC_RHO,

    ψ_ext(θ) − ψ_ext(R0, Z0) ≈ Σ_m c_m cos mθ + s_m sin mθ,   m = 1..N_HARMONIC,

with θ measured from the outboard midplane. Coefficients are divided by μ0 R0 Ip / (2π), the flux scale of
the plasma current, so each is a ratio of coil shaping flux to plasma flux. m = 2 drives elongation,
m = 3 triangularity, the sin terms up-down asymmetry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import features_v2 as fv
from common import _MU0

HARMONIC_RHO = (0.5, 1.0)
N_HARMONIC = 5
N_THETA = 64
_ALL_HARMONICS = [f"{kind}{m}_rho{rho}" for rho in HARMONIC_RHO for kind in ("c", "s") for m in range(1, N_HARMONIC + 1)]
# c4 fails the MAST overlap test at both radii (86-93% inside, median about -2 DIII-D sd), so it is left out.
DROPPED = ("c4_rho0.5", "c4_rho1.0")
HARMONICS = [n for n in _ALL_HARMONICS if n not in DROPPED]
FLAGS = fv.FLAGS
FEATURES = [n for n in fv.FEATURES if n not in FLAGS] + HARMONICS + FLAGS
N_FEATURES = len(FEATURES)
VERSION = "v3"


def _ring_weights(r: np.ndarray, z: np.ndarray, R0: float, Z0: float, a: float):
    """Sparse bilinear weights (index, weight) for the ring points and the centre on the flat 65×65 grid."""
    theta = 2.0 * np.pi * np.arange(N_THETA) / N_THETA
    pts = [(R0, Z0)] + [(R0 + rho * a * np.cos(t), Z0 + rho * a * np.sin(t)) for rho in HARMONIC_RHO for t in theta]
    n_r = len(r)
    W = np.zeros((len(pts), len(r) * len(z)))
    for i, (pr, pz) in enumerate(pts):
        if not (r[0] <= pr <= r[-1] and z[0] <= pz <= z[-1]):
            raise RuntimeError(f"ring point ({pr:.3f}, {pz:.3f}) lies outside the grid")
        ir = min(max(int(np.searchsorted(r, pr, side="right") - 1), 0), len(r) - 2)
        iz = min(max(int(np.searchsorted(z, pz, side="right") - 1), 0), len(z) - 2)
        tr = (pr - r[ir]) / (r[ir + 1] - r[ir])
        tz = (pz - z[iz]) / (z[iz + 1] - z[iz])
        W[i, iz * n_r + ir] += (1 - tz) * (1 - tr)
        W[i, iz * n_r + ir + 1] += (1 - tz) * tr
        W[i, (iz + 1) * n_r + ir] += tz * (1 - tr)
        W[i, (iz + 1) * n_r + ir + 1] += tz * tr
    return W, theta


def harmonics(cur: np.ndarray, ip: np.ndarray, pictures: dict, machine: str) -> np.ndarray:
    """(T, len(HARMONICS)) from mirrored currents (T, columns) and mirrored Ip (T,) in kA."""
    geo = fv.geometry(machine)
    W, theta = _ring_weights(geo["r"], geo["z"], geo["R0"], geo["Z0"], geo["a"])
    # Ring samples per unit current of each column, then per frame.
    per_col = pictures["green"] @ W.T            # (columns, 1 + rings·N_THETA)
    samples = cur @ per_col                       # (T, 1 + rings·N_THETA)
    centre = samples[:, :1]
    ring = (samples[:, 1:] - centre).reshape(len(cur), len(HARMONIC_RHO), N_THETA)
    m = np.arange(1, N_HARMONIC + 1)
    cos_b = np.cos(np.outer(m, theta)) * (2.0 / N_THETA)
    sin_b = np.sin(np.outer(m, theta)) * (2.0 / N_THETA)
    scale = _MU0 * geo["R0"] * np.maximum(np.abs(ip), fv.IP_FLOOR_KA) * 1000.0 / (2.0 * np.pi)
    out = []
    for k in range(len(HARMONIC_RHO)):
        out.append(ring[:, k] @ cos_b.T)
        out.append(ring[:, k] @ sin_b.T)
    full = np.concatenate(out, axis=1) / scale[:, None]
    return full[:, [_ALL_HARMONICS.index(n) for n in HARMONICS]]


def shot_features(path: Path, pictures: dict) -> np.ndarray:
    """(T, N_FEATURES) float32: the v2 columns, then the harmonics, then the flags."""
    path = Path(path)
    base, extra = fv.shot_features(path, pictures, return_inputs=True)
    harm = harmonics(extra["cur"], extra["ip"], pictures, extra["machine"])
    n_flag = len(FLAGS)
    out = np.concatenate([base[:, :-n_flag], harm, base[:, -n_flag:]], axis=1)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)
