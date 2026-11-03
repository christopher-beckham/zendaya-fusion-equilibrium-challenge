"""Vacuum poloidal-flux Green's functions for the DIII-D PF coils on the 65x65 EFIT grid.

psi here is EFIT's convention, poloidal flux per radian (Wb/rad) = R * A_phi. For a circular
filament of radius a at height z0 carrying current I:

    psi(R, Z) = mu0 I / (2 pi) * sqrt(a R) / k * [(2 - k^2) K(k^2) - 2 E(k^2)],
    k^2 = 4 a R / ((R + a)^2 + (Z - z0)^2)

Each coil rectangle is split into sub-filaments (sheared into a parallelogram where EFIT's
AF/AF2 angles are non-zero) and averaged, so the result is the flux per unit total coil current.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi


def filament_psi(a: np.ndarray, z0: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Flux per unit current (Wb/rad/A) at points (R, Z) from filaments (a, z0). Broadcasts."""
    a, z0, R, Z = np.broadcast_arrays(a, z0, R, Z)
    k2 = 4.0 * a * R / ((R + a) ** 2 + (Z - z0) ** 2)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-9)             # a grid point sitting on a filament
    k = np.sqrt(k2)
    return MU0 / (2 * np.pi) * np.sqrt(a * R) / k * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))


def coil_filaments(R0, Z0, W, H, angle1, angle2, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """n x n filament positions filling one EFIT coil (rectangle or AF/AF2 parallelogram)."""
    u = (np.arange(n) + 0.5) / n - 0.5              # [-1/2, 1/2) cell centres
    uu, vv = np.meshgrid(u, u, indexing="ij")
    t1 = np.tan(np.deg2rad(angle1)) if angle1 else 0.0
    c2 = 1.0 / np.tan(np.deg2rad(angle2)) if angle2 else 0.0
    r = R0 + uu * W + vv * H * c2
    z = Z0 + vv * H + uu * W * t1
    return r.ravel(), z.ravel()


def coil_basis(row: dict, grid_R: np.ndarray, grid_Z: np.ndarray, n: int = 8
               ) -> tuple[list[str], np.ndarray]:
    """Per-coil flux images (C, nZ, nR) per unit current, keyed by `coil_input_column`."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)            # rows = Z, cols = R (EFIT layout)
    cols = [str(c) for c in row["coil_input_column"]]
    images = np.zeros((len(cols), len(grid_Z), len(grid_R)))
    for i in range(len(cols)):
        fr, fz = coil_filaments(float(row["coil_R"][i]), float(row["coil_Z"][i]),
                                float(row["coil_width"][i]), float(row["coil_height"][i]),
                                float(row["coil_angle1"][i]), float(row["coil_angle2"][i]), n)
        g = filament_psi(fr[:, None, None], fz[:, None, None], RR[None], ZZ[None])
        images[i] = g.mean(axis=0)
    return cols, images
