"""Green's functions and the two fixed matrices of a machine, cached on disk.

  coil_matrix(m)  (n_columns, nZ*nR)  flux on the grid per 1 kA in each input column (coils are exact
                  circular filaments filling each coil element)
  cell_matrix(m)  (n_cells, nZ*nR)    flux on the grid per 1 A spread uniformly over each plasma cell
                  (wall-region cell), the same order as np.nonzero(m.wall_region())
Flux is "per radian" (Wb/rad), produced by current flowing in +phi. Cache folder: $GSOLVE_CACHE or
~/.cache/gsolve (MAST: about 40 MB, built once in about a minute).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

from .geometry import MU0, Machine


def greens(Rc, Zc, R, Z):
    """Flux per radian at (R, Z) from a 1 A circular loop of radius Rc at height Zc (arguments broadcast).

    psi = mu0/(2 pi) sqrt(Rc R)/k [(2 - k^2) K(k^2) - 2 E(k^2)],  k^2 = 4 Rc R / ((R + Rc)^2 + (Z - Zc)^2),
    K and E the complete elliptic integrals. k^2 is clipped so a point on the loop itself stays finite."""
    Rc, Zc, R, Z = (np.asarray(x, np.float64) for x in (Rc, Zc, R, Z))
    k2 = np.clip(4.0 * Rc * R / ((R + Rc) ** 2 + (Z - Zc) ** 2), 1e-12, 1.0 - 1e-9)
    return MU0 / (2 * np.pi) * np.sqrt(Rc * R) / np.sqrt(k2) * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))


def filaments(R, Z, w, h, a1=0.0, a2=0.0, step=0.01, nmin=2):
    """Filament positions filling one coil element (EFIT convention: angle1 shears Z along R, angle2 R along Z)."""
    nr, nz = max(nmin, int(np.ceil(w / step))), max(nmin, int(np.ceil(h / step)))
    DR, DZ = np.meshgrid((np.arange(nr) + 0.5) / nr * w - w / 2, (np.arange(nz) + 0.5) / nz * h - h / 2, indexing="ij")
    Rf, Zf = R + DR, Z + DZ
    if a1:
        Zf = Zf + DR * np.tan(np.radians(a1))
    if a2:
        Rf = Rf + DZ / np.tan(np.radians(a2))
    return Rf.ravel(), Zf.ravel()


def _cached(m: Machine, kind: str, build):
    folder = Path(os.environ.get("GSOLVE_CACHE", Path.home() / ".cache" / "gsolve"))
    f = folder / f"{m.key()}_{kind}.npy"
    if f.exists():
        return np.load(f)
    a = build(m)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, f)
    return a


def _coil_matrix(m: Machine):
    RR, ZZ = np.meshgrid(m.R, m.Z)
    Rp, Zp = RR.ravel(), ZZ.ravel()
    e = m.elements
    M = np.zeros((len(m.columns), Rp.size))
    for i in range(len(e["R"])):
        Rf, Zf = filaments(e["R"][i], e["Z"][i], e["width"][i], e["height"][i], e["angle1"][i], e["angle2"][i],
                           step=m.filament_step)
        flux = greens(Rf[:, None], Zf[:, None], Rp[None], Zp[None]).mean(0)
        M[e["col"][i]] += e["turns"][i] * flux
    scale = np.array([m.column_scale.get(c, 1.0) for c in m.columns])
    return M * scale[:, None] * 1e3


def _cell_matrix(m: Machine, near_cells=4, sub_near=6, sub_far=2):
    RR, ZZ = np.meshgrid(m.R, m.Z)
    Rp, Zp = RR.ravel(), ZZ.ravel()
    dR, dZ = m.R[1] - m.R[0], m.Z[1] - m.Z[0]
    gz, gr = np.divmod(np.arange(Rp.size), len(m.R))
    iz, ir = np.nonzero(m.wall_region())
    out = np.zeros((iz.size, Rp.size), np.float32)
    for n, (a, b) in enumerate(zip(iz, ir)):
        near = (np.abs(gz - a) <= near_cells) & (np.abs(gr - b) <= near_cells)
        for sub, sel in ((sub_far, ~near), (sub_near, near)):     # finer filaments next to the cell itself
            off = (np.arange(sub) + 0.5) / sub - 0.5
            oR, oZ = (x.ravel()[:, None] for x in np.meshgrid(off * dR, off * dZ, indexing="ij"))
            out[n, sel] = greens(m.R[b] + oR, m.Z[a] + oZ, Rp[sel][None], Zp[sel][None]).mean(0)
    return out


@functools.lru_cache(maxsize=8)
def _get(m: Machine, kind: str):
    return _cached(m, kind, _coil_matrix if kind == "coils" else _cell_matrix)


def coil_matrix(m: Machine) -> np.ndarray:
    return _get(m, "coils")


def cell_matrix(m: Machine) -> np.ndarray:
    return _get(m, "cells")


def coil_flux(m: Machine, currents_kA) -> np.ndarray:
    """(B, nZ, nR) vacuum flux of coil currents (B, n_columns) in kA (physical sign, Ip > 0)."""
    I = np.atleast_2d(np.asarray(currents_kA, np.float64))
    return (I @ coil_matrix(m)).reshape(-1, *m.shape)
