"""Green's function against an independent quadrature and the dipole limit; matrices against the stored reference."""
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from gsolve import cell_matrix, coil_flux, greens, load_machine

MU0 = 4e-7 * np.pi
REF = np.load(Path(__file__).parent / "data" / "regression.npz")


def flux_by_quadrature(Rc, Zc, R, Z):
    """psi = R A_phi of a 1 A loop, A_phi = mu0/(4 pi) int Rc cos(phi) / |r - r'| dphi (Biot-Savart)."""
    f = lambda p: Rc * np.cos(p) / np.sqrt(R ** 2 + Rc ** 2 - 2 * R * Rc * np.cos(p) + (Z - Zc) ** 2)  # noqa: E731
    return R * MU0 / (4 * np.pi) * quad(f, 0, 2 * np.pi, epsabs=0, epsrel=1e-12, limit=200)[0]


def test_greens_against_quadrature():
    pts = [(0.5, 0.0, 0.8, 0.3), (1.2, -0.4, 0.3, 0.9), (0.85, 0.0, 0.86, 0.01), (2.0, 1.0, 0.1, -1.5), (0.15, 1.4, 1.5, 0.0)]
    for Rc, Zc, R, Z in pts:
        assert np.isclose(greens(Rc, Zc, R, Z), flux_by_quadrature(Rc, Zc, R, Z), rtol=1e-9, atol=0)


def test_greens_dipole_limit():
    Rc, R, Z = 0.7, 30.0, 40.0                       # far away: psi -> mu0 I Rc^2 R^2 / (4 r^3)
    assert np.isclose(greens(Rc, 0.0, R, Z), MU0 * Rc ** 2 * R ** 2 / (4 * np.hypot(R, Z) ** 3), rtol=1e-3)


def test_matrices_match_reference():
    m = load_machine("mast")
    assert (m.wall_region() == REF["W"]).all()
    G = cell_matrix(m)
    assert np.allclose(G[REF["G_idx"]], REF["G_rows"], rtol=1e-6, atol=1e-6 * np.abs(REF["G_rows"]).max())
    for case in "ABC":
        ext, ref = coil_flux(m, REF[f"{case}_I_kA"])[0], REF[f"{case}_ext"]
        assert np.abs(ext - ref).max() < 1e-6 * np.abs(ref).max()
