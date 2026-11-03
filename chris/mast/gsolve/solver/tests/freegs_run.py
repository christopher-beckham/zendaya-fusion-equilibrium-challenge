"""Run one FreeGS 0.8.2 free-boundary solve for test_freegs.py (in an interpreter that has FreeGS; no torch needed).

    python freegs_run.py in.npz out.npz
in.npz: coil filaments (R, Z, amps per kA of each column, column index), column currents (kA), Ip, F_vac, paxis,
alpha_m, alpha_n, R0, grid extent, wall box, start-current parameters. FreeGS gets the same coils (one FreeGS coil
per input column made of our filaments), the same ConstrainPaxisIp profile and the wall box as its limiter.
Patches needed with current numpy/scipy: scipy's removed interp2d, critical points
restricted to the wall box (the MAST grid also holds the solenoid and P coils), and a fallback when FreeGS's
limiter check leaves no wall point. FreeGS 0.8.2's critical.py also needs a one-line numpy-2 fix (a size-1 array
assigned to a matrix element); use an install that has it.
"""
import sys

import numpy as np
import scipy.interpolate as si

import freegs
import freegs.critical as fcrit
import freegs.equilibrium as feq
from freegs.coil import Coil
from freegs.gradshafranov import Greens, GreensBr, GreensBz


class _Interp2d:
    def __init__(self, x=None, y=None, z=None, **kw):
        self.f = si.RectBivariateSpline(np.asarray(y), np.asarray(x), np.asarray(z), kx=1, ky=1)

    def __call__(self, xq, yq):
        return np.atleast_1d(self.f(np.atleast_1d(yq), np.atleast_1d(xq), grid=False))


feq.interpolate.interp2d = _Interp2d
BOX = [None]
_find = fcrit.find_critical


def _boxed(R, Z, psi, *a, **kw):
    opt, xpt = _find(R, Z, psi, *a, **kw)
    r0, r1, z0, z1 = BOX[0]
    ins = lambda p: r0 <= p[0] <= r1 and z0 <= p[1] <= z1  # noqa: E731
    return [p for p in opt if ins(p)], [p for p in xpt if ins(p)]


fcrit.find_critical = _boxed
_ubp = feq.Equilibrium._updateBoundaryPsi


def _safe_ubp(self, psi=None):
    try:
        return _ubp(self, psi)
    except ValueError as e:
        if "empty sequence" not in str(e):
            raise
        self.check_limited = False
        try:
            return _ubp(self, psi)
        finally:
            self.check_limited = True


feq.Equilibrium._updateBoundaryPsi = _safe_ubp


class FilamentCoil(Coil):
    """Fixed weighted filaments; current = column value (kA); weight = amps per filament per kA."""

    def __init__(self, Rf, Zf, wf):
        self.Rf, self.Zf, self.wf = Rf, Zf, wf
        self._R, self._Z = float(Rf.mean()), float(Zf.mean())
        self.current, self.turns, self.control, self.area, self._area = 0.0, 1, False, 0.0, 0.0

    def controlPsi(self, R, Z):
        return sum(w * Greens(r, z, R, Z) for r, z, w in zip(self.Rf, self.Zf, self.wf))

    def controlBr(self, R, Z):
        return sum(w * GreensBr(r, z, R, Z) for r, z, w in zip(self.Rf, self.Zf, self.wf))

    def controlBz(self, R, Z):
        return sum(w * GreensBz(r, z, R, Z) for r, z, w in zip(self.Rf, self.Zf, self.wf))

    def createPsiGreens(self, R, Z):
        return self.controlPsi(R, Z)

    def calcPsiFromGreens(self, pgreen):
        return self.current * pgreen

    R = property(lambda s: s._R, lambda s, v: setattr(s, "_R", v))
    Z = property(lambda s: s._Z, lambda s, v: setattr(s, "_Z", v))


def main():
    d = dict(np.load(sys.argv[1]))
    BOX[0] = tuple(d["box"])
    coils = []
    for j, I in enumerate(d["I_kA"]):
        s = d["col"] == j
        c = FilamentCoil(d["Rf"][s], d["Zf"][s], d["wf"][s])
        c.current = float(I)
        coils.append((f"C{j}", c))
    r0, r1, z0, z1 = BOX[0]
    tok = freegs.machine.Machine(coils, wall=freegs.machine.Wall([r0, r1, r1, r0, r0], [z0, z0, z1, z1, z0]))
    (Rmin, Rmax, nR), (Zmin, Zmax, nZ) = d["grid_R"], d["grid_Z"]
    eq = freegs.Equilibrium(tokamak=tok, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=int(nR), ny=int(nZ),
                            boundary=freegs.boundary.freeBoundaryHagenow, check_limited=True)
    prof = freegs.jtor.ConstrainPaxisIp(eq, float(d["paxis"]), float(d["ip"]), float(d["fvac"]), alpha_m=float(d["am"]),
                                        alpha_n=float(d["an"]), Raxis=float(d["R0"]))
    x, y = (eq.R - 0.9) / 0.55, eq.Z / (1.8 * 0.55)                  # same analytic start current as gsolve
    rho2 = (x + 0.3 * y ** 2) ** 2 + y ** 2
    J0 = np.where(rho2 < 1, 1 - rho2, 0.0)
    J0 *= float(d["ip"]) / (J0.sum() * (eq.R[1, 0] - eq.R[0, 0]) * (eq.Z[0, 1] - eq.Z[0, 0]))
    eq.solve(prof, Jtor=J0)
    conv = False
    for _ in range(300):                                             # Picard, blend 0.5, mirror-averaged
        old = eq.plasma_psi.copy()
        eq.check_limited = True
        eq.solve(prof, psi=eq.psi())
        bl = 0.5 * (eq.plasma_psi + old)
        bl = 0.5 * (bl + bl[:, ::-1])
        eq._updatePlasmaPsi(bl)
        if np.max(np.abs(bl - old)) / abs(eq.psi_axis - eq.psi_bndry) < 1e-5:
            conv = True
            break
    opt, _ = fcrit.find_critical(eq.R, eq.Z, eq.psi())
    pts = d["green_pts"]
    np.savez(sys.argv[2], psi=eq.psi().T, converged=conv, R_ax=opt[0][0], Z_ax=opt[0][1], psi_ax=eq.psi_axis,
             psi_b=eq.psi_bndry, L=prof.L, beta0=prof.Beta0, greens=Greens(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]))


if __name__ == "__main__":
    main()
