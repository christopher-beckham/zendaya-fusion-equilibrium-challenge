"""Minimal free-boundary Grad-Shafranov solver on the 65x65 challenge grid, CPU only.

The plasma is a set of toroidal current rings, one per grid cell inside the machine envelope.
Total flux = vacuum flux of the coils (our coil pictures) + sum over cells of (Green's function x cell current).
Because every cell has its own Green's function, there is no outer boundary condition to solve for: the
free-boundary problem becomes a fixed-point problem on the in-envelope cell currents.

Picard iteration:
    psi = psi_coils + G @ I_cells
    axis = flux maximum near the current centre; boundary flux = highest level whose region around the axis
    first touches the envelope edge (handles both X-point and limiter boundaries)
    J = profile(psi_N, R), scaled so that sum(I) = Ip
    vertical position: the current centroid is pinned at Zc; Zc is moved by a secant step until the centroid of the
    profile current computed from psi agrees with Zc (the vertically unstable direction is handled outside the
    Picard map, as in forward free-boundary codes)

Conventions: internally psi is "physical" (positive current gives a flux maximum at the axis). The challenge
stores DIII-D with the axis at a minimum; our coil pictures carry sign S_PIC = -1 relative to the filament
formula, so label-convention psi = S_PIC * psi_phys. Units: Wb/rad, A, m, Pa.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

NN = Path(__file__).resolve().parents[1]
REPO = NN.parents[1]
sys.path.insert(0, str(NN))
from greens import filament_psi  # noqa: E402  (flux per kA.turn, Wb/rad)

MU0 = 4e-7 * np.pi
S_PIC = -1.0
CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))


class Grid:
    """Grid, envelope and the in-envelope plasma Green's matrix for one machine ('d3d' or 'mast')."""

    def __init__(self, machine: str, mast_mask: str | None = None):
        """MAST's region is mast_region's (`mast_mask`, default the FEC_MAST_MASK selection); DIII-D its envelope."""
        self.machine = machine
        if machine == "mast":
            import mast_region
            reg = mast_region.region(mast_mask)
            self.r, self.z, self.mask = reg["r"], reg["z"], reg["mask"]
        else:
            z = np.load(REPO / f"fusion_scoring/masks/{machine}_envelope.npz")
            self.r = np.asarray(z["grid_R"], np.float64)
            self.z = np.asarray(z["grid_Z"], np.float64)
            self.mask = np.asarray(z["mask_coarse"], bool)
        self.RR, self.ZZ = np.meshgrid(self.r, self.z, indexing="xy")
        self.dr = float(self.r[1] - self.r[0])
        self.dz = float(self.z[1] - self.z[0])
        self.dA = self.dr * self.dz
        self.idx = np.flatnonzero(self.mask.ravel())
        R, Z = self.RR.ravel(), self.ZZ.ravel()
        rc, zc = R[self.idx], Z[self.idx]
        # G[p, c]: flux at grid point p from 1 A in cell c. filament_psi is per kA.turn.
        G = filament_psi(R[:, None], Z[:, None], rc[None, :], zc[None, :]) / 1000.0
        # Self term: a filament evaluated at the geometric mean distance of a dr x dz cell from itself.
        gmd = 0.2235 * (self.dr + self.dz)
        self_flux = filament_psi(rc, zc + gmd, rc, zc) / 1000.0
        G[self.idx, np.arange(len(self.idx))] = self_flux
        self.G = np.ascontiguousarray(G)
        # Cells inside the envelope that touch the outside: reaching one means the region is open.
        outside = ~self.mask
        self.edge = self.mask & ndimage.binary_dilation(outside, structure=np.ones((3, 3), bool))
        er = self.RR[self.mask]
        self.R0 = 0.5 * (er.min() + er.max())
        self.Z0 = float(self.ZZ[self.mask].mean())


# ----------------------------------------------------------------------------------------------
# boundary and axis
# ----------------------------------------------------------------------------------------------

def find_axis(g: Grid, psi: np.ndarray, near: tuple[float, float]) -> tuple[int, int]:
    """Local maximum of psi inside the envelope (not on its edge) nearest to `near` (R, Z)."""
    loc = (psi == ndimage.maximum_filter(psi, size=3)) & g.mask & ~g.edge
    iz, ir = np.nonzero(loc)
    if len(iz) == 0:
        k = int(np.argmax(np.where(g.mask, psi, -np.inf)))
        return divmod(k, psi.shape[1])
    d = (g.r[ir] - near[0]) ** 2 + (g.z[iz] - near[1]) ** 2
    k = int(np.argmin(d))
    return int(iz[k]), int(ir[k])


def region_at(g: Grid, psi: np.ndarray, level: float, iz: int, ir: int) -> np.ndarray | None:
    """Connected region {psi > level} inside the envelope containing the axis, or None if it reaches the edge."""
    lab, _ = ndimage.label((psi > level) & g.mask)
    k = lab[iz, ir]
    if k == 0:
        return None
    reg = lab == k
    if (reg & g.edge).any():
        return None
    return reg


def boundary(g: Grid, psi: np.ndarray, iz: int, ir: int, n_iter: int = 24):
    """(psi_boundary, region). The lowest level whose axis region stays clear of the envelope edge."""
    hi = float(psi[iz, ir])
    lo = float(psi[g.mask].min())
    best = None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        reg = region_at(g, psi, mid, iz, ir)
        if reg is None:
            lo = mid
        else:
            hi, best = mid, reg
    if best is None:
        best = np.zeros_like(g.mask)
        best[iz, ir] = True
    return hi, best


# ----------------------------------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------------------------------

class PowerProfile:
    """J = lam * [beta0 R/R0 + (1 - beta0) R0/R] * (1 - psiN^am)^an  (FreeGS ConstrainBetapIp form)."""

    def __init__(self, beta0=0.4, am=2.0, an=1.5):
        self.beta0, self.am, self.an = beta0, am, an

    def current(self, g, psiN, reg, ip_a, psi_span, row=None):
        shape = np.clip(1.0 - np.clip(psiN, 0, 1) ** self.am, 0, None) ** self.an
        rfac = self.beta0 * g.RR / g.R0 + (1 - self.beta0) * g.R0 / g.RR
        J = np.where(reg, rfac * shape, 0.0)
        tot = J.sum() * g.dA
        return J * (ip_a / tot) if tot > 0 else J


class ThomsonProfile:
    """Pressure term from Thomson, FF' term from the Ip constraint.

    p(psiN) = sum_k c_k (1 - psiN)^k, k = 1..3, c_k >= 0, fitted to ION_FACTOR * ne * Te at the Thomson channels
    mapped onto the current psiN. J = R dp/dpsi + lam * (1 - psiN^am)^an / R, lam from Ip.
    """

    ION_FACTOR = 1.9

    def __init__(self, te, ne, cr, cz, am=2.0, an=1.0, pscale=1.0):
        self.te, self.ne, self.cr, self.cz = te, ne, cr, cz
        self.am, self.an, self.pscale = am, an, pscale
        self.coef = np.zeros(3)

    def fit(self, g, psiN_fn):
        ok = np.isfinite(self.te) & np.isfinite(self.ne) & np.isfinite(self.cr) & np.isfinite(self.cz) & (self.te > 0)
        if ok.sum() < 4:
            self.coef = np.zeros(3)
            return
        x = psiN_fn(self.cr[ok], self.cz[ok])
        p = self.ION_FACTOR * self.pscale * self.te[ok] * self.ne[ok] * 1e19 * 1.602176634e-19
        inside = (x >= 0) & (x <= 1.0)
        if inside.sum() < 4:
            self.coef = np.zeros(3)
            return
        from scipy.optimize import nnls
        u = 1.0 - x[inside]
        A = np.stack([u, u ** 2, u ** 3], axis=1)
        self.coef, _ = nnls(A, p[inside])

    def current(self, g, psiN, reg, ip_a, psi_span, row=None):
        x = np.clip(psiN, 0, 1)
        u = 1.0 - x
        dp_dx = -(self.coef[0] + 2 * self.coef[1] * u + 3 * self.coef[2] * u ** 2)
        dp_dpsi = -dp_dx / psi_span  # psi_span = psi_axis - psi_boundary > 0
        Jp = np.where(reg, g.RR * dp_dpsi, 0.0)
        ip_p = Jp.sum() * g.dA
        if ip_p > 0.9 * ip_a:  # pressure alone would exceed Ip: scale it down
            Jp *= 0.9 * ip_a / ip_p
            ip_p = 0.9 * ip_a
        shape = np.where(reg, np.clip(1 - x ** self.am, 0, None) ** self.an / g.RR, 0.0)
        tot = shape.sum() * g.dA
        Jf = shape * ((ip_a - ip_p) / tot) if tot > 0 else shape
        return Jp + Jf


class ThomsonBetaProfile(PowerProfile):
    """Generic power-law shape; the pressure share beta0 is set so the model's volume-averaged pressure matches
    Thomson (ION_FACTOR x ne x Te mapped onto the current psiN). This fixes poloidal beta from the measurement and
    leaves the current peaking (li) generic.
    """

    ION_FACTOR = 1.9

    def __init__(self, te, ne, cr, cz, am=2.0, an=1.0, beta0=0.4, relax=0.5):
        super().__init__(beta0, am, an)
        self.te, self.ne, self.cr, self.cz, self.relax = te, ne, cr, cz, relax
        self._pn = None

    def fit(self, g, psiN_fn):
        ok = np.isfinite(self.te) & np.isfinite(self.ne) & np.isfinite(self.cr) & np.isfinite(self.cz) & (self.te > 0)
        if ok.sum() < 4:
            self._pn = None
            return
        x = psiN_fn(self.cr[ok], self.cz[ok])
        p = self.ION_FACTOR * self.te[ok] * self.ne[ok] * 1e19 * 1.602176634e-19
        keep = (x >= 0) & (x <= 1.0)
        if keep.sum() < 4:
            self._pn = None
            return
        o = np.argsort(x[keep])
        self._pn = (x[keep][o], p[keep][o])

    def current(self, g, psiN, reg, ip_a, psi_span, row=None):
        J = super().current(g, psiN, reg, ip_a, psi_span)
        if self._pn is None or not reg.any():
            return J
        # model pressure for this J: p'(psi) = lam * beta0 / R0 * shape(psiN), integrated from the edge inward
        x = np.clip(psiN[reg], 0, 1)
        xs = np.linspace(0, 1, 201)
        sh = np.clip(1 - xs ** self.am, 0, None) ** self.an
        cum = np.concatenate([[0], np.cumsum(0.5 * (sh[1:] + sh[:-1]) * np.diff(xs))])
        integral_to_edge = cum[-1] - np.interp(x, xs, cum)  # int_x^1 shape dx'
        rfac = self.beta0 * g.RR / g.R0 + (1 - self.beta0) * g.R0 / g.RR
        shape = np.where(reg, rfac * np.clip(1 - np.clip(psiN, 0, 1) ** self.am, 0, None) ** self.an, 0.0)
        lam = ip_a / (shape.sum() * g.dA)
        w = g.RR[reg]
        p_model_unit = lam / g.R0 * psi_span * integral_to_edge  # per unit beta0
        p_th = np.interp(x, *self._pn, left=self._pn[1][0], right=self._pn[1][-1])
        avg_model = np.sum(p_model_unit * w) / np.sum(w)
        avg_th = np.sum(p_th * w) / np.sum(w)
        if avg_model > 0:
            target = float(np.clip(avg_th / avg_model, 0.0, 0.95))
            self.beta0 = (1 - self.relax) * self.beta0 + self.relax * target
        return J


# ----------------------------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------------------------

def _shift(g: Grid, J: np.ndarray, dr: float, dz: float) -> np.ndarray:
    """Move J by (dr, dz) metres (linear interpolation), keep it inside the envelope."""
    if dr == 0 and dz == 0:
        return J
    out = ndimage.shift(J, (dz / g.dz, dr / g.dr), order=1, mode="constant", cval=0.0)
    return np.where(g.mask, np.clip(out, 0, None), 0.0)


def _centroid(g, J):
    s = J.sum()
    if s <= 0:
        return g.R0, g.Z0
    return float((J * g.RR).sum() / s), float((J * g.ZZ).sum() / s)


def initial_current(g: Grid, ip_ka: float, rc: float | None = None, zc: float | None = None):
    """Parabolic blob, half the envelope size, at (rc, zc) (default: envelope centre)."""
    er, ez = g.RR[g.mask], g.ZZ[g.mask]
    a = 0.25 * (er.max() - er.min())
    b = 0.25 * (ez.max() - ez.min())
    rc = 0.5 * (er.max() + er.min()) + 0.1 * a if rc is None else rc
    zc = g.Z0 if zc is None else zc
    rho2 = ((g.RR - rc) / a) ** 2 + ((g.ZZ - zc) / b) ** 2
    J0 = np.where(g.mask & (rho2 < 1), 1 - rho2, 0.0)
    return J0 * (abs(ip_ka) * 1000.0 / (J0.sum() * g.dA))


class _Secant:
    """1-D root finder for delta(c) = 0 from successive (c, delta) pairs; first step c + gain * delta."""

    def __init__(self, gain, max_step):
        self.hist, self.gain, self.max_step = [], gain, max_step

    def step(self, c, delta):
        self.hist.append((c, delta))
        if len(self.hist) >= 2 and self.hist[-1][0] != self.hist[-2][0]:
            (c1, d1), (c2, d2) = self.hist[-2], self.hist[-1]
            slope = (d2 - d1) / (c2 - c1)
            st = -d2 / slope if abs(slope) > 1e-3 else self.gain * d2
        else:
            st = self.gain * delta
        return float(np.clip(st, -self.max_step, self.max_step))


def solve(g: Grid, psi_ext: np.ndarray, ip_ka: float, profile, J0: np.ndarray | None = None,
          max_outer: int = 14, inner: int = 25, omega: float = 0.5, tol: float = 1e-4,
          fit_every: int = 5, pin_r: bool = True, verbose: bool = False) -> dict:
    """Free-boundary equilibrium for fixed coil flux `psi_ext` (65x65, physical sign) and Ip (kA).

    The current centroid is pinned at (rc, zc) during each block of `inner` Picard steps; between blocks a secant
    step per direction moves the pin until the centroid of the profile current computed from psi agrees with it.
    Returns psi (physical sign), J, axis, boundary flux, convergence info.
    """
    ip_a = abs(ip_ka) * 1000.0
    J = initial_current(g, ip_ka) if J0 is None else J0 * (ip_a / (J0.sum() * g.dA))
    rc, zc = _centroid(g, J)
    sec_r, sec_z = _Secant(0.5, 0.1), _Secant(0.5, 0.1)
    near = (rc, zc)
    it_total = 0
    info = {"converged": False}

    def flux(Jc):
        return psi_ext + (g.G @ (Jc[g.mask] * g.dA)).reshape(psi_ext.shape)

    for outer in range(max_outer):
        for it in range(inner):
            it_total += 1
            psi = flux(J)
            iz, ir = find_axis(g, psi, near)
            psi_a = float(psi[iz, ir])
            psi_b, reg = boundary(g, psi, iz, ir)
            span = psi_a - psi_b
            if span <= 0 or reg.sum() < 4:
                info["fail"] = "no closed region"
                return {"psi": psi, "J": J, "ok": False, "iters": it_total, **info}
            psiN = (psi_a - psi) / span
            if hasattr(profile, "fit") and (it % fit_every == 0):
                from scipy.interpolate import RectBivariateSpline
                sp = RectBivariateSpline(g.z, g.r, psi, kx=1, ky=1)
                profile.fit(g, lambda R, Z: (psi_a - sp(Z, R, grid=False)) / span)
            Jn = profile.current(g, psiN, reg, ip_a, span)
            r_free, z_free = _centroid(g, Jn)  # where the profile current wants to be
            Jn = _shift(g, Jn, (rc - r_free) if pin_r else 0.0, zc - z_free)
            tot = Jn.sum() * g.dA
            if tot <= 0:
                info["fail"] = "zero current"
                return {"psi": psi, "J": J, "ok": False, "iters": it_total, **info}
            Jn *= ip_a / tot
            dJ = np.abs(Jn - J).sum() / np.abs(Jn).sum()
            J = (1 - omega) * J + omega * Jn
            near = (float(g.r[ir]), float(g.z[iz]))
            if dJ < tol:
                break
        dr_, dz_ = (r_free - rc) if pin_r else 0.0, z_free - zc
        if verbose:
            print(f"outer {outer} rc {rc:.4f} zc {zc:+.4f} dr {dr_:+.5f} dz {dz_:+.5f} dJ {dJ:.2e} its {it_total}")
        if abs(dz_) < 2e-4 and abs(dr_) < 2e-4 and dJ < 10 * tol:
            info["converged"] = True
            break
        sz = sec_z.step(zc, dz_)
        sr = sec_r.step(rc, dr_) if pin_r else 0.0
        J = _shift(g, J, sr, sz)
        J *= ip_a / (J.sum() * g.dA)
        rc, zc = rc + sr, zc + sz
    psi = flux(J)
    iz, ir = find_axis(g, psi, near)
    psi_b, reg = boundary(g, psi, iz, ir)
    return {"psi": psi, "J": J, "ok": True, "axis": (g.r[ir], g.z[iz]), "psi_a": float(psi[iz, ir]),
            "psi_b": psi_b, "region": reg, "iters": it_total, "centroid": (rc, zc), **info}


# ----------------------------------------------------------------------------------------------
# GS operator on a grid (for the "how good is the coil model" test)
# ----------------------------------------------------------------------------------------------

def delta_star(g: Grid, psi: np.ndarray) -> np.ndarray:
    """Delta* psi = R d/dR (1/R dpsi/dR) + d2psi/dZ2, second-order finite differences, NaN on the grid edge."""
    out = np.full_like(psi, np.nan)
    R = g.RR[1:-1, 1:-1]
    d2r = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / g.dr ** 2
    d1r = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * g.dr)
    d2z = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / g.dz ** 2
    out[1:-1, 1:-1] = d2r - d1r / R + d2z
    return out
