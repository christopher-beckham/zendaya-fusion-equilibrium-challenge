"""Differentiable Grad-Shafranov layer of the pegs model (physics-embedded GS network). Any tokamak.

Hard structure (nothing here is learned):
    psi      = psi_ext + G J                         G = Green's function of uniformly filled grid cells
    psi_ext  = (g * I) @ M  [+ sum_m c_m h_m]        exact coil flux, optional per-coil gains g and a small
                                                      vacuum-harmonic correction (Delta* h_m = 0: vessel currents)
    J        = w(psi_N) [ (R / R0) P(psi_N) + (R0 / R) Q(psi_N) ]   inside the plasma; P ~ R0 p', Q ~ FF' / (mu0 R0)
    sum J dA = Ip                                     exactly (normalisation)

P and Q are 1-D profiles on a uniform psi_N grid (the profile network's output), so p' and FF' are flux
functions by construction. psi_N needs the axis and the boundary of the CURRENT iterate: both are recomputed at
every Picard step (inside / outside reassigned every cycle). Index searches (axis cell, bisection level, X-point
cell, ray bracket) are non-differentiable; the values read at them are differentiable (parabolic axis value,
quadratic saddle value, straight-through boundary flux, linear ray crossings), so gradients reach J and psi.

Machine geometry enters only through the Machine object (grid, coil matrix M, domain, cell Green's functions).
The domain is symmetrised in Z (intersection with its mirror) so the layer is exactly up-down equivariant.
Convention: physical flux, Ip > 0, magnetic axis = maximum.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

MU0 = 4e-7 * math.pi
N_HARM = 8
HARM_PARITY = (1, 1, -1, -1, 1, -1, 1, 1)     # Z-parity of each vacuum harmonic (see harmonics())


def _erode(mask: np.ndarray, cells: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(cells):
        e = m.copy()
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m


class GSLayer:
    def __init__(self, machine, device, n_rays: int = 128, sym="auto"):
        m = machine
        self.m = m
        self.dev = device
        f32 = dict(dtype=torch.float32, device=device)
        gR, gZ = np.asarray(m.grid_R, np.float64), np.asarray(m.grid_Z, np.float64)
        self.nZ, self.nR = len(gZ), len(gR)
        self.P = self.nZ * self.nR
        self.dR, self.dZ = float(gR[1] - gR[0]), float(gZ[1] - gZ[0])
        self.area = self.dR * self.dZ
        if sym == "auto":      # symmetrise (exact up-down equivariance) only on Z-symmetric grids
            sym = bool(np.allclose(gZ, -gZ[::-1], atol=1e-4))
        if sym and not np.allclose(gZ, -gZ[::-1], atol=1e-4):
            raise ValueError(f"{m.name}: grid not symmetric in Z, cannot symmetrise")
        self.sym = sym
        dom = m.domain.astype(bool)
        if sym:
            dom = dom & dom[::-1]
        keep = dom[m.cell_iz, m.cell_ir]
        self.iz = torch.tensor(m.cell_iz[keep], dtype=torch.long, device=device)
        self.ir = torch.tensor(m.cell_ir[keep], dtype=torch.long, device=device)
        self.flat = self.iz * self.nR + self.ir
        self.K = int(keep.sum())
        self.G = torch.tensor(np.asarray(m.G_cells)[keep], **f32)                 # (K, P)
        self.M = torch.tensor(m.M, **f32)                                        # (C, P)
        # mirror permutation of the cells (Z -> -Z)
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(m.cell_iz[keep], m.cell_ir[keep]))}
        self.cell_mirror = torch.tensor([pos[(self.nZ - 1 - int(a), int(b))] for a, b in zip(m.cell_iz[keep], m.cell_ir[keep])],
                                        dtype=torch.long, device=device) if sym else None
        self.R = torch.tensor(gR, **f32)
        self.Z = torch.tensor(gZ, **f32)
        ZZ, RR = torch.meshgrid(self.Z, self.R, indexing="ij")
        self.RR, self.ZZ = RR, ZZ
        self.dom_np = dom
        self.dom = torch.tensor(dom, device=device)
        ax = _erode(dom, 2)
        self.dom_axis = torch.tensor(ax, device=device)
        self.dom_in1 = torch.tensor(_erode(dom, 1), device=device)
        self.rim = torch.tensor(dom & ~_erode(dom, 1), device=device)
        self.Rk, self.Zk = self.R[self.ir], self.Z[self.iz]
        self.R0f, self.R1f = float(gR[0]), float(gR[-1])
        self.Z0f, self.Z1f = float(gZ[0]), float(gZ[-1])
        th = torch.arange(n_rays, **f32) * (2 * math.pi / n_rays)
        self.theta = th
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        self.ray_s = torch.linspace(0.0, 1.0, 200, **f32)
        self.ray_len = 0.5 * math.hypot(self.R1f - self.R0f, self.Z1f - self.Z0f)
        Rd, Zd = RR[self.dom], ZZ[self.dom]
        self.centre = torch.stack([0.5 * (Rd.min() + Rd.max()), torch.zeros((), **f32)])
        self.halfwidth = float(0.5 * (Rd.max() - Rd.min()))
        self.tf_turns = float(m.tf_turns)
        self.bracket = False                     # narrow the boundary bisection around the previous Picard iterate

    # ------------------------------------------------------------------ machine-fixed bases (no first-pass dependence)
    def harmonics_fixed(self):
        """Edge-normalised harmonics about the machine centre (domain centre, Z = 0, radius half the domain half-width):
        (1, N_HARM, P), constant for a machine. The external-field correction expanded here does not inherit the
        first-pass position jitter."""
        if not hasattr(self, "_harm_fixed"):
            R0 = self.centre[0:1]; Z0 = torch.zeros(1, device=self.dev); a = torch.full((1,), 0.5 * self.halfwidth, device=self.dev)
            self._harm_fixed = self.harmonics_edge(R0, Z0, a).detach()
        return self._harm_fixed

    def vac_picture_matrix(self, n=8):
        """Coil-flux sampler: (C, n*n) columns of M at an n x n grid of nodes spanning the plasma domain's bounding box.
        (I @ this) / (mu0 Ip R_centre) is an exact, dimensionless picture of the coil field in machine coordinates."""
        key = f"_vacpic_{n}"
        if not hasattr(self, key):
            Rd, Zd = self.RR[self.dom], self.ZZ[self.dom]
            rs = torch.linspace(float(Rd.min()), float(Rd.max()), n, device=self.dev)
            zs = torch.linspace(float(Zd.min()), float(Zd.max()), n, device=self.dev)
            ir = ((rs - self.R0f) / self.dR).round().long().clamp(0, self.nR - 1)
            iz = ((zs - self.Z0f) / self.dZ).round().long().clamp(0, self.nZ - 1)
            idx = (iz[:, None] * self.nR + ir[None, :]).reshape(-1)
            setattr(self, key, self.M[:, idx].contiguous())
        return getattr(self, key)

    # ------------------------------------------------------------------ basic maps
    def vacuum(self, I, gain=None):
        I = torch.nan_to_num(I)
        return (I * gain if gain is not None else I) @ self.M

    def plasma(self, Jc):
        return Jc @ self.G

    def scatter(self, Jc):
        img = Jc.new_zeros(Jc.shape[0], self.P)
        img[:, self.flat] = Jc
        return img.view(-1, self.nZ, self.nR)

    def mirror_cells(self, Jc):
        return Jc[:, self.cell_mirror]

    def harmonics_edge(self, R0, Z0, a, n=24):
        """Each harmonic scaled by its largest magnitude on the circle of radius a about (R0, Z0) (no grad): a unit
        coefficient then means a flux of mu0 Ip R0 at the plasma edge, whatever the machine size. (B, N_HARM, P)"""
        H = self.harmonics(R0, Z0)
        with torch.no_grad():
            t = torch.arange(n, device=self.dev, dtype=torch.float32) * (2 * math.pi / n)
            Rr = R0[:, None] + a[:, None] * torch.cos(t)[None]
            Zr = Z0[:, None] + a[:, None] * torch.sin(t)[None]
            Hr = self.harmonics(R0, Z0, Rr, Zr)
            sc = Hr.abs().amax(2).clamp(min=1e-6)
            sc[:, 0] = 1.0
        return H / sc[:, :, None]

    def harmonics(self, R0, Z0, RR=None, ZZ=None):
        """Vacuum harmonics about (R0, Z0): (B, N_HARM, P), x = R / R0, y = (Z - Z0) / R0, dimensionless:
        constant (flux offset), x^2 (vertical field), y (radial field), x^2 y, x^4 - 4 x^2 y^2 (quadrupole),
        3 x^4 y - 4 x^2 y^3, x^6 - 12 x^4 y^2 + 8 x^2 y^4 (hexapole), y^2 - x^2 ln x. All satisfy Delta* h = 0;
        constants are subtracted so every term vanishes near the centre. Parities in HARM_PARITY."""
        RR = self.RR.reshape(1, -1) if RR is None else RR
        ZZ = self.ZZ.reshape(1, -1) if ZZ is None else ZZ
        x = RR / R0[:, None]
        y = (ZZ - Z0[:, None]) / R0[:, None]
        x2, y2 = x * x, y * y
        return torch.stack([torch.ones_like(x).expand_as(x2), x2 - 1, y, x2 * y, x2 * x2 - 4 * x2 * y2 - 1, 3 * x2 * x2 * y - 4 * x2 * y2 * y,
                            x2 * x2 * x2 - 12 * x2 * x2 * y2 + 8 * x2 * y2 * y2 - 1, y2 - x2 * torch.log(x)], 1)

    def _grid(self, pts):
        x = (pts[..., 0] - self.R0f) / (self.R1f - self.R0f) * 2 - 1
        y = (pts[..., 1] - self.Z0f) / (self.Z1f - self.Z0f) * 2 - 1
        return torch.stack([x, y], -1)

    def sample(self, maps, pts):
        """maps (B, nZ, nR) at pts (B, ..., 2) -> (B, ...), bilinear (border padding)."""
        shp = pts.shape[1:-1]
        g = self._grid(pts.reshape(pts.shape[0], 1, -1, 2))
        v = F.grid_sample(maps[:, None], g, mode="bilinear", align_corners=True, padding_mode="border")[:, 0, 0]
        return v.reshape(pts.shape[0], *shp)

    def grad(self, phi):
        gZ, gR = torch.gradient(phi, spacing=(self.dZ, self.dR), dim=(1, 2))
        return gR, gZ

    # ------------------------------------------------------------------ axis
    def axis(self, phi, near=None):
        """Interior local maximum nearest to `near` (B, 2) (the current centroid), else the strongest.
        Returns iz, ir (no grad) and the parabolic sub-cell position and value (differentiable)."""
        B = phi.shape[0]
        with torch.no_grad():
            mp = F.max_pool2d(phi[:, None], 3, stride=1, padding=1)[:, 0]
            lm = (phi >= mp) & self.dom_axis
            if near is None:
                sc = torch.where(lm, phi, torch.full_like(phi, -float("inf"))).view(B, -1)
            else:
                d2 = (self.RR[None] - near[:, 0, None, None]) ** 2 + (self.ZZ[None] - near[:, 1, None, None]) ** 2
                sc = torch.where(lm, -d2, torch.full_like(d2, -float("inf"))).view(B, -1)
            best, idx = sc.max(1)
            none = ~torch.isfinite(best)
            # speed-up: fallback computed for the whole batch and selected (no CPU-GPU wait; same values)
            g = torch.where(self.dom_axis, phi, torch.full_like(phi, -float("inf"))).view(B, -1)
            idx = torch.where(none, g.argmax(1), idx)
            iz = (idx // self.nR).clamp(1, self.nZ - 2)
            ir = (idx % self.nR).clamp(1, self.nR - 2)
        ar = torch.arange(B, device=self.dev)
        b = phi[ar, iz, ir]
        az, cz = phi[ar, iz - 1, ir], phi[ar, iz + 1, ir]
        al, cr = phi[ar, iz, ir - 1], phi[ar, iz, ir + 1]
        dzd, drd = az - 2 * b + cz, al - 2 * b + cr
        okz, okr = dzd < -1e-12, drd < -1e-12
        dzd_s = torch.where(okz, dzd, -torch.ones_like(dzd)); drd_s = torch.where(okr, drd, -torch.ones_like(drd))
        dz = torch.where(okz, (0.5 * (az - cz) / dzd_s).clamp(-1, 1), torch.zeros_like(b))
        dr = torch.where(okr, (0.5 * (al - cr) / drd_s).clamp(-1, 1), torch.zeros_like(b))
        val = b - torch.where(okz, 0.125 * (az - cz) ** 2 / dzd_s, torch.zeros_like(b)) \
                - torch.where(okr, 0.125 * (al - cr) ** 2 / drd_s, torch.zeros_like(b))
        val = torch.maximum(val, b)
        return iz, ir, val, self.R[ir] + dr * self.dR, self.Z[iz] + dz * self.dZ

    # ------------------------------------------------------------------ saddles (X-points)
    def saddles(self, phi, k: int = 4):
        """Up to k saddle candidates per frame inside the domain: cells where |grad psi|^2 is a 3x3 local minimum
        and the Hessian determinant is negative. Quadratic refinement gives a differentiable position and value.
        Returns val (B, k), R (B, k), Z (B, k), ok (B, k)."""
        B = phi.shape[0]
        pR = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * self.dR)
        pZ = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * self.dZ)
        pRR = (phi[:, :, 2:] - 2 * phi[:, :, 1:-1] + phi[:, :, :-2]) / self.dR ** 2
        pZZ = (phi[:, 2:, :] - 2 * phi[:, 1:-1, :] + phi[:, :-2, :]) / self.dZ ** 2
        pR, pRR = pR[:, 1:-1, :], pRR[:, 1:-1, :]
        pZ, pZZ = pZ[:, :, 1:-1], pZZ[:, :, 1:-1]
        pRZ = (phi[:, 2:, 2:] - phi[:, 2:, :-2] - phi[:, :-2, 2:] + phi[:, :-2, :-2]) / (4 * self.dR * self.dZ)
        with torch.no_grad():
            g2 = (pR ** 2 + pZ ** 2)
            mn = -F.max_pool2d(-g2[:, None], 3, stride=1, padding=1)[:, 0]
            det = pRR * pZZ - pRZ ** 2
            cand = (g2 <= mn) & (det < 0) & self.dom_in1[1:-1, 1:-1]
            score = torch.where(cand, g2, torch.full_like(g2, float("inf"))).view(B, -1)
            sv, si = torch.topk(score, k, dim=1, largest=False)
            ok = torch.isfinite(sv)
            iz, ir = si // (self.nR - 2), si % (self.nR - 2)
        take = lambda a: a.reshape(B, -1).gather(1, si)
        gR_, gZ_, hRR, hZZ, hRZ = take(pR), take(pZ), take(pRR), take(pZZ), take(pRZ)
        det_ = hRR * hZZ - hRZ ** 2
        det_ = torch.where(det_ < -1e-12, det_, -torch.ones_like(det_))
        dRx = -(hZZ * gR_ - hRZ * gZ_) / det_
        dZx = -(-hRZ * gR_ + hRR * gZ_) / det_
        dRx = dRx.clamp(-self.dR, self.dR); dZx = dZx.clamp(-self.dZ, self.dZ)
        v0 = phi[:, 1:-1, 1:-1].reshape(B, -1).gather(1, si)
        val = v0 + gR_ * dRx + gZ_ * dZx + 0.5 * (hRR * dRx ** 2 + 2 * hRZ * dRx * dZx + hZZ * dZx ** 2)
        Rx = self.R[1:-1][ir] + dRx
        Zx = self.Z[1:-1][iz] + dZx
        return val, Rx, Zx, ok

    # ------------------------------------------------------------------ boundary
    @torch.no_grad()
    def flood(self, ok, seed_iz, seed_ir, n_iter=128):
        B = ok.shape[0]
        reg = torch.zeros_like(ok)
        reg[torch.arange(B, device=self.dev), seed_iz, seed_ir] = True
        reg &= ok
        for i in range(n_iter):
            new = (F.max_pool2d(reg[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0) & ok
            if i % 8 == 7 and torch.equal(new, reg):
                break
            reg = new
        return reg

    @torch.no_grad()
    def reach_level(self, phi, seed_iz, seed_ir, n_iter=128):
        """L[c] = highest level v such that c is in flood((phi > v) & dom, seed) after the same n_iter dilations: the
        best path minimum of phi from the seed to c. flood((phi > v) & self.dom, seed) == (L > v) for every v, so one
        propagation answers every bisection step (speed-up; results identical to the per-level floods)."""
        B = phi.shape[0]
        ph = torch.where(self.dom & ~torch.isnan(phi), phi, torch.full_like(phi, -float("inf")))[:, None]
        L = torch.full_like(ph, -float("inf")).view(B, -1)
        idx = (seed_iz * self.nR + seed_ir)[:, None]
        L.scatter_(1, idx, ph.view(B, -1).gather(1, idx))
        L = L.view_as(ph)
        static = getattr(self, "_static_iters", False)          # inside a CUDA graph: all n_iter steps, no convergence check
        for i in range(n_iter):
            new = torch.minimum(ph, F.max_pool2d(L, 3, stride=1, padding=1))
            if not static and i % 8 == 7 and torch.equal(new, L):
                break
            L = new
        return L[:, 0]

    def boundary(self, phi, aiz, air, aval, n_bisect=18, prev=None):
        """Boundary flux = lowest level whose axis-connected region {psi > level} stays off the domain rim (bisection +
        flood fill, the separatrix or limiter level). Returns psi_b (straight-through differentiable: forward value
        = bisection level, gradient through the nearest saddle value or the rim maximum), the hard plasma mask and
        a flag for X-point (True) or limiter boundaries."""
        B = phi.shape[0]
        with torch.no_grad():
            lo = torch.where(self.dom, phi, torch.full_like(phi, float("inf"))).view(B, -1).min(1).values
            hi = aval.detach() - 1e-6 * (aval.detach() - lo).clamp(min=1e-9)
            if prev is not None:
                # bracket around the previous iterate's boundary (+-15 % of its flux swing), checked below
                sw0 = (aval.detach() - prev).clamp(min=1e-9)
                lo2, hi2 = torch.maximum(lo, prev - 0.15 * sw0), torch.minimum(hi, prev + 0.15 * sw0)
                t_lo = (self.flood((phi > lo2[:, None, None]) & self.dom, aiz, air) & self.rim).view(B, -1).any(1)
                t_hi = (self.flood((phi > hi2[:, None, None]) & self.dom, aiz, air) & self.rim).view(B, -1).any(1)
                good = t_lo & ~t_hi
                lo, hi = torch.where(good, lo2, lo), torch.where(good, hi2, hi)
                n_bisect = 10 if bool(good.all()) else n_bisect
            # speed-up: the region at level v is {L > v} and it touches the rim iff max(L on the rim) > v
            L = self.reach_level(phi, aiz, air)
            L_rim = torch.where(self.rim, L, torch.full_like(L, -float("inf"))).view(B, -1).max(1).values
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                touch = L_rim > mid
                lo = torch.where(touch, mid, lo)
                hi = torch.where(touch, hi, mid)
            psi_b0 = hi
            plasma = L > psi_b0[:, None, None]
        sv, sR, sZ, sok = self.saddles(phi)
        sw = (aval.detach() - psi_b0).clamp(min=1e-9)
        with torch.no_grad():
            gap = torch.where(sok, (sv.detach() - psi_b0[:, None]).abs() / sw[:, None], torch.full_like(sv, float("inf")))
            gmin, gi = gap.min(1)
            is_x = gmin < 0.02
            near = F.max_pool2d(plasma[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0
            rim_near = self.rim[None] & near
        rim_val, rim_i = torch.where(rim_near, phi, torch.full_like(phi, -float("inf"))).view(B, -1).max(1)
        rim_val = torch.where(torch.isfinite(rim_val), rim_val, psi_b0)
        self._lim_RZ = (self.RR.reshape(-1)[rim_i], self.ZZ.reshape(-1)[rim_i])     # limiter contact (if limited)
        sel = torch.where(is_x, sv.gather(1, gi[:, None])[:, 0], rim_val)
        psi_b = psi_b0 + (sel - sel.detach())
        return psi_b, plasma, is_x, (sR.gather(1, gi[:, None])[:, 0], sZ.gather(1, gi[:, None])[:, 0])

    # ------------------------------------------------------------------ profiles -> current
    @staticmethod
    def interp1(prof, x):
        """prof (B, n) on a uniform grid over [0, 1]; x (B, K) -> (B, K), linear, clamped."""
        n = prof.shape[1]
        u = x.clamp(0, 1) * (n - 1)
        i0 = u.detach().floor().clamp(0, n - 2).long()
        t = u - i0
        return prof.gather(1, i0) * (1 - t) + prof.gather(1, i0 + 1) * t

    def current(self, phi, aval, psi_b, plasma, ip, R0, P, Q, tau=0.03):
        """Cell currents (A) from the profiles, normalised to Ip; w(psi_N) is a linear ramp of width 2 tau about
        psi_N = 1, applied on the plasma region dilated by one cell (so the boundary can move with gradient)."""
        B = phi.shape[0]
        phi_k = phi.reshape(B, -1)[:, self.flat]
        with torch.no_grad():
            reg = (F.max_pool2d(plasma[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0) & self.dom
            reg_k = reg.reshape(B, -1)[:, self.flat]
        sw = (aval - psi_b).clamp(min=1e-9)
        x = (aval[:, None] - phi_k) / sw[:, None]
        w = ((1 + tau - x) / (2 * tau)).clamp(0, 1) * reg_k
        Rn = self.Rk[None] / R0[:, None]
        J = w * (Rn * self.interp1(P, x) + self.interp1(Q, x) / Rn)
        tot = J.sum(1, keepdim=True)
        tot = torch.where(tot.abs() > 1e-12, tot, torch.ones_like(tot))
        scale = ip[:, None] / tot
        return J * scale, x, scale[:, 0]

    # ------------------------------------------------------------------ equilibrium iteration
    def picard(self, psi_ext, J0, ip, R0, P, Q, n_steps=3, omega=1.0, vstab=False, z_pin=None):
        """Unrolled Picard steps from the cell currents J0 (B, K). Returns final psi (B, P), the currents that made
        it, the self-consistent currents of the final psi (for the GS-consistency loss) and the final geometry.
        vstab: after every step, rigidly shift the current to zero net vertical force from the external field (the
        shift is found without gradient; the shifted currents stay differentiable)."""
        J = J0
        prev = None
        if vstab:
            B = psi_ext.shape[0]
            with torch.no_grad():
                _, gZv = self.grad(psi_ext.detach().view(B, self.nZ, self.nR))
                BRv = -gZv / self.RR
        for _ in range(n_steps):
            psi = psi_ext + self.plasma(J)
            st = self.state(psi, J, ip, prev=prev if self.bracket else None)
            prev = st["psi_b"].detach()
            Jn, _, _ = self.current(st["phi"], st["psi_axis"], st["psi_b"], st["plasma"], ip, R0, P, Q)
            J = (1 - omega) * J + omega * Jn
            if vstab and z_pin is not None:
                # m2: pinned height (from solve_vc's root search): shift the centroid to z_pin instead of the per-step root
                dz = z_pin - self.centroid(J)[:, 1]
                J = self.shift_cells(J, torch.zeros_like(dz), dz, centre=self.centroid(J))
                J = J * (ip / J.sum(1).where(J.sum(1).abs() > 1e-6, torch.ones_like(ip)))[:, None]
            elif vstab:
                # m2: the vertical force on a current element is 2 pi R J B_R; vstab_rw weights the root search by R
                Jw = J.detach() * self.Rk[None] if getattr(self, 'vstab_rw', False) else J.detach()
                dz, none = self.fz_root(Jw, BRv, return_none=True)
                if getattr(self, 'vstab_safe', False):
                    # m2: no force zero inside the search window = the iteration is running away; do not follow it,
                    # and never move more than two cells per step
                    dz = torch.where(none, torch.zeros_like(dz), dz).clamp(-2 * self.dZ, 2 * self.dZ)
                J = self.shift_cells(J, torch.zeros_like(dz), dz, centre=self.centroid(J))
                J = J * (ip / J.sum(1).where(J.sum(1).abs() > 1e-6, torch.ones_like(ip)))[:, None]
        psi = psi_ext + self.plasma(J)
        st = self.state(psi, J, ip, prev=prev if self.bracket else None)
        Jn, x, scale = self.current(st["phi"], st["psi_axis"], st["psi_b"], st["plasma"], ip, R0, P, Q)
        st.update(psi=psi, J=J, J_next=Jn, x_cells=x, scale=scale)
        return st

    # ------------------------------------------------------------------ speed-up: CUDA-graphed Picard (training batches)
    _ST_KEYS = ("aiz", "air", "psi_axis", "R_axis", "Z_axis", "psi_b", "plasma", "is_x", "xpt_R", "xpt_Z", "lim_R", "lim_Z",
                "psi", "J", "J_next", "x_cells", "scale")

    def picard_graphed(self, psi_ext, J0, ip, R0, P, Q, n_steps=3, omega=1.0):
        """Same as picard(psi_ext, J0, ip, R0, P, Q, n_steps, omega) (no vstab / z_pin), forward and backward replayed from a
        CUDA graph captured on the first call for these shapes, so the CPU issues one launch instead of thousands."""
        args = (psi_ext, J0, ip, R0, P, Q, omega)
        key = (n_steps,) + tuple((tuple(t.shape), t.dtype, t.requires_grad) for t in args)
        graphs = self.__dict__.setdefault("_graphs", {})
        f = graphs.get(key)
        if f is None:
            def fn(psi_ext, J0, ip, R0, P, Q, omega):
                st = self.picard(psi_ext, J0, ip, R0, P, Q, n_steps=n_steps, omega=omega)
                return tuple(st[k] for k in self._ST_KEYS)
            sample = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in args)
            self._static_iters = True
            try:
                f = graphs[key] = torch.cuda.make_graphed_callables(fn, sample)
            finally:
                self._static_iters = False
        st = dict(zip(self._ST_KEYS, f(*args)))
        st["phi"] = st["psi"].view(st["psi"].shape[0], self.nZ, self.nR)
        return st

    def centroid(self, J):
        w = J.detach().clamp(min=0)
        tot = w.sum(1).clamp(min=1e-9)
        return torch.stack([(w * self.Rk[None]).sum(1) / tot, (w * self.Zk[None]).sum(1) / tot], 1)

    def state(self, psi, J, ip, prev=None):
        B = psi.shape[0]
        phi = psi.view(B, self.nZ, self.nR)
        iz, ir, aval, Rax, Zax = self.axis(phi, self.centroid(J))
        psi_b, plasma, is_x, xpt = self.boundary(phi, iz, ir, aval, prev=prev)
        return {"phi": phi, "aiz": iz, "air": ir, "psi_axis": aval, "R_axis": Rax, "Z_axis": Zax, "psi_b": psi_b,
                "plasma": plasma, "is_x": is_x, "xpt_R": xpt[0], "xpt_Z": xpt[1], "lim_R": self._lim_RZ[0], "lim_Z": self._lim_RZ[1]}

    # ------------------------------------------------------------------ vertical stabilisation (no grad, first pass)
    @torch.no_grad()
    def fz_root(self, J, BRv, n_half=8, return_none=False):
        B = J.shape[0]
        s = torch.arange(-n_half, n_half + 1, device=self.dev, dtype=torch.float32) * (0.5 * self.dZ)
        pts = torch.stack([self.Rk[None, :].expand(len(s), -1), self.Zk[None, :] + s[:, None]], -1)
        br = self.sample(BRv, pts[None].expand(B, -1, -1, -1))
        Fs = (br * J[:, None, :]).sum(-1)
        sg = torch.sign(Fs)
        change = (sg[:, 1:] * sg[:, :-1]) <= 0
        mid = 0.5 * (s[1:] + s[:-1])
        dist = torch.where(change, mid.abs()[None].expand(B, -1), torch.full_like(Fs[:, 1:], float("inf")))
        j = dist.argmin(1)
        f0, f1 = Fs.gather(1, j[:, None])[:, 0], Fs.gather(1, (j + 1)[:, None])[:, 0]
        den = f0 - f1
        t = (f0 / torch.where(den.abs() > 1e-12, den, torch.full_like(den, 1e-12))).clamp(0, 1)
        root = s[j] + t * (s[j + 1] - s[j])
        none = ~torch.isfinite(dist.min(1).values)
        out = torch.where(none, s[Fs.abs().argmin(1)], root)
        return (out, none) if return_none else out

    def shift_cells(self, J, dR, dZ, sR=None, sZ=None, centre=None):
        """Move / stretch a cell-current distribution: J'(x) = J(c + (x - c - d) / s), bilinear; differentiable in
        dR, dZ (B,) metres and the log-stretches sR, sZ (B,). Renormalisation is left to the caller."""
        B = J.shape[0]
        img = self.scatter(J)[:, None]
        c = centre if centre is not None else self.centroid(J)
        sR = torch.zeros_like(dR) if sR is None else sR
        sZ = torch.zeros_like(dZ) if sZ is None else sZ
        Rq = c[:, 0, None, None] + (self.RR[None] - c[:, 0, None, None] - dR[:, None, None]) * torch.exp(-sR)[:, None, None]
        Zq = c[:, 1, None, None] + (self.ZZ[None] - c[:, 1, None, None] - dZ[:, None, None]) * torch.exp(-sZ)[:, None, None]
        g = self._grid(torch.stack([Rq, Zq], -1))
        out = F.grid_sample(img, g, mode="bilinear", align_corners=True, padding_mode="zeros")[:, 0]
        return out.reshape(B, -1)[:, self.flat]

    @torch.no_grad()
    def solve(self, psi_ext, ip, P, Q, R0=None, n_iter=40, omega=0.5, J0=None, vertical=True):
        """Converged free-boundary solve (no grad) with vertical stabilisation by a rigid shift of the current to zero
        net vertical force from the external field. Used for the first pass and for evaluation-time polishing."""
        B = psi_ext.shape[0]
        R0 = self.centre[0].expand(B) if R0 is None else R0
        _, gZv = self.grad(psi_ext.view(B, self.nZ, self.nR))
        BRv = -gZv / self.RR
        if J0 is None:
            c = self.centre[None].expand(B, 2)
            r0 = 0.45 * self.halfwidth
            d2 = (self.Rk[None] - c[:, :1]) ** 2 + (self.Zk[None] - c[:, 1:]) ** 2
            J0 = (d2 < r0 ** 2).float()
            J0 = J0 / J0.sum(1, keepdim=True).clamp(min=1) * ip[:, None]
        J = J0
        best_J, best_res = J.clone(), torch.full((B,), float("inf"), device=self.dev)
        res = best_res.clone()
        for it in range(n_iter):
            psi = psi_ext + self.plasma(J)
            st = self.state(psi, J, ip)
            Jn, _, _ = self.current(st["phi"], st["psi_axis"], st["psi_b"], st["plasma"], ip, R0, P, Q)
            sw = (st["psi_axis"] - st["psi_b"]).clamp(min=1e-9)
            res = (self.plasma(Jn - J)).abs().amax(1) / sw
            better = res < best_res
            best_res = torch.where(better, res, best_res)
            best_J = torch.where(better[:, None], J, best_J)
            J = (1 - omega) * J + omega * Jn
            if vertical:
                dz = self.fz_root(J * self.Rk[None] if getattr(self, 'vstab_rw', False) else J, BRv)
                J = self.shift_cells(J, torch.zeros_like(dz), dz, centre=self.centroid(J))
                J = J * (ip / J.sum(1).where(J.sum(1).abs() > 1e-9, torch.ones_like(ip)))[:, None]
        use_last = res <= 2 * best_res
        J = torch.where(use_last[:, None], J, best_J)
        res = torch.where(use_last, res, best_res)
        bad = ~torch.isfinite(J).all(1) | ~torch.isfinite(res)
        J = torch.where(bad[:, None], J0, J)
        return J, res

    # ------------------------------------------------------------------ m2: vertically controlled converged solve
    @torch.no_grad()
    def solve_vc(self, psi_ext, ip, P, Q, R0=None, J0=None, n_inner=10, n_outer=8, n_final=20, omega=0.5, width=0.2):
        """Converged free-boundary solve with the vertical degree of freedom handled by an outer root search instead of a
        per-step rigid shift: the current centroid is PINNED at a trial height z while the Picard iteration converges (the
        vertical instability is removed from the fixed-point map), and z is found by bisection on the net vertical force
        F(z) = sum_k R_k J_k B_R,ext(R_k, Z_k) of the pinned equilibrium (zero at a true equilibrium, whatever its stability).
        Bracket z0 +- width (z0 = centroid of J0), widened once if F has no sign change. Returns J, |F| at the root, and
        ok (sign change found)."""
        B = psi_ext.shape[0]
        R0 = self.centre[0].expand(B) if R0 is None else R0
        _, gZv = self.grad(psi_ext.view(B, self.nZ, self.nR))
        BRk = (-gZv / self.RR).reshape(B, -1)[:, self.flat] * self.Rk[None]          # R B_R,ext at the cells
        if J0 is None:
            c = self.centre[None].expand(B, 2)
            r0 = 0.45 * self.halfwidth
            d2 = (self.Rk[None] - c[:, :1]) ** 2 + (self.Zk[None] - c[:, 1:]) ** 2
            J0 = (d2 < r0 ** 2).float()
            J0 = J0 / J0.sum(1, keepdim=True).clamp(min=1) * ip[:, None]
        zero = torch.zeros(B, device=self.dev)
        def renorm(J):
            return J * (ip / J.sum(1).where(J.sum(1).abs() > 1e-9, torch.ones_like(ip)))[:, None]
        def pin(J, z):
            dz = z - self.centroid(J)[:, 1]
            return renorm(self.shift_cells(J, zero, dz, centre=self.centroid(J)))
        def pinned(J, z, n):
            J = pin(J, z)
            for _ in range(n):
                psi = psi_ext + self.plasma(J)
                st = self.state(psi, J, ip)
                Jn, _, _ = self.current(st["phi"], st["psi_axis"], st["psi_b"], st["plasma"], ip, R0, P, Q)
                J = pin((1 - omega) * J + omega * Jn, z)
            return J, (BRk * J).sum(1)
        zc = self.centroid(J0)[:, 1]
        lo, hi = zc - width, zc + width
        Jlo, Flo = pinned(J0, lo, n_inner)
        Jhi, Fhi = pinned(J0, hi, n_inner)
        nosc = (Flo * Fhi) > 0
        if nosc.any():                                              # widen the bracket once for those frames
            lo2, hi2 = zc - 2 * width, zc + 2 * width
            Jlo2, Flo2 = pinned(J0, lo2, n_inner)
            Jhi2, Fhi2 = pinned(J0, hi2, n_inner)
            lo = torch.where(nosc, lo2, lo); hi = torch.where(nosc, hi2, hi)
            Flo = torch.where(nosc, Flo2, Flo); Fhi = torch.where(nosc, Fhi2, Fhi)
            Jlo = torch.where(nosc[:, None], Jlo2, Jlo); Jhi = torch.where(nosc[:, None], Jhi2, Jhi)
        ok = (Flo * Fhi) <= 0
        for _ in range(n_outer):
            mid = 0.5 * (lo + hi)
            warm = torch.where((Flo.abs() <= Fhi.abs())[:, None], Jlo, Jhi)
            Jm, Fm = pinned(warm, mid, n_inner)
            same = torch.sign(Fm) == torch.sign(Flo)                 # root lies in [mid, hi]
            lo = torch.where(same, mid, lo); Flo = torch.where(same, Fm, Flo); Jlo = torch.where(same[:, None], Jm, Jlo)
            hi = torch.where(same, hi, mid); Fhi = torch.where(same, Fhi, Fm); Jhi = torch.where(same[:, None], Jhi, Jm)
        z = 0.5 * (lo + hi)
        # frames without a sign change: take the end with the smaller |F| (no better information)
        z = torch.where(ok, z, torch.where(Flo.abs() <= Fhi.abs(), lo, hi))
        warm = torch.where((Flo.abs() <= Fhi.abs())[:, None], Jlo, Jhi)
        J, Fz = pinned(warm, z, n_final)
        bad = ~torch.isfinite(J).all(1)
        J = torch.where(bad[:, None], J0, J)
        return J, Fz.abs(), ok & ~bad

    # ------------------------------------------------------------------ contours and functionals
    def contour(self, phi, Rax, Zax, level):
        """First crossing of psi = level along rays from the axis: (B, n_rays, 2), differentiable in phi, level and
        the axis position; ok (B, n_rays)."""
        B = phi.shape[0]
        L = self.ray_len * self.ray_s
        with torch.no_grad():
            pts = torch.stack([Rax.detach()[:, None, None] + self.cos[None, :, None] * L[None, None, :],
                               Zax.detach()[:, None, None] + self.sin[None, :, None] * L[None, None, :]], -1)
            v = self.sample(phi, pts) - level.detach()[:, None, None]
            inb = (pts[..., 0] > self.R0f) & (pts[..., 0] < self.R1f) & (pts[..., 1] > self.Z0f) & (pts[..., 1] < self.Z1f)
            neg = (v <= 0) | ~inb
            neg[..., 0] = False
            ok = neg.any(-1)
            j = torch.where(ok, neg.float().argmax(-1), torch.full(neg.shape[:2], neg.shape[-1] - 1, device=self.dev)).clamp(min=1)
        L0, L1 = L[j - 1], L[j]
        p0 = torch.stack([Rax[:, None] + self.cos[None] * L0, Zax[:, None] + self.sin[None] * L0], -1)
        p1 = torch.stack([Rax[:, None] + self.cos[None] * L1, Zax[:, None] + self.sin[None] * L1], -1)
        v0 = self.sample(phi, p0) - level[:, None]
        v1 = self.sample(phi, p1) - level[:, None]
        den = v0 - v1
        t = (v0 / torch.where(den.abs() > 1e-12, den, torch.full_like(den, 1e-12))).clamp(0, 1)
        return p0 + t[..., None] * (p1 - p0), ok

    @staticmethod
    def shape(poly):
        """Scorer definitions on a polygon (B, N, 2): kappa, tri_top, tri_bot, volume (+ R_geo, a, Z_geo, lp, area)."""
        R, Z = poly[..., 0], poly[..., 1]
        R2, Z2 = torch.roll(R, -1, 1), torch.roll(Z, -1, 1)
        lp = torch.hypot(R2 - R, Z2 - Z).sum(1)
        cross = R * Z2 - R2 * Z
        area = 0.5 * cross.sum(1)
        vol = 2 * math.pi * ((R + R2) * cross).sum(1).abs() / 6.0
        Rmin, Rmax = R.min(1).values, R.max(1).values
        Zmax, iu = Z.max(1)
        Zmin, il = Z.min(1)
        a = 0.5 * (Rmax - Rmin)
        Rg = 0.5 * (Rmax + Rmin)
        a_s = a.clamp(min=1e-4)
        return {"kappa": (Zmax - Zmin) / (2 * a_s), "tri_top": (Rg - R.gather(1, iu[:, None])[:, 0]) / a_s,
                "tri_bot": (Rg - R.gather(1, il[:, None])[:, 0]) / a_s, "volume": vol, "R_geo": Rg, "a": a,
                "Z_geo": 0.5 * (Zmax + Zmin), "lp": lp, "area": area.abs()}

    def bp(self, phi):
        gR, gZ = self.grad(phi)
        return torch.sqrt(gR ** 2 + gZ ** 2 + 1e-20) / self.RR

    def li(self, phi, plasma_w, poly):
        """Scorer li(2): <Bp^2>_V / (mu0 Ip / lp)^2 with mu0 Ip = loop integral of Bp dl on the contour (Ampere).
        plasma_w (B, nZ, nR) weights in [0, 1]. Also returns the Ampere loop integral."""
        Bp = self.bp(phi)
        dV = self.RR[None] * plasma_w
        Bp2V = (Bp ** 2 * dV).sum((1, 2)) / dV.sum((1, 2)).clamp(min=1e-12)
        bc = self.sample(Bp, poly)
        seg = torch.linalg.norm(torch.roll(poly, -1, 1) - poly, dim=-1)
        Gl = (0.5 * (bc + torch.roll(bc, -1, 1)) * seg).sum(1)
        lp = seg.sum(1)
        return Bp2V / (Gl / lp).clamp(min=1e-12) ** 2, Gl

    def q_at(self, phi, Rax, Zax, psi_ax, psi_b, Fpol, level=0.95):
        """q on the psi_N = level surface: q = F / (2 pi) * loop integral dl / (R |grad psi|)."""
        lev = psi_ax - level * (psi_ax - psi_b)
        poly, ok = self.contour(phi, Rax, Zax, lev)
        mid = 0.5 * (poly + torch.roll(poly, -1, 1))
        dl = torch.linalg.norm(torch.roll(poly, -1, 1) - poly, dim=-1)
        gR, gZ = self.grad(phi)
        g = torch.sqrt(self.sample(gR, mid) ** 2 + self.sample(gZ, mid) ** 2 + 1e-12)
        return Fpol / (2 * math.pi) * (dl / (mid[..., 0] * g)).sum(1), ok.all(1)

    def psin_at(self, phi, psi_ax, psi_b, pts):
        """psi_N at arbitrary points (B, N, 2)."""
        v = self.sample(phi, torch.nan_to_num(pts, nan=0.0))
        return (psi_ax[:, None] - v) / (psi_ax - psi_b).clamp(min=1e-9)[:, None]
