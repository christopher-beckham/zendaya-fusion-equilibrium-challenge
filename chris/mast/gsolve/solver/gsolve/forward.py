"""Free-boundary Grad-Shafranov solver, forward mode: batched, torch, CPU.

Given coil currents, plasma current Ip and the pressure and current profiles, it finds the equilibrium.
The plasma edge, X-point and magnetic axis come out of the solution; nothing is fitted to a target map.

Convention: psi is the poloidal flux per radian (Wb/rad) with Ip > 0, so the magnetic axis is a MAXIMUM.
psi_N = (psi_ax - psi) / (psi_ax - psi_b) is 0 on the axis and 1 on the plasma edge (last closed surface).
Grad-Shafranov current density: J = R p'(psi) + FF'(psi) / (mu0 R), and psi = psi_ext + G J, with G the
exact Green's matrix of the grid cells. Picard iteration: flux -> axis, edge, psi_N -> current -> flux.

Profile modes (prof dict; numbers are (B,) or (B, nx) arrays):
  "paxis"  FreeGS ConstrainPaxisIp: J = L [beta0 R/R0 + (1 - beta0) R0/R] s(psi_N), s = (1 - psi_N^alpha_m)^alpha_n;
           L and beta0 are set so the pressure on axis is `paxis` (Pa) and the total current is Ip.
           keys: paxis, alpha_m, alpha_n
  "tab"    pressure p(psi_N) tabulated on a uniform psi_N grid (B, nx), p(1) = 0, and FF' = mu0 R0 lam s(psi_N),
           lam set by the total current. keys: p_tab, alpha_m, alpha_n (optional s_tab (B, nx): tabulated FF' shape)

Vertical position. Elongated plasmas are vertically unstable, and with fixed coil currents plain Picard
iteration drifts up or down. Options:
  z_hold (B,) heights: each pass the current is moved rigidly so its centroid sits at z_hold, and a small
         radial field (amplitude h_br, mT at R0) makes the net vertical force zero. It stands for the vertical
         feedback and vessel currents that hold real plasmas.
  FwdConfig.symmetrize: average psi with its mirror image (up-down symmetric problems only).
Optional r_hold (B,): a uniform vertical field b_v (mT, "vessel field") is adjusted each pass until the
outer midplane edge of the plasma sits at r_hold (NaN = off for that frame).

Wall-edge fix (FwdConfig.wall_fix, on by default): for a plasma touching the wall, the edge flux is read on
the nominal wall lines of the box rather than on the first grid column outside it. Without it a MAST
plasma limited on the centre column sits about 1 cm too far inboard (found in the FreeGS cross-check).

Status per frame: 0 converged, 1 not converged, 3 plasma too small, 4 no axis or non-finite,
7 hold field too large, 8 r_hold target not reached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import MU0, Machine
from .greens import cell_matrix

_GRIDS: dict = {}


class Grid:
    """Fixed torch tensors of one machine, cropped to the plasma (wall) region plus one cell."""

    def __init__(self, m: Machine, dtype=torch.float64):
        W = m.wall_region()
        self.nZ, self.nR = m.shape
        iz, ir = np.nonzero(W)
        self.z0, self.z1 = max(iz.min() - 1, 0), min(iz.max() + 2, self.nZ)
        self.r0, self.r1 = max(ir.min() - 1, 0), min(ir.max() + 2, self.nR)
        Wc = W[self.z0:self.z1, self.r0:self.r1]
        self.H, self.Wd = Wc.shape
        t = lambda x, d=dtype: torch.as_tensor(np.ascontiguousarray(x), dtype=d)  # noqa: E731
        Wint = Wc.copy()
        for _ in range(2):                                   # axis candidates: at least 2 cells from the wall
            e = Wint.copy()
            e[1:, :] &= Wint[:-1, :]; e[:-1, :] &= Wint[1:, :]
            e[:, 1:] &= Wint[:, :-1]; e[:, :-1] &= Wint[:, 1:]
            Wint = e
        self.W, self.Wint = t(Wc, torch.bool), t(Wint, torch.bool)
        RR, ZZ = np.meshgrid(m.R, m.Z)
        self.Rc, self.Zc = t(self.crop(RR)), t(self.crop(ZZ))
        self.Rfull, self.Zfull = t(RR.ravel()), t(ZZ.ravel())
        self.Rw, self.Zw = self.Rc[self.W], self.Zc[self.W]
        self.R0, self.dA, self.dZ = m.R0, abs(float((m.R[1] - m.R[0]) * (m.Z[1] - m.Z[0]))), float(m.Z[1] - m.Z[0])
        self.iz_mid = int(np.argmin(np.abs(m.Z)))
        self.G = t(cell_matrix(m)).to(dtype)               # (n_cells, nZ*nR)
        self.dtype, self.machine = dtype, m

    def crop(self, x):
        return x[..., self.z0:self.z1, self.r0:self.r1]


def grid(m) -> Grid:
    if isinstance(m, Grid):
        return m
    if id(m) not in _GRIDS:
        _GRIDS[id(m)] = (m, Grid(m))
    return _GRIDS[id(m)][1]


# ---------------------------------------------------------------- axis and edge (no contours, batched)

def _cross_max(v, fill):
    p = F.pad(v, (1, 1, 1, 1), value=fill)
    return torch.maximum(torch.maximum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]), torch.maximum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]))


def edge_and_mask(g: Grid, pc, ax_idx, max_iter=400, check=8):
    """Edge flux psi_b (B,) and plasma mask (B, H, Wd) of cropped maps pc.

    v(p) = the best "bottleneck" flux over all grid paths from cell p to the wall. v at the axis is the edge
    level: an X-point for a diverted plasma, the wall contact for a limited one. The plasma is the set of
    cells with psi > psi_b that leave through the same bottleneck as the axis."""
    W = g.W[None]
    v = torch.where(W, torch.tensor(-torch.inf, dtype=pc.dtype), pc)
    for it in range(1, max_iter + 1):
        vn = torch.where(W, torch.maximum(v, torch.minimum(pc, _cross_max(v, -torch.inf))), v)
        if it % check == 0 and torch.equal(vn, v):
            break
        v = vn
    v = vn
    psi_b = v.flatten(1).gather(1, ax_idx[:, None])[:, 0]
    return psi_b, W & (pc > psi_b[:, None, None]) & (v == psi_b[:, None, None])


def find_axis(g: Grid, pc, prev_R=None, prev_Z=None, radius=None):
    """Strongest interior local maximum (optionally within `radius` of a previous axis): (idx, value, ok)."""
    ninf = torch.tensor(-torch.inf, dtype=pc.dtype)
    phi = torch.where(g.Wint[None], pc, ninf)
    lm = (phi == F.max_pool2d(phi[:, None], 3, 1, 1)[:, 0]) & g.Wint[None] & torch.isfinite(phi)
    if prev_R is not None:
        lm = lm & ((g.Rc[None] - prev_R[:, None, None]) ** 2 + (g.Zc[None] - prev_Z[:, None, None]) ** 2 <= radius ** 2)
    val, idx = torch.where(lm, pc, ninf).flatten(1).max(1)
    return idx, val, torch.isfinite(val)


def subpixel(g: Grid, pc, idx):
    """Axis position (R, Z) refined by a parabola through the peak cell and its neighbours."""
    b = torch.arange(pc.shape[0])
    iz, ir = (idx // g.Wd).clamp(1, g.H - 2), (idx % g.Wd).clamp(1, g.Wd - 2)
    c = pc[b, iz, ir]

    def d1(a, e):
        den = a - 2 * c + e
        ok = den != 0
        return torch.where(ok, (0.5 * (a - e) / torch.where(ok, den, torch.ones_like(den))).clamp(-1, 1), torch.zeros_like(den))
    dz, dr = d1(pc[b, iz - 1, ir], pc[b, iz + 1, ir]), d1(pc[b, iz, ir - 1], pc[b, iz, ir + 1])
    return g.Rc[iz, ir] + dr * (g.Rc[0, 1] - g.Rc[0, 0]), g.Zc[iz, ir] + dz * (g.Zc[1, 0] - g.Zc[0, 0])


def sample(g: Grid, psi, R, Z):
    """Bilinear values of full maps psi (B, nZ, nR) at points R, Z (B, C)."""
    Rg, Zg = g.Rfull.reshape(g.nZ, g.nR)[0], g.Zfull.reshape(g.nZ, g.nR)[:, 0]
    xy = torch.stack([(R - Rg[0]) / (Rg[-1] - Rg[0]) * 2 - 1, (Z - Zg[0]) / (Zg[-1] - Zg[0]) * 2 - 1], -1)
    return F.grid_sample(psi[:, None], xy[:, None].to(psi.dtype), align_corners=True)[:, 0, 0]


def wall_points(g: Grid, box):
    """Points along the four lines of the wall box, with the nearest plasma-region cell of each."""
    r0, r1, z0, z1 = box
    zz, rr = torch.linspace(z0, z1, 161, dtype=torch.float64), torch.linspace(r0, r1, 81, dtype=torch.float64)
    R = torch.cat([torch.full_like(zz, r0), torch.full_like(zz, r1), rr, rr])
    Z = torch.cat([zz, zz, torch.full_like(rr, z0), torch.full_like(rr, z1)])
    iz, ir = np.nonzero(g.W.numpy())
    d = (R.numpy()[:, None] - g.Rc.numpy()[iz, ir][None]) ** 2 + (Z.numpy()[:, None] - g.Zc.numpy()[iz, ir][None]) ** 2
    j = d.argmin(1)
    return R, Z, torch.as_tensor(iz[j]), torch.as_tensor(ir[j])


def limiter_edge(g: Grid, psi, pc, psi_b, mask, wp):
    """Wall-edge fix: a limited plasma's edge flux is the highest flux on the wall lines next to the plasma."""
    R, Z, iz, ir = wp
    B = pc.shape[0]
    pw = sample(g, psi.reshape(B, g.nZ, g.nR).double(), R[None].expand(B, -1), Z[None].expand(B, -1))
    pl = torch.where(mask[:, iz, ir], pw.to(psi_b.dtype), torch.full_like(pw, -math.inf, dtype=psi_b.dtype)).amax(1)
    psi_b = torch.where(pl > psi_b, pl, psi_b)
    return psi_b, mask & (pc > psi_b[:, None, None])


def outer_edge(g: Grid, psi, psi_ax, psi_b, R_ax):
    """Outer plasma edge on the midplane row: first radius outward of the axis where psi_N reaches 1."""
    B = psi.shape[0]
    Rg = g.Rfull.reshape(g.nZ, g.nR)[0].to(psi.dtype)
    x = (psi_ax[:, None] - psi.reshape(B, g.nZ, g.nR)[:, g.iz_mid]) / (psi_ax - psi_b).clamp(min=1e-12)[:, None]
    hit = (Rg[None] > R_ax.to(psi.dtype)[:, None]) & (x >= 1.0)
    j = torch.where(hit.any(1), hit.float().argmax(1), torch.full((B,), g.nR - 1)).clamp(1, g.nR - 1)
    x1, x0 = x.gather(1, j[:, None])[:, 0], x.gather(1, (j - 1)[:, None])[:, 0]
    R = Rg[j - 1] + ((1.0 - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0, 1) * (Rg[j] - Rg[j - 1])
    return torch.where(hit.any(1), R, torch.full_like(R, math.nan))


# ---------------------------------------------------------------- profiles

def shape(x, am, an):
    """(1 - x^am)^an for x in [0, 1]."""
    return (1.0 - x.clamp(0, 1) ** am).clamp(min=0) ** an


def shape_integral(am, an, n=2001):
    x = torch.linspace(0, 1, n, dtype=torch.float64)[None]
    return torch.trapezoid(shape(x, am[:, None].double(), an[:, None].double()), x, dim=1)


def interp_tab(tab, x):
    """Linear interpolation of tab (B, nx), given on a uniform grid of [0, 1], at x (B, n)."""
    nx = tab.shape[1]
    t = x.clamp(0, 1) * (nx - 1)
    i0 = t.floor().long().clamp(0, nx - 2)
    w = t - i0
    return tab.gather(1, i0) * (1 - w) + tab.gather(1, i0 + 1) * w


def _dtab(tab, x):
    """d tab / dx: node slopes by central differences, linearly interpolated (continuous in x)."""
    h = 1.0 / (tab.shape[1] - 1)
    sl = torch.empty_like(tab)
    sl[:, 1:-1] = (tab[:, 2:] - tab[:, :-2]) / (2 * h)
    sl[:, 0], sl[:, -1] = (tab[:, 1] - tab[:, 0]) / h, (tab[:, -1] - tab[:, -2]) / h
    return interp_tab(sl, x)


def _ff_shape(prof, x):
    s = shape(x, prof["alpha_m"].to(x.dtype)[:, None], prof["alpha_n"].to(x.dtype)[:, None])
    if prof["mode"] == "tab" and prof.get("s_tab") is not None:
        st = prof["s_tab"].to(x.dtype)
        s = torch.where(torch.isfinite(st).all(1)[:, None], interp_tab(torch.nan_to_num(st), x), s)
    return s


def profile_current(g: Grid, x, mask, flux_range, ip, prof):
    """Current per plasma cell (B, n_cells) in A, and the profile constants, for psi_N x (B, n_cells)."""
    Rw, R0, dA = g.Rw.to(x.dtype)[None], g.R0, g.dA
    s = _ff_shape(prof, x) * mask
    I_R, I_inv = ((Rw / R0) * s).sum(1) * dA, ((R0 / Rw) * s).sum(1) * dA
    if prof["mode"] == "paxis":
        Lb0 = prof["paxis"].to(x.dtype) * R0 / (flux_range * shape_integral(prof["alpha_m"], prof["alpha_n"]).to(x.dtype))
        L = ip / I_inv.clamp(min=1e-30) - Lb0 * (I_R / I_inv.clamp(min=1e-30) - 1.0)
        J = Lb0[:, None] * (Rw / R0) * s + (L - Lb0)[:, None] * (R0 / Rw) * s
        return J * dA, dict(L=L, beta0=Lb0 / L)
    if prof["mode"] == "tab":
        Jp = Rw * (-_dtab(prof["p_tab"].to(x.dtype), x) / flux_range[:, None]) * mask
        lam = (ip - Jp.sum(1) * dA) / I_inv.clamp(min=1e-30)
        return (Jp + lam[:, None] * (R0 / Rw) * s) * dA, dict(lam=lam)
    raise ValueError(prof["mode"])


def start_current(m, Rc, Zc, a, kappa, delta=0.0, peaking=1.0):
    """(B, n_cells) start-current share per plasma cell (rows sum to 1) of an analytic D-shaped plasma."""
    g = grid(m)
    f = lambda v: torch.as_tensor(v, dtype=torch.float64).reshape(-1, 1)  # noqa: E731
    Rc, Zc, a, kappa, d = f(Rc), f(Zc), f(a), f(kappa), f(delta)
    x, y = (g.Rw.double()[None] - Rc) / a, (g.Zw.double()[None] - Zc) / (kappa * a)
    rho2 = (x + d * y ** 2) ** 2 + y ** 2
    J = torch.where(rho2 < 1, (1 - rho2).clamp(min=0) ** peaking, torch.zeros_like(rho2))
    return J / J.sum(1, keepdim=True).clamp(min=1e-30)


def balance_height(m, psi_ext, Rc, a, kappa=1.8, delta=0.3, z_range=(-0.4, 0.4), dz=0.02):
    """Coil force-balance height (B,) in m: the height at which a rigid D-shaped plasma current (start_current
    shape, centre radius Rc, minor radius a) feels no net vertical force from the external flux psi_ext
    (B, nZ, nR). Uses only the coil currents and a nominal plasma shape.

    The vertical force on the plasma is F_Z = -2 pi sum_cells I dpsi_ext/dZ. Heights are scanned from
    z_range[0] to z_range[1]; of the heights where F_Z changes sign, the one nearest the midplane is returned
    (linear interpolation between scan points). NaN where F_Z keeps one sign over the whole range."""
    g = grid(m)
    psi_ext = torch.as_tensor(np.asarray(psi_ext), dtype=torch.float64).reshape(-1, g.nZ, g.nR)
    B = psi_ext.shape[0]
    d = torch.zeros_like(psi_ext)
    d[:, 1:-1] = (psi_ext[:, 2:] - psi_ext[:, :-2]) / (2 * g.dZ)
    dext = g.crop(d)[:, g.W]                                            # (B, n_cells) dpsi_ext/dZ
    zs = np.arange(z_range[0], z_range[1] + 1e-9, dz)
    one = np.ones(B)
    f = torch.stack([-(start_current(m, Rc, z * one, a, kappa * one, delta * one) * dext).sum(1) for z in zs], 1)
    f = f.numpy()
    out = np.full(B, np.nan)
    for b in range(B):
        j = np.nonzero(np.sign(f[b, :-1]) * np.sign(f[b, 1:]) <= 0)[0]
        if len(j):
            zc = zs[j] + dz * f[b, j] / np.where(f[b, j] != f[b, j + 1], f[b, j] - f[b, j + 1], 1.0)
            out[b] = zc[np.argmin(np.abs(zc))]
    return out


# ---------------------------------------------------------------- solution

class Solution:
    """psi (B, nZ, nR) plus per-frame results in `info` (also readable as attributes, e.g. sol.status)."""

    def __init__(self, machine, psi, ip, prof, info):
        self.machine, self.psi, self.ip, self.prof, self.info = machine, psi, ip, prof, info

    def __getattr__(self, k):
        if k != "info" and k in self.__dict__.get("info", {}):
            return self.info[k]
        raise AttributeError(k)

    def profiles(self, x, k=None):
        """(p [Pa], FF' [T^2 m]) at psi_N values x, as (B, len(x)) numpy arrays (or (len(x),) for frame k)."""
        B = len(self.ip)
        sel = (lambda v: v[k:k + 1] if torch.is_tensor(v) and v.ndim and len(v) == B else v) if k is not None else (lambda v: v)
        pr, inf, ip = {q: sel(v) for q, v in self.prof.items()}, {q: sel(v) for q, v in self.info.items()}, sel(self.ip)
        x = torch.as_tensor(np.clip(np.asarray(x, np.float64), 0, 1))[None].expand(len(ip), -1)
        R0, fr = self.machine.R0, (inf["psi_ax"] - inf["psi_b"]).double()[:, None]
        if pr["mode"] == "basis":                           # correction mode: weighted (R/R0)^a (1 - psi_N)^b shapes
            c = inf["coef"].double() * ip.double()[:, None] / (inf["S"].double() * inf["dA"])
            p, ff = torch.zeros_like(x), torch.zeros_like(x)
            for j, (a, b) in enumerate(pr["shapes"]):
                if a == 1:
                    p = p + c[:, j:j + 1] * fr / R0 * (1 - x) ** (b + 1) / (b + 1)
                else:
                    ff = ff + MU0 * R0 * c[:, j:j + 1] * (1 - x) ** b
        elif pr["mode"] == "paxis":
            xf = torch.linspace(0, 1, 2001, dtype=torch.float64)
            sf = shape(xf[None], pr["alpha_m"][:, None], pr["alpha_n"][:, None])
            tail = torch.flip(torch.cumulative_trapezoid(torch.flip(sf, [1]), dx=float(xf[1]), dim=1), [1])
            tail = torch.cat([tail, torch.zeros_like(tail[:, :1])], 1)            # int_x^1 s dx' on xf
            p = pr["paxis"][:, None] * interp_tab(tail, x) / tail[:, :1]
            ff = MU0 * R0 * (inf["L"] * (1 - inf["beta0"]))[:, None] * _ff_shape(pr, x)
        else:
            p, ff = interp_tab(pr["p_tab"], x), MU0 * R0 * inf["lam"][:, None] * _ff_shape(pr, x)
        return (p.numpy(), ff.numpy()) if k is None else (p[0].numpy(), ff[0].numpy())


@dataclass
class FwdConfig:
    n_iter: int = 80
    relax: float = 0.5            # psi <- psi + relax (psi_new - psi)
    tol: float = 1e-4             # converged if max |psi_new - psi| < tol (psi_ax - psi_b)
    axis_track: float = 0.15      # axis = strongest local maximum within this distance of the previous one (m)
    min_cells: int = 12
    symmetrize: bool = False      # average psi with its up-down mirror each pass (symmetric problems only)
    hold_step: float = 1.0        # fraction of the centroid height error removed per pass (z_hold)
    max_h_mT: float = 200.0       # hold field above this marks the frame as failed
    wall_fix: bool = True         # edge flux of a limited plasma read on the wall-box lines
    r_gain: float = 0.05          # r_hold: field change per pass per metre of edge error (T/m)
    bv_max: float = 60.0          # r_hold: largest vessel field (mT)


def forward_solve(m: Machine, psi_ext, ip, prof, cfg: FwdConfig = FwdConfig(), z_hold=None, J0=None,
                  psi_start=None, r_hold=None, bv0=None) -> Solution:
    """psi_ext (B, nZ, nR): flux of everything outside the plasma (coils, fixed vacuum fields); ip (B,) in A.
    Start from J0 (B, n_cells) current shares (start_current) or from a map psi_start (J0 then only seeds the
    first axis search). bv0 (B,): start value of the r_hold vessel field (mT)."""
    g, dt = grid(m), torch.float64
    B = psi_ext.shape[0]
    psi_ext = torch.as_tensor(psi_ext).to(dt).reshape(B, -1)
    ip = torch.as_tensor(ip).to(dt)
    G, Rf, Zf = g.G, g.Rfull, g.Zfull
    Vbr, Vbv = -1e-3 * Rf ** 2 * Zf / g.R0, 1e-3 * (Rf ** 2 - g.R0 ** 2) / 2      # radial / vertical field per mT
    flipZ = torch.arange(g.nZ * g.nR).reshape(g.nZ, g.nR).flip(0).flatten()
    prof = {k: (v if k == "mode" else torch.as_tensor(v).to(dt)) for k, v in prof.items()}
    psi = psi_start.to(dt).reshape(B, -1).clone() if psi_start is not None else psi_ext + (J0.to(dt) * ip[:, None]) @ G
    h = torch.zeros(B, dtype=dt)
    bv = torch.zeros(B, dtype=dt) if bv0 is None else torch.nan_to_num(torch.as_tensor(bv0).to(dt))
    rt = None if r_hold is None else torch.as_tensor(r_hold).to(dt)
    zt = None if z_hold is None else torch.as_tensor(z_hold).to(dt)
    R_out = torch.full((B,), math.nan, dtype=dt)
    change = torch.full((B,), math.inf, dtype=dt)
    R_prev = Z_prev = None
    if J0 is not None:                                    # first axis search near the start current's centroid
        w0 = J0.to(dt)
        R_prev, Z_prev = ((w0 * v[None]).sum(1) / w0.sum(1).clamp(min=1e-30) for v in (g.Rw, g.Zw))
    wp = wall_points(g, m.wall_box) if cfg.wall_fix else None
    if zt is not None:                                    # vertical derivatives for the force balance of the hold
        def dz_(f):
            d = torch.zeros_like(f); d[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * g.dZ)
            return g.crop(d)[..., g.W]
        dext, dV = dz_(psi_ext.reshape(B, g.nZ, g.nR)), dz_(Vbr.reshape(g.nZ, g.nR))
    for it in range(cfg.n_iter):
        pc = g.crop(psi.reshape(B, g.nZ, g.nR))
        idx, psi_ax, ok_ax = find_axis(g, pc) if R_prev is None else \
            find_axis(g, pc, R_prev, Z_prev, cfg.axis_track if it else 0.30)
        if R_prev is not None:
            idx0, _, ok0 = find_axis(g, pc)
            idx, ok_ax = torch.where(ok_ax, idx, idx0), ok_ax | ok0
        psi_ax = pc.flatten(1).gather(1, idx[:, None])[:, 0]
        psi_b, mask = edge_and_mask(g, pc, idx)
        if wp is not None:
            psi_b, mask = limiter_edge(g, psi, pc, psi_b, mask, wp)
        R_prev, Z_prev = subpixel(g, pc, idx)
        fr = psi_ax - psi_b
        good = ok_ax & torch.isfinite(psi_b) & (fr > 0)
        frs = torch.where(good, fr, torch.ones_like(fr))
        x = ((psi_ax[:, None, None] - pc) / frs[:, None, None])[:, g.W]
        Jc, extra = profile_current(g, x, mask[:, g.W].to(dt), frs, ip, prof)
        if zt is not None:
            # move the current rigidly to the target centroid height (linear interpolation between rows) ...
            zc = (Jc * g.Zw[None]).sum(1) / Jc.sum(1).clamp(min=1e-30)
            Jmap = torch.zeros(B, g.H, g.Wd, dtype=dt); Jmap[:, g.W] = Jc
            sh = cfg.hold_step * (zt - zc) / g.dZ
            k0 = torch.floor(sh); f = (sh - k0)[:, None, None]
            rows = torch.arange(g.H)[None, :] - k0.long()[:, None]

            def take(r):
                rr = r.clamp(0, g.H - 1)[:, :, None].expand(-1, -1, g.Wd)
                return Jmap.gather(1, rr) * ((r >= 0) & (r < g.H))[:, :, None]
            Jsh = ((1 - f) * take(rows) + f * take(rows - 1))[:, g.W]
            Jc = Jsh * (ip / Jsh.sum(1).clamp(min=1e-30))[:, None]
            # ... and choose the radial field h so that the net vertical force sum_cells I dpsi/dZ vanishes
            g1, gV = (Jc * dext).sum(1), (Jc * dV[None]).sum(1)
            h = torch.where(gV.abs() > 0, -g1 / torch.where(gV.abs() > 0, gV, torch.ones_like(gV)), torch.zeros_like(gV))
            psi_new = psi_ext + Jc @ G + h[:, None] * Vbr[None]
        else:
            psi_new = psi_ext + Jc @ G
        if rt is not None:
            R_out = outer_edge(g, psi, psi_ax, psi_b, R_prev)
            bv = (bv + 1e3 * cfg.r_gain * torch.nan_to_num(rt - R_out, nan=0.0)).clamp(-cfg.bv_max, cfg.bv_max)
            psi_new = psi_new + bv[:, None] * Vbv[None]
        if cfg.symmetrize:
            psi_new = 0.5 * (psi_new + psi_new[:, flipZ])
        change = (psi_new - psi).abs().amax(1) / frs.abs().clamp(min=1e-12)
        psi = torch.where(good[:, None], psi + cfg.relax * (psi_new - psi), psi)
        if bool((change < cfg.tol * 0.5).all()) and it > 5:
            break
    # results on the final map
    pc = g.crop(psi.reshape(B, g.nZ, g.nR))
    idx, _, ok_ax = find_axis(g, pc, R_prev, Z_prev, cfg.axis_track)
    psi_ax = pc.flatten(1).gather(1, idx[:, None])[:, 0]
    psi_b, mask = edge_and_mask(g, pc, idx)
    if wp is not None:
        psi_b, mask = limiter_edge(g, psi, pc, psi_b, mask, wp)
    R_ax, Z_ax = subpixel(g, pc, idx)
    ncell = mask.flatten(1).sum(1)
    status = torch.zeros(B, dtype=torch.int64)
    status[~(change < cfg.tol)] = 1
    status[ncell < cfg.min_cells] = 3
    status[h.abs() > cfg.max_h_mT] = 7
    if rt is not None:
        R_out = outer_edge(g, psi, psi_ax, psi_b, R_ax)
        status[torch.isfinite(rt) & ((bv.abs() >= cfg.bv_max * 0.999) | ~((R_out - rt).abs() < 0.03))] = 8
    status[~(ok_ax & torch.isfinite(psi_b) & torch.isfinite(psi).all(1) & (psi_ax > psi_b))] = 4
    info = dict(status=status, change=change, R_ax=R_ax, Z_ax=Z_ax, psi_ax=psi_ax, psi_b=psi_b, ncell=ncell,
                h_br=h, bv=bv, R_out=R_out, mask=mask, J=Jc, n_iter=it + 1, **extra)
    return Solution(m, psi.reshape(B, g.nZ, g.nR), ip, prof, info)
