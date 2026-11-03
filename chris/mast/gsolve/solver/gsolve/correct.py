"""Correction mode: the free-boundary equilibrium closest to a given flux map (for example a model's prediction).

The coil flux psi_ext is kept exactly. The plasma current is a weighted sum of a few fixed profile shapes,
J = sum_k c_k (R/R0)^a_k (1 - psi_N)^b_k (terms in R carry pressure current, terms in 1/R the toroidal-field
current), optionally plus a few vacuum fields (uniform vertical "bv", radial "br", harmonics "H0".."H8") that
stand for unmeasured external currents. Each Picard pass finds axis, edge and psi_N on the current map,
then picks the weights by least squares so the result stays close to the given map, with the total current
fixed to Ip (sum c = 1) and, unless `signed`, c >= 0. At convergence the result obeys Grad-Shafranov with
the real coils; its edge and shape are wherever the solution puts them.

Status per frame: 0 ok, 1 not converged, 2 axis moved more than max_shift, 3 plasma too small,
4 no axis or non-finite, 5 Ip below min_ip, 6 strongly negative current. Rejected frames keep the given map.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .forward import Solution, edge_and_mask, find_axis, grid, subpixel

# Profile shapes: (power of R/R0, power of (1 - psi_N)). Terms in R carry pressure current (p'), terms in
# 1/R toroidal-field current (FF'). Every family is a low-order polynomial in psi_N, the representation of
# EFIT's "Lao profiles" (Lao et al. 1985).
PROFILES = {
    "pf2": [(1, 1), (-1, 1)],                              # p', FF' linear, zero at the edge
    "pf3": [(1, 1), (-1, 1), (-1, 2)],                     # pf2 plus a quadratic FF' term
    # p' and FF' quadratic and zero at the edge: the Lao-profile family (n_p = n_F = 1, zero edge value) that
    # Pentland et al. 2024 (arXiv:2407.12432, section 4.3) document for EFIT++ magnetics-only reconstructions of MAST-U
    "pf4": [(1, 1), (1, 2), (-1, 1), (-1, 2)],
    "pf5": [(1, 1), (1, 2), (-1, 1), (-1, 2), (-1, 3)],    # pf4 plus a cubic FF' term
    "pf3e": [(1, 1), (-1, 1), (-1, 2), (-1, 0)],
    # flat shapes carry current up to the edge (pedestal current); designed on DIII-D (H-mode edge current)
    "pf4e": [(1, 0), (1, 1), (-1, 0), (-1, 1), (-1, 2)],
}


def vacuum_field(name, R, Z, R0):
    """Flux pattern of one vacuum field (a solution of Delta* psi = 0): bv, br per 1 mT; H0..H8 per 0.01 unit.

    H0..H8 are homogeneous solutions of the Grad-Shafranov equation in x = R/R0, y = Z/R0 (those listed by
    Cerfon & Freidberg 2010, some shifted by a constant), up-down even and odd: constant, uniform vertical
    field, uniform radial field, quadrupole shaping field, a term linear in Z, and four higher-order terms."""
    x, y = R / R0, Z / R0
    if name == "bv":
        return 1e-3 * (R ** 2 - R0 ** 2) / 2                   # uniform vertical field
    if name == "br":
        return -1e-3 * R ** 2 * Z / R0                          # radial field, 1 mT at R0
    lx = np.log(x)
    H = [1 + 0 * x, x ** 2 - 1, x ** 2 * y, x ** 4 - 4 * x ** 2 * y ** 2 - 1, y, y ** 2 - x ** 2 * lx,
         x ** 6 - 12 * x ** 4 * y ** 2 + 8 * x ** 2 * y ** 4 - 1, y ** 3 - 3 * x ** 2 * y * lx, x ** 4 * y - 4.0 / 3.0 * x ** 2 * y ** 3]
    return 1e-2 * H[int(name[1:])]


@dataclass
class CorrConfig:
    """Settings of the correction mode. The source of each default is given next to it."""
    # Picard loop: blend 0.5 as in forward mode and in the FreeGS comparison. Pass count and tolerance come from a
    # convergence study of the solver itself (no measured equilibrium involved): 400 synthetic MAST equilibria
    # from examples/generate_synthetic.py, each moved by one grid row or column or given a 2 mT vertical field,
    # then corrected with these defaults. After 120 passes 98% have their last change below 0.1% of the flux
    # range (81% after 60 passes).
    n_iter: int = 120
    relax: float = 0.5
    # profile family: EFIT++'s documented magnetics-only family for MAST-U (Pentland et al. 2024), weights kept >= 0 so the current
    # stays positive (signed=True gives the free-sign fit EFIT++ itself uses)
    profile: str = "pf4"
    # vacuum fields standing for unmeasured external currents: the five lowest-order patterns (constant,
    # vertical field, radial field, shaping field, Z term), the five fields of the DIII-D comparison in README
    vacuum: tuple = ("H0", "H1", "H2", "H3", "H4")
    vac_sigma_mT: float = 5.0       # weak prior on the vacuum-field amplitudes: a few % of the ~130 mT coil field
    w_out: float = 1.0              # fit weight of pixels outside the plasma region (inside = 1): no weighting
    tol: float = 1e-3               # converged if the last change < 0.1% of the axis-to-edge flux (study above)
    max_shift: float = 0.5          # axis may not move further than this from the given map's (m): the minor
                                    # radius of either machine; a larger move means a different plasma
    min_cells: int = 12             # numerical: fewer plasma cells than this cannot resolve a plasma
    axis_track: float = 0.15        # as in forward mode
    nnls_rounds: int = 3            # active-set rounds of the non-negative fit
    signed: bool = False            # allow negative weights (net current stays Ip)
    nonneg: str = ""                # with signed: "flat" keeps the flat (edge-current) shapes >= 0
    neg_current: float = 0.25       # reject if the most negative cell current exceeds this share of the largest
                                    # (only reachable with signed weights; set on DIII-D: 0, 5%, 25%, 50% tried)
    min_ip: float = 1e5             # A; below 100 kA there is no developed plasma to correct


SIMPLE = CorrConfig()
# DIII-D setting: signed edge-current family and all nine vacuum fields, designed on DIII-D (H-mode edge current)
RICH = CorrConfig(profile="pf4e", signed=True, vacuum=tuple(f"H{j}" for j in range(9)))


def _solve(m, psi_model, psi_ext, ip, cfg):
    g, dt = grid(m), torch.float64
    B, P = psi_model.shape[0], g.nZ * g.nR
    psi_model, psi_ext, ip = psi_model.to(dt), psi_ext.to(dt), ip.to(dt)
    shapes = PROFILES[cfg.profile]
    K = len(shapes)
    V = torch.stack([torch.as_tensor(vacuum_field(v, g.Rfull.numpy(), g.Zfull.numpy(), g.R0)) for v in cfg.vacuum]) \
        if cfg.vacuum else torch.zeros(0, P, dtype=dt)
    M = V.shape[0]
    n = K + M
    wfull = torch.zeros(g.nZ, g.nR, dtype=torch.bool)
    g.crop(wfull)[:] = g.W
    w = torch.where(wfull, 1.0, cfg.w_out).double().flatten()
    w = (w / w.sum())[None].expand(B, -1)
    y = (psi_model - psi_ext).flatten(1).double()
    psi = psi_model.clone()
    idx0, _, ok = find_axis(g, g.crop(psi))
    R_ax0, Z_ax0 = subpixel(g, g.crop(psi), idx0)
    R_prev, Z_prev = R_ax0.clone(), Z_ax0.clone()
    flux_range = torch.ones(B, dtype=dt)
    constrained = torch.tensor([b == 0 for _, b in shapes]) if cfg.signed and cfg.nonneg == "flat" \
        else torch.full((K,), not cfg.signed)
    s_vec = torch.cat([torch.ones(K), torch.zeros(M)]).double()
    for it in range(cfg.n_iter):
        pc = g.crop(psi)
        idx, _, ok_ax = find_axis(g, pc, R_prev, Z_prev, cfg.axis_track)
        idx = torch.where(ok_ax, idx, idx0)
        psi_ax = pc.flatten(1).gather(1, idx[:, None])[:, 0]
        psi_b, mask = edge_and_mask(g, pc, idx)
        R_prev, Z_prev = subpixel(g, pc, idx)
        flux_range = psi_ax - psi_b
        good = ok_ax & torch.isfinite(psi_b) & (flux_range > 0)
        x = ((psi_ax[:, None, None] - pc) / torch.where(good, flux_range, torch.ones_like(flux_range))[:, None, None])
        xW, maskW = x.clamp(0, 1)[:, g.W], mask[:, g.W]
        b_ = torch.stack([(g.Rw / g.R0) ** a * (1 - xW) ** b * maskW for a, b in shapes], 1)    # (B, K, n_cells)
        S = b_.sum(2)
        I = b_ / torch.where(S > 0, S, torch.ones_like(S))[:, :, None]                         # 1 A in total each
        A = torch.cat([(I * ip[:, None, None]) @ g.G, V[None].expand(B, -1, -1)], 1).double()
        Aw = A * w[:, None]
        N, rhs = Aw @ A.transpose(1, 2), (Aw @ y[:, :, None])[:, :, 0]
        if M:   # prior: amplitudes ~ N(0, sigma^2), scaled to the mean squared flux of the given map
            lam = (y ** 2 * w).sum(1) * 1e-4 / cfg.vac_sigma_mT ** 2
            N[:, K:, K:] += torch.diag_embed(lam[:, None].expand(-1, M))
        scale = N.diagonal(dim1=1, dim2=2).abs().mean(1).clamp(min=1e-30)
        N = N + 1e-9 * scale[:, None, None] * torch.eye(n, dtype=N.dtype)
        active = torch.ones(B, K, dtype=torch.bool)
        for _ in range(cfg.nnls_rounds if constrained.any() else 1):    # active set: drop the most negative weight
            kkt = torch.zeros(B, n + 1, n + 1, dtype=torch.float64)
            kkt[:, :n, :n] = N
            kkt[:, :K, :K] += torch.diag_embed(torch.where(active, 0.0, 1e8).double() * scale[:, None])
            kkt[:, :n, n] = s_vec; kkt[:, n, :n] = s_vec
            sol = torch.linalg.solve(kkt, torch.cat([rhs, torch.ones(B, 1, dtype=torch.float64)], 1)[:, :, None])[:, :n, 0]
            neg = (sol[:, :K] < -1e-9) & active & constrained[None]
            if not neg.any():
                break
            worst = torch.where(neg, sol[:, :K], torch.zeros_like(sol[:, :K])).argmin(1)
            drop = torch.zeros_like(active)
            drop[torch.arange(B), worst] = True
            active = active & ~(drop & neg.any(1, keepdim=True))
        sol[:, :K] = torch.where(constrained[None], sol[:, :K].clamp(min=0), sol[:, :K])
        sol[:, :K] = sol[:, :K] / sol[:, :K].sum(1, keepdim=True).clamp(min=1e-12)     # sum c = 1 exactly
        Jc = (sol[:, :K, None].to(dt) * I).sum(1)
        neg_frac = (-Jc.amin(1)).clamp(min=0) / Jc.amax(1).clamp(min=1e-12)
        psi_new = (psi_ext.flatten(1) + (sol.to(dt)[:, :, None] * A.to(dt)).sum(1)).reshape(psi.shape)
        change = (psi_new - psi).flatten(1).abs().amax(1) / flux_range.abs().clamp(min=1e-12)
        relax = cfg.relax if it < cfg.n_iter - 1 else 1.0          # last step: take the solver map
        psi = torch.where(good[:, None, None], psi + relax * (psi_new - psi), psi)
    pc = g.crop(psi)
    idx, psi_ax, ok_ax = find_axis(g, pc, R_prev, Z_prev, cfg.axis_track)
    psi_b, mask = edge_and_mask(g, pc, torch.where(ok_ax, idx, idx0))
    R_ax, Z_ax = subpixel(g, pc, idx)
    shift = torch.sqrt((R_ax - R_ax0) ** 2 + (Z_ax - Z_ax0) ** 2)
    ncell = mask.flatten(1).sum(1)
    status = torch.zeros(B, dtype=torch.int64)
    status[~(change < cfg.tol)] = 1
    status[shift > cfg.max_shift] = 2
    status[ncell < cfg.min_cells] = 3
    status[neg_frac > cfg.neg_current] = 6
    status[~(ok_ax & ok & torch.isfinite(psi).flatten(1).all(1) & torch.isfinite(psi_b))] = 4
    status[ip < cfg.min_ip] = 5
    psi = torch.where((status == 0)[:, None, None], psi, psi_model)
    return psi, dict(status=status, change=change, R_ax0=R_ax0, Z_ax0=Z_ax0, R_ax=R_ax, Z_ax=Z_ax, shift=shift,
                     ncell=ncell, coef=sol[:, :K], vac=sol[:, K:], S=S.double(), psi_ax=psi_ax, psi_b=psi_b,
                     neg_frac=neg_frac, dA=g.dA)


def correct(m, psi_model, psi_ext, ip, cfg: CorrConfig = SIMPLE, fallback: CorrConfig | None = None) -> Solution:
    """psi_model, psi_ext: (B, nZ, nR) physical flux (axis = maximum); ip (B,) in A.
    fallback: frames rejected by `cfg` get a second solve with this setting (e.g. cfg=RICH, fallback=SIMPLE);
    its profile shapes must be a subset of cfg's. sol.psi holds the solver map where status == 0."""
    psi_model, psi_ext, ip = (torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v for v in (psi_model, psi_ext, ip))
    psi, info = _solve(m, psi_model, psi_ext, ip, cfg)
    info["fallback_used"] = torch.zeros(len(ip), dtype=torch.bool)
    bad = torch.nonzero((info["status"] != 0) & (info["status"] != 5)).squeeze(1)
    if fallback is not None and len(bad):
        p2, i2 = _solve(m, psi_model[bad], psi_ext[bad], ip[bad], replace(fallback, min_ip=cfg.min_ip))
        gb = bad[i2["status"] == 0]
        ok2 = i2["status"] == 0
        psi[gb] = p2[ok2]
        for k in ("status", "change", "R_ax", "Z_ax", "shift", "ncell", "psi_ax", "psi_b", "neg_frac"):
            info[k][gb] = i2[k][ok2].to(info[k].dtype)
        main, fb = PROFILES[cfg.profile], PROFILES[fallback.profile]
        for j, sh in enumerate(main):          # fallback weights in the main shape list (missing shapes: 0)
            info["coef"][gb, j] = i2["coef"][ok2, fb.index(sh)] if sh in fb else 0.0
            info["S"][gb, j] = i2["S"][ok2, fb.index(sh)] if sh in fb else 1.0
        info["vac"][gb] = float("nan")
        info["fallback_used"][gb] = True
    return Solution(m, psi, ip.double(), {"mode": "basis", "shapes": PROFILES[cfg.profile]}, info)
