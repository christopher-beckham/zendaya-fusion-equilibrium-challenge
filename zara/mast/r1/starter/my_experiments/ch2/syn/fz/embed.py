"""Geometry-agnostic input embeddings of the pegs model, from a first-pass equilibrium. Any tokamak.

First pass: free-boundary solve (physics.GSLayer.solve, 40 damped Picard steps) with a generic profile, Q = 1 - psi_N
and pressure share P = 0.5 (1 - psi_N) (scan on 24 shots per machine: share 0.3 / 0.5 / 0.7 puts the DIII-D axis
-12 / -8 / -4 cm off EFIT on average; with share 1 many MAST frames found no radial equilibrium and drifted to the wall),
from the exact coil flux and the measured Ip. It fixes each frame's own units: R0 = R_geo, a, the flux unit
mu0 Ip R0 and the poloidal field unit mu0 Ip / l_p, and the psi_N coordinate used to read the Thomson points.

Every feature is dimensionless and carries a Z-parity (+1 even, -1 odd under Z -> -Z), so the network can be made
exactly up-down equivariant. No machine name, coil index, channel index, absolute time or per-machine statistic
enters. Groups:
    geometry     eps, kappa, triangularity (mean / up-down difference), Z offsets, Shafranov shift, l_p / 2 pi a,
                 first-pass l_i and residual, upper / lower X-point positions and flux gaps
    coil field   vacuum-harmonic coefficients about (R0, Z_axis) in units of mu0 Ip R0 (vertical field, radial
                 field, decay index, quadrupole, hexapole); B_Z -> Shafranov Lambda_vac; B_R in mu0 Ip / 2 pi a;
                 Fourier modes of psi_coil on the first-pass LCFS and half-radius surface
    current      q* = 5 a^2 B_T (1 + kappa^2) / (2 R0 Ip[MA]), Uckan q95
    kinetics     Thomson on first-pass psi_N: electron beta_p, beta_t, beta_t q*^2 / eps, Greenwald fraction,
                 Te / ne / pe peaking, Spitzer-current l_i, binned Te / ne shapes, psi_N of the Te / ne edge drop,
                 midplane Te-peak radius, Thomson LCFS crossings, a, R0, eps and Shafranov shift from Thomson
    time         log tau_R (Spitzer, <Te>, a), d ln Ip / dt x tau_R (instantaneous and exponentially weighted over
                 tau_R), Ip / max Ip, eddy proxies d/dt of the coil multipoles x tau_R
Thomson also enters as a point cloud (tokens): (R - R0) / a, (Z - Z_axis) / a, psi_N, Te / Te0, ne / ne0,
log Te[keV], log ne[1e19], log local electron beta.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from physics import GSLayer, MU0, N_HARM, HARM_PARITY

E_KEV = 1.602176634e-16
BINS = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.1])
MODES = (1, 2, 3, 4)
N_TOK = 8
TOK_PARITY = (1, -1, 1, 1, 1, 1, 1, 1)


P_SHARE = 0.5          # pressure share of the generic first-pass profile (P = P_SHARE (1 - psi_N))


def generic_profile(B, device, n=33):
    x = torch.linspace(0, 1, n, device=device)
    q = (1 - x)[None].expand(B, -1).contiguous()
    return P_SHARE * q, q.clone()


HARM_FIT = (0, 2, 3, 4, 5, 6, 7)   # harmonics fitted to the coil flux: constant, radial field, x^2 y, quadrupole,
                                   # 3 x^4 y - 4 x^2 y^3, hexapole, y^2 - x^2 ln x (the vertical field is bz_norm)


def _names():
    n, par = [], []
    def add(k, p):
        n.append(k); par.append(p)
    for k, p in [("eps", 1), ("kappa", 1), ("tri_avg", 1), ("tri_dif", -1), ("zgeo_a", -1), ("zax_a", -1),
                 ("shift_a", 1), ("lp_2pia", 1), ("li_fp", 1), ("log_res_fp", 1), ("isx_fp", 1),
                 ("xdpsi_sum", 1), ("xdpsi_dif", -1), ("xz_sum", 1), ("xz_dif", -1), ("xr_sum", 1), ("xr_dif", -1)]:
        add(k, p)
    for j, col in enumerate(HARM_FIT[1:], 1):
        add(f"harm{j}", HARM_PARITY[col])
    for k, p in [("bz_norm", 1), ("lam_vac", 1), ("br_norm", -1), ("decay", 1)]:
        add(k, p)
    for ring in ("lcfs", "half"):
        for m in MODES:
            add(f"vac_{ring}_c{m}", 1)
        for m in MODES:
            add(f"vac_{ring}_s{m}", -1)
    for k in ("log_qstar", "log_q95u"):
        add(k, 1)
    for k in ("log_beta_pe", "log_beta_te", "log_bq2e", "log_fgw", "log_te_peak", "log_ne_peak", "log_pe_peak",
              "li_spitzer", "psin_te30", "psin_ne30", "frac_hot_out", "log_n_in"):
        add(k, 1)
    nb = len(BINS) - 1
    for b in range(nb):
        add(f"te_b{b}", 1)
    for b in range(nb):
        add(f"ne_b{b}", 1)
    for k in ("ts_rpk_a", "ts_rout_a", "ts_rin_a", "ts_a_a", "ts_r0_a", "ts_eps", "ts_shift"):
        add(k, 1)
    for k, p in [("log_tauR", 1), ("dlnip_tauR", 1), ("dlnip_ew", 1), ("ip_over_max", 1),
                 ("ddt_bz", 1), ("ddt_br", -1), ("ddt_quad", 1), ("ddt_flux", 1)]:
        add(k, p)
    return n, par


NAMES, PARITY = _names()
TIME_FEATS = ["log_tauR", "dlnip_tauR", "dlnip_ew", "ip_over_max", "ddt_bz", "ddt_br", "ddt_quad", "ddt_flux"]


def _fourier(vals, theta):
    v = vals - vals.mean(1, keepdim=True)
    c = [2 * (v * torch.cos(m * theta)).mean(1) for m in MODES]
    s = [2 * (v * torch.sin(m * theta)).mean(1) for m in MODES]
    return c + s


# Profile ladder for frames whose first pass fails. Every frame of the challenge data is diverted (dataset selection
# on both machines), so a limited first pass is a wrong-branch solution:
#   leaning on the INNER wall (limited, contact inboard of the axis)  -> too little outward force -> more pressure
#   outer-wall contact, collapse or no convergence                    -> too much outward force   -> less pressure,
#                                                                         then zero pressure with a broad current
# (MAST: the collapse rate rises from 16 % to 75 % as |I_P5| / Ip falls; DIII-D: 25 % of frames converged limited on
# the inner wall with the axis ~25 cm inboard at pressure share 0.7.) The best result is kept: converged and diverted
# first, then converged, then the lowest residual.
LADDER_DOWN = ((0.2, "lin"), (0.0, "broad"))
LADDER_UP = ((1.2, "lin"), (2.0, "lin"))
LADDER = ((None, "lin"),) + LADDER_DOWN                  # kept for reference by older code


def ladder_profile(B, device, share, shape, n=33):
    x = torch.linspace(0, 1, n, device=device)[None].expand(B, -1)
    q = (1 - x) if shape == "lin" else (1 - x ** 2)
    return ((P_SHARE if share is None else share) * q).contiguous(), q.contiguous()


@torch.no_grad()
def _classify(gs, J, res, ip, psi_ext):
    """Quality score (higher is better) and failure direction of first-pass solutions.
    score: 3 converged + diverted, 2 converged, 1 unconverged, 0 collapsed / non-finite.
    inner: limited with the contact point inboard of the magnetic axis."""
    st = gs.state(psi_ext + gs.plasma(J), J, ip)
    poly, _ = gs.contour(st["phi"], st["R_axis"], st["Z_axis"], st["psi_b"])
    a = gs.shape(poly)["a"]
    fin = torch.isfinite(res) & torch.isfinite(a)
    collapsed = a < 0.6 * 0.45 * gs.halfwidth
    conv = res < 3e-2
    score = torch.where(fin & ~collapsed, 1 + conv.float() + (conv & st["is_x"]).float(), torch.zeros_like(res))
    inner = ~st["is_x"] & (st["lim_R"] < st["R_axis"]) & ~collapsed
    return score, inner


@torch.no_grad()
def first_pass(gs: GSLayer, I, ip, n_iter=40, omega=0.5, ladder=True):
    B = ip.shape[0]
    psi_ext = gs.vacuum(I)
    P, Q = generic_profile(B, gs.dev)
    J, res = gs.solve(psi_ext, ip, P, Q, n_iter=n_iter, omega=omega)
    rung = torch.zeros(B, device=gs.dev)
    if ladder:
        score, inner = _classify(gs, J, res, ip, psi_ext)
        for direction, steps, code in (("up", LADDER_UP, 10), ("down", LADDER_DOWN, 1)):
            for r, (share, shape) in enumerate(steps):
                todo = (score < 3) & (inner if direction == "up" else ~inner)
                if not bool(todo.any()):
                    break
                ib = torch.nonzero(todo)[:, 0]
                Pb, Qb = ladder_profile(len(ib), gs.dev, share, shape)
                Jb, rb = gs.solve(psi_ext[ib], ip[ib], Pb, Qb, n_iter=n_iter, omega=omega)
                sb, nb = _classify(gs, Jb, rb, ip[ib], psi_ext[ib])
                take = (sb > score[ib]) | ((sb == score[ib]) & (rb < res[ib]))
                it = ib[take]
                J[it], res[it], rung[it], score[it] = Jb[take], rb[take], float(code + r), sb[take]
                if direction == "up":
                    inner[it] = nb[take]
    psi = psi_ext + gs.plasma(J)
    st = gs.state(psi, J, ip)
    poly, ok = gs.contour(st["phi"], st["R_axis"], st["Z_axis"], st["psi_b"])
    geo = gs.shape(poly)
    w = st["plasma"].float()
    li, amp = gs.li(st["phi"], w, poly)
    return {"J": J, "res": res, "psi": psi, "psi_ext": psi_ext, "poly": poly, "poly_ok": ok.all(1), "li": li, "rung": rung, **st, **geo}


@torch.no_grad()
def state_to_fp(gs: GSLayer, st: dict, I) -> dict:
    """The model's own equilibrium as the reference of a second embedding pass: same keys as first_pass(). The coil field
    is the pure vacuum flux (no learned correction), so the coil-field features keep their meaning."""
    poly, ok = gs.contour(st["phi"], st["R_axis"], st["Z_axis"], st["psi_b"])
    geo = gs.shape(poly)
    li, _ = gs.li(st["phi"], st["plasma"].float(), poly)
    sw = (st["psi_axis"] - st["psi_b"]).clamp(min=1e-9)
    res = gs.plasma(st["J_next"] - st["J"]).abs().amax(1) / sw
    keys = ("phi", "aiz", "air", "psi_axis", "R_axis", "Z_axis", "psi_b", "plasma", "is_x", "xpt_R", "xpt_Z", "lim_R", "lim_Z")
    return {"J": st["J"].detach(), "res": res, "psi": st["psi"].detach(), "psi_ext": gs.vacuum(I), "poly": poly, "poly_ok": ok.all(1),
            "li": li, "rung": torch.zeros_like(res), **{k: (st[k].detach() if torch.is_tensor(st[k]) else st[k]) for k in keys}, **geo}


@torch.no_grad()
def frame_features(gs: GSLayer, fp: dict, ip, itf, pts, te, ne):
    """Per-frame features (B, F) without the time group (filled later per shot), Thomson tokens (B, N, N_TOK),
    token mask (B, N), and auxiliary per-frame quantities for the time group."""
    B = ip.shape[0]
    dev = gs.dev
    nan = torch.full((B,), float("nan"), device=dev)
    poly = fp["poly"]
    Rg, a, kap, lp = fp["R_geo"], fp["a"].clamp(min=1e-3), fp["kappa"], fp["lp"]
    Rax, Zax = fp["R_axis"], fp["Z_axis"]
    sw = (fp["psi_axis"] - fp["psi_b"]).clamp(min=1e-9)
    f = {}
    f["eps"] = a / Rg
    f["kappa"] = kap
    f["tri_avg"] = 0.5 * (fp["tri_top"] + fp["tri_bot"])
    f["tri_dif"] = 0.5 * (fp["tri_top"] - fp["tri_bot"])
    f["zgeo_a"] = fp["Z_geo"] / a
    f["zax_a"] = Zax / a
    f["shift_a"] = (Rax - Rg) / a
    f["lp_2pia"] = lp / (2 * math.pi * a)
    f["li_fp"] = fp["li"]
    f["log_res_fp"] = torch.log10(fp["res"].clamp(min=1e-8))
    f["isx_fp"] = fp["is_x"].float()
    # upper / lower X-points: saddle with the highest flux above / below the axis
    sv, sR, sZ, sok = gs.saddles(fp["phi"], k=6)
    up = sok & (sZ > Zax[:, None]); lo = sok & (sZ < Zax[:, None])
    def pick(msk):
        v = torch.where(msk, sv, torch.full_like(sv, -float("inf")))
        best, i = v.max(1)
        okb = torch.isfinite(best)
        g = lambda t: torch.where(okb, t.gather(1, i[:, None])[:, 0], nan)
        return g(sv), g(sR), g(sZ)
    vu, Ru, Zu = pick(up); vl, Rl, Zl = pick(lo)
    du, dl = (fp["psi_b"] - vu) / sw, (fp["psi_b"] - vl) / sw
    f["xdpsi_sum"], f["xdpsi_dif"] = du + dl, du - dl
    zu, zl = (Zu - Zax) / a, (Zax - Zl) / a
    f["xz_sum"], f["xz_dif"] = zu + zl, zu - zl
    ru, rl = (Ru - Rg) / a, (Rl - Rg) / a
    f["xr_sum"], f["xr_dif"] = ru + rl, ru - rl
    # coil field: harmonic least squares about (R0, Z_axis) on the first-pass plasma region (dilated)
    unit = MU0 * ip * Rg
    H = gs.harmonics(Rg, Zax)[:, list(HARM_FIT)]                                       # (B, 7, P)
    reg = (torch.nn.functional.max_pool2d(fp["plasma"][:, None].float(), 5, stride=1, padding=2)[:, 0] > 0).view(B, -1).float()
    A = H * reg[:, None]
    y = (fp["psi_ext"] / unit[:, None]) * reg
    AtA = A @ A.transpose(1, 2) + 1e-9 * torch.eye(len(HARM_FIT), device=dev)[None]
    coef = torch.linalg.solve(AtA, (A @ y[:, :, None]))[:, :, 0]
    for j in range(1, len(HARM_FIT)):
        f[f"harm{j}"] = coef[:, j]
    vac = fp["psi_ext"].view(B, gs.nZ, gs.nR)
    gR, gZ = gs.grad(vac)
    BZ, BR = gR / gs.RR, -gZ / gs.RR
    dBZ = torch.gradient(BZ, spacing=gs.dR, dim=2)[0]
    ctr = torch.stack([Rg, Zax], 1)[:, None]
    bz, br, dbz = gs.sample(BZ, ctr)[:, 0], gs.sample(BR, ctr)[:, 0], gs.sample(dBZ, ctr)[:, 0]
    f["bz_norm"] = bz * 4 * math.pi * Rg / (MU0 * ip)
    f["lam_vac"] = -f["bz_norm"] - torch.log(8 * Rg / (a * kap.clamp(min=0.3).sqrt())) + 1.5
    f["br_norm"] = br * 2 * math.pi * a / (MU0 * ip)
    f["decay"] = (-Rg * dbz / torch.where(bz.abs() > 1e-9, bz, torch.full_like(bz, 1e-9))).clamp(-20, 20)
    th = gs.theta
    vl = gs.sample(vac, poly) / unit[:, None]
    half = torch.stack([Rax[:, None] + 0.5 * (poly[..., 0] - Rax[:, None]), Zax[:, None] + 0.5 * (poly[..., 1] - Zax[:, None])], -1)
    vh = gs.sample(vac, half) / unit[:, None]
    for ring, vals in (("lcfs", vl), ("half", vh)):
        modes = _fourier(vals, th)
        for i, m in enumerate(MODES):
            f[f"vac_{ring}_c{m}"] = modes[i]
            f[f"vac_{ring}_s{m}"] = modes[len(MODES) + i]
    # current / field
    BT0 = MU0 * gs.tf_turns * itf / (2 * math.pi * Rg)
    IpMA = ip / 1e6
    qstar = 5 * a ** 2 * BT0 * (1 + kap ** 2) / (2 * Rg * IpMA)
    eps = a / Rg
    dlt = f["tri_avg"]
    q95u = (5 * a ** 2 * BT0 / (Rg * IpMA) * (1 + kap ** 2 * (1 + 2 * dlt ** 2 - 1.2 * dlt ** 3)) / 2
            * (1.17 - 0.65 * eps) / (1 - eps ** 2).clamp(min=0.05) ** 2)
    f["log_qstar"], f["log_q95u"] = torch.log(qstar.clamp(min=1e-3)), torch.log(q95u.clamp(min=1e-3))
    # ---- Thomson on first-pass psi_N
    ok_pts = torch.isfinite(pts).all(-1)
    psin = gs.psin_at(fp["phi"], fp["psi_axis"], fp["psi_b"], pts)
    ong = ok_pts & (pts[..., 0] > gs.R0f) & (pts[..., 0] < gs.R1f) & (pts[..., 1] > gs.Z0f) & (pts[..., 1] < gs.Z1f)
    ok = ong & torch.isfinite(te) & torch.isfinite(ne) & (te > 0) & (ne > 0) & torch.isfinite(psin)
    # private-flux / far-SOL points can read psi_N < 1: points outside the plasma polygon count as psi_N >= 1
    inside_poly = _inside(poly, pts)
    psin = torch.where(inside_poly | (psin >= 1), psin, 2 - psin.clamp(max=1))
    nb = len(BINS) - 1
    edges = torch.tensor(BINS, device=dev, dtype=torch.float32)
    okb = ok & (psin >= 0) & (psin < BINS[-1])
    bidx = (torch.bucketize(psin.clamp(0, BINS[-1] - 1e-6), edges, right=True) - 1).clamp(0, nb - 1)
    def binned(v):
        s = torch.zeros(B, nb, device=dev).scatter_add_(1, bidx, torch.where(okb, v, torch.zeros_like(v)))
        n = torch.zeros(B, nb, device=dev).scatter_add_(1, bidx, okb.float())
        return s / n.clamp(min=1), n
    te_b, nbin = binned(te); ne_b, _ = binned(ne)
    pe_b, _ = binned(te * ne * 1e19 * E_KEV)
    has = nbin > 0
    nanb = torch.full_like(te_b, float("nan"))
    def core(v):
        c = torch.where(has[:, :3], v[:, :3], nanb[:, :3]).nanmean(1)
        mx = torch.where(has, v, torch.full_like(v, -float("inf"))).max(1).values
        return torch.where(torch.isfinite(c), c, torch.where(torch.isfinite(mx), mx, torch.full_like(mx, float("nan"))))
    te0, ne0, pe0 = core(te_b), core(ne_b), core(pe_b)
    te_s, ne_s = torch.where(has, te_b / te0[:, None], nanb), torch.where(has, ne_b / ne0[:, None], nanb)
    for b in range(nb):
        f[f"te_b{b}"], f[f"ne_b{b}"] = te_s[:, b], ne_s[:, b]
    ctrs = 0.5 * (edges[1:] + edges[:-1])
    def fill(v):
        idx = torch.arange(nb, device=dev)[None].expand(B, -1)
        left = torch.cummax(torch.where(has, idx, torch.full_like(idx, -1)), 1).values
        right = torch.flip(torch.cummin(torch.flip(torch.where(has, idx, torch.full_like(idx, 2 * nb)), [1]), 1).values, [1])
        use = torch.where(left >= 0, left, right).clamp(0, nb - 1)
        return torch.where(has, v, nanb).gather(1, use)
    plasma = fp["plasma"].view(B, -1)
    psin_g = ((fp["psi_axis"][:, None] - fp["psi"]) / sw[:, None]).clamp(0, 1)
    wv = torch.where(plasma, gs.RR.reshape(1, -1).expand(B, -1), torch.zeros_like(psin_g))
    Wsum = wv.sum(1).clamp(min=1e-12)
    def vavg(prof):
        fl = fill(prof)
        j = torch.bucketize(psin_g.clamp(0, 1 - 1e-6), ctrs[:8]).clamp(1, 7)
        c0, c1 = ctrs[j - 1], ctrs[j]
        t = ((psin_g - c0) / (c1 - c0)).clamp(0, 1)
        v = torch.nan_to_num(fl.gather(1, j - 1) * (1 - t) + fl.gather(1, j) * t, nan=0.0)
        return (v * wv).sum(1) / Wsum
    pe_avg, te_avg, ne_avg = vavg(pe_b), vavg(te_b), vavg(ne_b)
    any_ts = has[:, :8].sum(1) >= 3
    Bpbar = MU0 * ip / lp
    beta_pe = 2 * MU0 * pe_avg / Bpbar ** 2
    beta_te = 2 * MU0 * pe_avg / BT0 ** 2
    nG = IpMA / (math.pi * a ** 2)
    gate = lambda v: torch.where(any_ts, v, nan)
    lg = lambda v: torch.log(v.clamp(min=1e-6))
    f["log_beta_pe"], f["log_beta_te"] = gate(lg(beta_pe)), gate(lg(beta_te))
    f["log_bq2e"] = gate(lg(beta_te * qstar ** 2 / eps))
    f["log_fgw"] = gate(lg((ne_avg / 10) / nG))
    f["log_te_peak"], f["log_ne_peak"] = gate(lg(te0 / te_avg.clamp(min=1e-6))), gate(lg(ne0 / ne_avg.clamp(min=1e-6)))
    f["log_pe_peak"] = gate(lg(pe0 / pe_avg.clamp(min=1e-12)))
    tf = torch.nan_to_num(fill(te_b), nan=0.0).clamp(min=0)
    rho = ctrs[:8].sqrt()
    jj = tf[:, :8] ** 1.5
    drho = torch.diff(torch.cat([torch.zeros(1, device=dev), rho]))
    Ir = torch.cumsum(jj * rho * drho, 1)
    Bpr = Ir / rho
    f["li_spitzer"] = gate(2 * (Bpr ** 2 * rho * drho).sum(1) / (Bpr[:, -1] ** 2).clamp(min=1e-12))
    def cross(prof_s):
        below = (prof_s < 0.3) & has
        k = torch.where(below.any(1), below.float().argmax(1), torch.full((B,), nb, device=dev, dtype=torch.long))
        return torch.where(k < nb, ctrs[k.clamp(max=nb - 1)], torch.full((B,), float(BINS[-1]), device=dev))
    f["psin_te30"], f["psin_ne30"] = gate(cross(te_s)), gate(cross(ne_s))
    f["frac_hot_out"] = gate((ok & (psin >= 1) & (ne > 0.3 * ne0[:, None])).sum(1).float() / ok.sum(1).clamp(min=1))
    f["log_n_in"] = torch.log1p((ok & (psin < 1)).sum(1).float())
    # midplane Thomson geometry (points within a quarter minor radius of the axis height)
    mid = ok & ((pts[..., 1] - Zax[:, None]).abs() < 0.25 * a[:, None])
    Rp = pts[..., 0]
    wpk = torch.where(mid, (te / te0[:, None]).clamp(min=0) ** 4, torch.zeros_like(te))
    rpk = (wpk * Rp).sum(1) / wpk.sum(1).clamp(min=1e-9)
    hot = mid & (ne > 0.2 * ne0[:, None])
    cold = mid & (ne <= 0.2 * ne0[:, None])
    rout = torch.where(hot, Rp, torch.full_like(Rp, -1e9)).max(1).values
    rin = torch.where(hot, Rp, torch.full_like(Rp, 1e9)).min(1).values
    cold_out = (cold & (Rp > rout[:, None])).any(1)
    cold_in = (cold & (Rp < rin[:, None])).any(1)
    has_pk = mid.sum(1) >= 3
    f["ts_rpk_a"] = torch.where(has_pk, (rpk - Rg) / a, nan)
    f["ts_rout_a"] = torch.where(has_pk & cold_out, (rout - Rg) / a - 1, nan)
    f["ts_rin_a"] = torch.where(has_pk & cold_in, (Rg - rin) / a - 1, nan)
    both = has_pk & cold_out & cold_in
    a_ts, r0_ts = 0.5 * (rout - rin), 0.5 * (rout + rin)
    f["ts_a_a"] = torch.where(both, a_ts / a, nan)
    f["ts_r0_a"] = torch.where(both, (r0_ts - Rg) / a, nan)
    f["ts_eps"] = torch.where(both, a_ts / r0_ts.clamp(min=1e-3), nan)
    f["ts_shift"] = torch.where(both, (rpk - r0_ts) / a_ts.clamp(min=1e-3), nan)
    # time group placeholders (filled per shot)
    for k in TIME_FEATS:
        f[k] = nan
    X = torch.stack([f[k].float() for k in NAMES], 1)
    X = torch.where(torch.isfinite(X), X, torch.full_like(X, float("nan")))
    # tokens
    bloc = 2 * MU0 * te * ne * 1e19 * E_KEV / Bpbar[:, None] ** 2
    tok = torch.stack([(pts[..., 0] - Rg[:, None]) / a[:, None], (pts[..., 1] - Zax[:, None]) / a[:, None], psin,
                       te / te0[:, None], ne / ne0[:, None], torch.log(te.clamp(min=1e-3)), torch.log(ne.clamp(min=1e-3)),
                       torch.log(bloc.clamp(min=1e-6))], -1)
    tmask = ok & torch.isfinite(tok).all(-1)
    tok = torch.where(tmask[..., None], tok, torch.zeros_like(tok))
    aux = {"te_avg": te_avg, "a": a, "kappa": kap, "R0": Rg, "bz": f["bz_norm"], "br": f["br_norm"],
           "quad": coef[:, 3] if N_HARM > 3 else nan, "flux": gs.sample(vac, ctr)[:, 0] / unit, "ip": ip}
    return X, tok, tmask, psin, aux


def _inside(poly, pts):
    """Even-odd point-in-polygon, batched: poly (B, N, 2), pts (B, M, 2) -> (B, M)."""
    r1, z1 = poly[..., 0][:, None, :], poly[..., 1][:, None, :]
    r2, z2 = torch.roll(r1, -1, 2), torch.roll(z1, -1, 2)
    rq, zq = pts[..., 0][..., None], pts[..., 1][..., None]
    crs = (z1 > zq) != (z2 > zq)
    den = torch.where((z2 - z1).abs() > 1e-12, z2 - z1, torch.full_like(z2, 1e-12))
    xint = r1 + (zq - z1) * (r2 - r1) / den
    return ((crs & (rq < xint)).sum(-1) % 2) == 1


def time_features(t_ms, aux: dict) -> dict:
    """Per-shot time group from per-frame aux arrays (numpy). Time only through tau_R-normalised rates."""
    t = np.asarray(t_ms, np.float64) / 1e3
    te = np.clip(np.nan_to_num(aux["te_avg"], nan=np.nanmedian(aux["te_avg"]) if np.isfinite(aux["te_avg"]).any() else 0.5), 0.02, None)
    a, kap = aux["a"], np.clip(aux["kappa"], 0.5, 4)
    eta = 1.65e-9 * 17.0 * 2.0 / te ** 1.5
    tauR = MU0 * a ** 2 * kap / eta
    ip = np.clip(aux["ip"], 1e4, None)
    if len(t) > 1:
        g = lambda v: np.gradient(np.asarray(v, np.float64), t)
    else:
        g = lambda v: np.zeros_like(np.asarray(v, np.float64))
    dl = g(np.log(ip))
    ew = np.zeros_like(dl)
    for k in range(len(t)):
        if k == 0:
            ew[k] = dl[k]
        else:
            alpha = 1 - np.exp(-max(t[k] - t[k - 1], 0) / max(tauR[k], 1e-3))
            ew[k] = ew[k - 1] + alpha * (dl[k] - ew[k - 1])
    out = {"log_tauR": np.log(tauR), "dlnip_tauR": np.clip(dl * tauR, -50, 50), "dlnip_ew": np.clip(ew * tauR, -50, 50),
           "ip_over_max": ip / ip.max()}
    for k, key in (("ddt_bz", "bz"), ("ddt_br", "br"), ("ddt_quad", "quad"), ("ddt_flux", "flux")):
        out[k] = np.clip(g(np.nan_to_num(aux[key])) * tauR, -50, 50)
    return out
