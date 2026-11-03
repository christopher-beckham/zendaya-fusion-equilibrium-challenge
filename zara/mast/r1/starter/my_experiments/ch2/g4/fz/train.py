"""Train the upegs model (supervised + label-free physics losses; any number of labelled machines).

    python train.py --out runs/u1 [--epochs 24 --steps 4 --batch 256 --width 512 --depth 2 --weighting fixed --w-phys 0.3]

Losses (each mean-normalised to O(1)):
  supervised (labelled)  psi      MSE on the raw stored psi over the whole grid
                         band     boundary displacement (psi_pred - psi_b) / |grad psi| at the scorer's true LCFS, / a
                         func     scorer functionals of the predicted psi (axis, kappa, tri, volume, li) vs the scorer
                                  functionals of the true psi
                         q95, betaN
  label-free (any)       gs       GS consistency: currents that made psi vs currents psi implies (GS residual inside,
                                  vacuum outside)
                         ampere   loop integral of B_p on the LCFS / mu0 Ip - 1
                         ts       Te and ne vs psi_N collapse to one curve (leave-one-out kernel residual)
                         tsedge   Thomson points outside psi_N 1.05 are cold, points inside 0.8 are not
                         kin      volume-averaged model pressure = ratio x volume-averaged electron pressure (Thomson),
                                  ratio learned per frame within [1, 4] (a pressure-SCALE constraint; the SHAPE of p is
                                  left to the labels: EFIT's p is not proportional to p_e)
                         kinshape pegs v1-v3 point-wise version (model p(psi_N) = ratio x p_e at every Thomson point)
                         edge     p' and FF' -> 0 at psi_N = 1 (pegs v1-v3; off by default: EFIT's DIII-D p'(1) is
                                  0.4 p'(0) at the median, so this contradicts the labels)
Weighting: --weighting kendall (learned log-variances, clamped to [--s-min, 8]) or fixed (supervised terms weight 1,
label-free terms --w-phys). Curriculum: epochs < --warm use psi, q95 and betaN only.
Guards: targets outside physical ranges are NaN-masked at load time (data.py); a step whose active loss terms or
gradient norm are non-finite is skipped (counted and logged per term; training aborts above 1 % skipped steps);
weights are checked finite before every save. Checkpoints: model.pt (last), model_best.pt (best validation R2 psi),
model_ep<k>.pt every --save-every epochs.
Lineage: copy of pegs/train.py (2026-10-05 23:10 UTC); changes listed in README.md.
ch2/syn copy (2026-10-07) of the ml/muon copy (2026-10-06 22:50 UTC, from m2/upegs_fz/train.py): --opt {adamw,muon} (../muon_opt.py). With --opt muon the hidden
2D weight matrices use torch.optim.Muon (OneCycle lr schedule with peak --muon-lr, fixed momentum unless --muon-cycle-mom 1);
biases, LayerNorms, coil gains, omega, the Kendall weights and (unless --muon-inp / --muon-head) the input layers and output
heads stay on AdamW exactly as before. Every epoch record in hist.json carries the wall-clock since start (wall_s).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muon_opt                                     # noqa: E402
from data import FrameSet, holdout_split            # noqa: E402
from embed import E_KEV                             # noqa: E402
from machine import d3d, multi_machine, MULTI_MACHINES, MU0 as _MU0   # noqa: E402
import shots as S                                   # noqa: E402
from model import PegsNet, forward, forward_kwargs, pressure_profile, N1D, ARCH_KEYS, refine_inputs   # noqa: E402
from physics import GSLayer, MU0                    # noqa: E402

SUP = ["psi", "band", "func", "q95", "betaN"]
AUX = ["harm", "prof"]     # direct supervision of the harmonic head / profile net from per-frame label fits (build_aux.py)
CONS_SCALE = None          # (7,) set by main() with --cons-scale std
HARM_SCALE_T = None        # (8,) std of the oracle harmonic coefficients over the training frames
REF_UNIT = 2.1             # Wb/rad: mu0 Ip R0 of a 1 MA, R0 = 1.7 m plasma (DIII-D). The psi loss of a machine is scaled by
PSI_UNIT2 = 1.0            # (median mu0 Ip R0 of that machine / REF_UNIT)^2 (set per set by main(); DIII-D = 1): a machine of any
                           # size counts the same, while within a machine the absolute squared error is kept (what the scorer pools)
PHYS_ALL = ["gs", "ampere", "edge", "ts", "tsedge", "kin", "kinshape", "divert"]
DIVERT = False             # ch2/syn: the "divert" term is evaluated only on machines whose frames are known to be diverted with an
                           # inner gap (set per batch: MASTSYN / MAST; the challenge frames are all diverted)
DIVERT_MARGIN = 0.02       # required flux gap (fraction of the axis - X-point flux swing) between the X-point and the inner wall
PHYS = ["gs", "ampere", "ts", "tsedge", "kin"]          # default active label-free set
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
BN_MAG_W = 0.5           # ch2/g4: set from --bn-mag-w in main()
BN_KIN_DETACH = False    # ch2/g4: --bn-kin-detach


def thomson_psin(gs, st, b):
    """psi_N of every Thomson point on the predicted equilibrium; points outside the plasma that read psi_N < 1
    (private flux / far legs) are reflected to 2 - psi_N."""
    pts = b["pts"]
    psin = gs.psin_at(st["phi"], st["psi_axis"], st["psi_b"], pts)
    with torch.no_grad():
        inside = gs.sample(st["plasma"].float(), torch.nan_to_num(pts, nan=0.0)) > 0.5
    psin = torch.where(inside | (psin >= 1), psin, 2 - psin)
    ok = b["tmask"] & torch.isfinite(psin) & torch.isfinite(pts).all(-1)
    return psin, ok


def kernel_loo(x, y, ok, h=0.05):
    """Leave-one-out Nadaraya-Watson of y on x within each frame: residuals and local densities."""
    d = (x[:, :, None] - x[:, None, :]) / h
    k = torch.exp(-0.5 * d ** 2) * ok[:, None, :].float() * ok[:, :, None].float()
    k = k * (1 - torch.eye(x.shape[1], device=x.device))[None]
    ks = k.sum(2)
    yh = (k * y[:, None, :]).sum(2) / ks.clamp(min=1e-6)
    return y - yh, ks


def model_scalars(gs, st, b, p, ff):
    """q95 (field-line integral with diamagnetic F) and betaN (kinetic from Thomson x ratio, magnetic from p)."""
    B = b["ip"].shape[0]
    Fv = MU0 * gs.tf_turns * b["itf"] / (2 * math.pi)
    i95 = 0.95 * (N1D - 1)
    j0 = int(math.floor(i95)); t = i95 - j0
    ff95 = ff[:, j0] * (1 - t) + ff[:, j0 + 1] * t
    F95 = torch.sqrt((Fv ** 2 + 2 * ff95).clamp(min=1e-6))
    q, qok = gs.q_at(st["phi"], st["R_axis"], st["Z_axis"], st["psi_axis"], st["psi_b"], F95)
    q95 = q * torch.exp(st["qcorr"])
    poly, _ = gs.contour(st["phi"], st["R_axis"], st["Z_axis"], st["psi_b"])
    geo = gs.shape(poly)
    BT0 = Fv / geo["R_geo"].clamp(min=0.05)
    # magnetic: <p>_V from the model pressure on the plasma cells
    x = st["x_cells"]
    inside = (x < 1).float() * (st["J"].detach().abs() > 0).float()
    Rk = gs.Rk[None]
    pc = gs.interp1(p, x)
    p_mag = (pc * Rk * inside).sum(1) / (Rk * inside).sum(1).clamp(min=1e-9)
    # kinetic: Thomson electron pressure on predicted psi_N, kernel-smoothed onto the plasma cells
    psin, ok = thomson_psin(gs, st, b)
    pe = torch.nan_to_num(b["ts_te"] * b["ts_ne"] * 1e19 * E_KEV, nan=0.0)
    okp = ok & (pe > 0) & (psin < 1.1)
    pe = torch.where(okp, pe, torch.zeros_like(pe))
    h = 0.08
    kk = torch.exp(-0.5 * ((x[:, :, None] - psin[:, None, :]) / h) ** 2) * okp[:, None, :].float()
    pe_c = (kk * pe[:, None, :]).sum(2) / kk.sum(2).clamp(min=1e-6)
    pe_avg = (pe_c * Rk * inside).sum(1) / (Rk * inside).sum(1).clamp(min=1e-9)
    has_ts = (okp & (psin < 0.5)).sum(1) >= 2
    a, IpMA = geo["a"], b["ip"] / 1e6
    bn = lambda pav: 100 * 2 * MU0 * pav / BT0 ** 2 * a * BT0 / IpMA
    if BN_KIN_DETACH:     # ch2/g4: the kinetic betaN route trains only the ratio head (no gradient into the geometry)
        bk_ = 100 * 2 * MU0 * st["ratio"] * pe_avg.detach() / BT0.detach() * a.detach() / IpMA
    else:
        bk_ = bn(st["ratio"] * pe_avg)
    return {"q95": q95, "q_ok": qok, "betaN_kin": bk_, "betaN_mag": bn(p_mag), "has_ts": has_ts,
            "poly": poly, "geo": geo, "psin_ts": psin, "ok_ts": ok, "p_mag": p_mag, "pe_avg": pe_avg}


def losses(gs, st, b, sc, stored_sign, stored_off, labelled=True):
    L = {}
    phi = st["phi"]
    B = phi.shape[0]
    a = b["fp_a"].clamp(min=0.05)
    geo, poly = sc["geo"], sc["poly"]
    if labelled:
        psi_st = stored_sign * st["psi"] + stored_off
        pok = torch.isfinite(b["psi"]).all(1)
        L["psi"] = torch.where(pok, ((psi_st - torch.nan_to_num(b["psi"])) ** 2).mean(1) / 1e-4 / PSI_UNIT2, torch.zeros_like(a))
        # band: displacement of the predicted boundary at the true LCFS points
        lc = b["lcfs"]
        lok = torch.isfinite(lc).all(-1)
        lcp = torch.nan_to_num(lc, nan=0.0)
        v = gs.sample(phi, lcp)
        gR, gZ = gs.grad(phi)
        g = torch.sqrt(gs.sample(gR, lcp) ** 2 + gs.sample(gZ, lcp) ** 2 + 1e-12)
        gmed = torch.where(lok, g, torch.full_like(g, float("nan"))).nanmedian(1).values
        g = torch.maximum(g, 0.3 * torch.nan_to_num(gmed, nan=1.0)[:, None])
        disp = (v - st["psi_b"][:, None]) / g / a[:, None]
        L["band"] = torch.where(lok, disp ** 2, torch.zeros_like(disp)).sum(1) / lok.sum(1).clamp(min=1) / 1e-2
        L["band"] = torch.where(lok.any(1), L["band"], torch.zeros_like(L["band"]))
        # scorer functionals
        li, _ = gs.li(phi, st["plasma"].float(), poly)
        pred = torch.stack([st["R_axis"], st["Z_axis"], geo["kappa"], geo["tri_top"], geo["tri_bot"], geo["volume"], li], 1)
        tgt = b["cons"]
        if CONS_SCALE is not None:
            # each functional in units of its spread over the training frames: the scorer weights the seven terms
            # equally through their R2, so equal weight per unit of variance (a loss weight only, not an input)
            scale = CONS_SCALE[None].expand(len(a), -1)
        else:
            scale = torch.stack([a, a, torch.full_like(a, 0.05), torch.full_like(a, 0.05), torch.full_like(a, 0.05),
                                 0.05 * tgt[:, 5].abs().clamp(min=1e-3), torch.full_like(a, 0.05)], 1)
        fok = torch.isfinite(tgt) & torch.isfinite(pred)
        e = ((torch.where(fok, pred, torch.zeros_like(pred)) - torch.nan_to_num(tgt)) / scale) ** 2
        e = torch.where(fok, e.clamp(max=100), torch.zeros_like(e))
        L["func"] = e.sum(1) / fok.sum(1).clamp(min=1)
        L["func_terms"] = e.sum(0) / fok.sum(0).clamp(min=1)
        qok = sc["q_ok"] & torch.isfinite(sc["q95"]) & torch.isfinite(b["q95"])
        qp = torch.where(qok, sc["q95"], torch.ones_like(a))
        qt = torch.where(qok, b["q95"], torch.ones_like(a))
        L["q95"] = torch.where(qok, (torch.log(qp.clamp(min=0.3)) - torch.log(qt.clamp(min=0.3))) ** 2 / 0.01, torch.zeros_like(a))
        bok = torch.isfinite(b["betaN"]) & torch.isfinite(sc["betaN_kin"]) & torch.isfinite(sc["betaN_mag"])
        bt = torch.where(bok, b["betaN"], torch.zeros_like(a))
        bk = torch.where(sc["has_ts"], sc["betaN_kin"], sc["betaN_mag"])
        bk, bm = torch.where(bok, bk, bt), torch.where(bok, sc["betaN_mag"], bt)
        if BN_MAG_W >= 0.5:
            L["betaN"] = (0.5 * (bk - bt) ** 2 + 0.5 * (bm - bt) ** 2) / 0.01
        else:
            # ch2/g4 --bn-mag-w < 0.5: less (0 = no) supervision of the model's own pressure through betaN. On gen4 the magnetic
            # route forced too much pressure into the equilibrium (plasma ~4 cm too far out); the kinetic route (Thomson p_e x the
            # learned ratio) does not move the equilibrium. Frames without core Thomson get the magnetic route at weight BN_MAG_W.
            kin_ok = sc["has_ts"] & torch.isfinite(sc["betaN_kin"])
            ek = torch.where(kin_ok & bok, (sc["betaN_kin"].nan_to_num() - bt) ** 2, torch.zeros_like(a))
            L["betaN"] = ((1 - BN_MAG_W) * ek + BN_MAG_W * (bm - bt) ** 2) / 0.01
        if "harm_star" in b:
            # harmonic head vs the per-frame least-squares correction that completes the true-current representation
            hs = b["harm_star"]
            hok = torch.isfinite(hs).all(1)
            sc_h = HARM_SCALE_T[None] if HARM_SCALE_T is not None else torch.full_like(hs, 1e-3)
            eh = ((st["harm"] - torch.nan_to_num(hs)) / sc_h) ** 2
            L["harm"] = torch.where(hok, eh.mean(1).clamp(max=100), torch.zeros_like(a))
        if "coef" in b:
            # profile shapes vs EFIT's p' (linear) and FF' (quadratic): P* = mu0 R0^2 p'(x), Q* = FF'(x) (same units as
            # the model's P : Q ratio, see model.pressure_profile); both pairs normalised by their mean absolute value
            cf = b["coef"]
            cok = torch.isfinite(cf).all(1)
            cf = torch.nan_to_num(cf)
            x = torch.linspace(0, 1, N1D, device=cf.device)[None]
            Ps = MU0 * st["R0"][:, None] ** 2 * (cf[:, 0:1] + cf[:, 1:2] * x)
            Qs = cf[:, 2:3] + cf[:, 3:4] * x + cf[:, 4:5] * x ** 2
            def unit(P, Q):
                n = 0.5 * (P.abs().mean(1) + Q.abs().mean(1)).clamp(min=1e-12)[:, None]
                return P / n, Q / n
            Pm, Qm = unit(st["P"], st["Q"]); Pt, Qt = unit(Ps, Qs)
            ep_ = ((Pm - Pt) ** 2).mean(1) + ((Qm - Qt) ** 2).mean(1)
            L["prof"] = torch.where(cok, ep_.clamp(max=100) / 1e-2, torch.zeros_like(a))
    # ---- label-free
    Jn, J = st["J_next"], st["J"]
    L["gs"] = ((Jn - J) ** 2).sum(1) / (J ** 2).sum(1).clamp(min=1e-12) / 1e-3
    _, Gl = gs.li(phi, st["plasma"].float(), poly)
    L["ampere"] = (Gl / (MU0 * b["ip"]) - 1) ** 2 / 1e-3
    P, Q = st["P"], st["Q"]
    L["edge"] = (P[:, -1] ** 2 + Q[:, -1] ** 2) / (P[:, 0] ** 2 + Q[:, 0] ** 2 + 1e-3)
    psin, ok = sc["psin_ts"], sc["ok_ts"]
    # Te, ne normalised per frame by the 90th percentile of the measured values (robust, independent of the first
    # pass), clipped to [0, 2]
    te_raw, ne_raw = b["ts_te"], b["ts_ne"]
    ok = ok & torch.isfinite(te_raw) & torch.isfinite(ne_raw) & (te_raw > 0) & (ne_raw > 0)
    def norm(v):
        vv = torch.where(ok, v, torch.full_like(v, float("nan")))
        ref = torch.nanquantile(vv, 0.9, dim=1, keepdim=True)
        ref = torch.where(torch.isfinite(ref) & (ref > 0), ref, torch.ones_like(ref))
        return torch.where(ok, (torch.nan_to_num(v) / ref).clamp(0, 2), torch.zeros_like(v))
    ten, nen = norm(te_raw), norm(ne_raw)
    okc = ok & (psin < 1.3)
    rt, kt = kernel_loo(psin, ten, okc)
    rn, _ = kernel_loo(psin, nen, okc)
    wt = (kt.clamp(max=2) / 2) * okc.float()
    L["ts"] = ((rt ** 2 + rn ** 2) * wt).sum(1) / wt.sum(1).clamp(min=1e-6) / 1e-2
    out_hot = F.relu(ten - 0.15) * (ok & (psin > 1.05)).float()
    in_cold = F.relu(0.1 - ten) * (ok & (psin < 0.8)).float()
    L["tsedge"] = ((out_hot ** 2 + in_cold ** 2).sum(1) / ok.sum(1).clamp(min=1)) / 1e-3
    # kinetic pressure: scale constraint <p>_V = ratio <p_e>_V (frames with core Thomson coverage)
    pm, pe_avg = sc["p_mag"], sc["pe_avg"]
    rk = (pm - st["ratio"] * pe_avg) / (pm.abs() + st["ratio"] * pe_avg + 1.0)
    L["kin"] = torch.where(sc["has_ts"], rk ** 2, torch.zeros_like(rk)) / 1e-2
    # pegs v1-v3 point-wise version
    p = st["_p"]
    pe = b["ts_te"] * b["ts_ne"] * 1e19 * E_KEV
    okk = ok & (psin < 0.95) & torch.isfinite(pe) & (pe > 0)
    pe = torch.where(okk, pe, torch.zeros_like(pe))
    pmp = gs.interp1(p, psin.clamp(0, 1))
    ref = (p[:, 0].abs() + 1.0)
    rks = (pmp - st["ratio"][:, None] * pe) / ref[:, None]
    L["kinshape"] = torch.where(okk, rks ** 2, torch.zeros_like(rks)).sum(1) / okk.sum(1).clamp(min=1) / 1e-2
    # ch2/syn "divert": diverted plasma with an inner gap. The flux on the inner first wall (domain rim inboard of the axis,
    # |Z - Z_axis| < 1 m) must stay below the primary X-point flux by DIVERT_MARGIN of the axis-to-X-point swing. A limited
    # (inner-wall) solution violates it; real MAST predictions were limited on the centre column in 27-39 % of frames.
    if DIVERT:
        viol, okd = divert_violation(gs, st, a)
        L["divert"] = torch.where(okd, viol ** 2, torch.zeros_like(viol)) / 1e-3
    else:
        L["divert"] = torch.zeros_like(a)
    return L


def divert_violation(gs, st, a, margin=None):
    """(B,) relu((inner-wall flux - X-point flux) / (axis - X-point flux) + margin) and a validity mask (ch2/syn "divert")."""
    phi = st["phi"]
    B = phi.shape[0]
    sv, sR, sZ, sok = gs.saddles(phi, k=6)
    ax_v = st["psi_axis"].detach()
    cand = sok & (sv.detach() < ax_v[:, None]) & ((sZ - st["Z_axis"][:, None]).abs() > 0.25 * a[:, None])
    psi_x = torch.where(cand, sv, torch.full_like(sv, -1e9)).max(1).values
    has_x = cand.any(1)
    inner = gs.rim[None] & (gs.RR[None] < st["R_axis"][:, None, None]) & ((gs.ZZ[None] - st["Z_axis"][:, None, None]).abs() < 1.0)
    rim_in = torch.where(inner, phi, torch.full_like(phi, -1e9)).view(B, -1).max(1).values
    sw_x = (ax_v - psi_x.detach()).clamp(min=1e-6)
    viol = F.relu((rim_in - psi_x) / sw_x + (DIVERT_MARGIN if margin is None else margin))
    okd = has_x & inner.view(B, -1).any(1) & torch.isfinite(viol)
    return viol, okd


class Weighting(torch.nn.Module):
    """kendall: sum_i 0.5 exp(-s_i) L_i + 0.5 s_i with learned s_i (clamped to [s_min, 8]);
    fixed: sum_sup L_i + w_aux sum_aux L_i + w_phys sum_phys L_i."""
    def __init__(self, names, mode="kendall", w_phys=0.3, s_min=-8.0, w_aux=1.0, w_sup=None):
        super().__init__()
        self.names, self.mode, self.w_phys, self.s_min, self.w_aux = names, mode, w_phys, s_min, w_aux
        self.w_sup = w_sup or {}          # fixed mode: per-term weights of the supervised terms (default 1)
        self.s = torch.nn.Parameter(torch.zeros(len(names)))

    def forward(self, L, active):
        tot = 0.0
        for i, n in enumerate(self.names):
            if n in active and n in L:
                if self.mode == "kendall":
                    s = self.s[i].clamp(self.s_min, 8)
                    tot = tot + 0.5 * torch.exp(-s) * L[n] + 0.5 * s
                else:
                    tot = tot + (self.w_sup.get(n, 1.0) if n in SUP else self.w_aux if n in AUX else self.w_phys) * L[n]
        return tot


def run_batch(net, gs, b, args, stored, labelled=True, two_pass=False):
    # ch2/g4 --mast-harm-off: no learned vacuum-harmonic correction on MAST-geometry machines (synthetic MAST has no vessel and
    # real-MAST inference runs with --no-harm, so training matches inference)
    harm = not (getattr(args, "mast_harm_off", False) and gs.m.name.startswith("MAST"))
    if two_pass:
        with torch.no_grad():
            st1 = forward(net, gs, b, n_steps=args.steps, harm=harm, **forward_kwargs(vars(args)))
        b = refine_inputs(gs, b, st1)
    st = forward(net, gs, b, n_steps=args.steps, harm=harm, **forward_kwargs(vars(args)))
    p, ff = pressure_profile(st, gs)
    st["_p"] = p
    sc = model_scalars(gs, st, b, p, ff)
    L = losses(gs, st, b, sc, *stored, labelled=labelled)
    return st, sc, L


def evaluate(net, gs, vs, args, stored, n_max=4096):
    net.eval()
    idx = torch.arange(0, len(vs), max(1, len(vs) // n_max), device=vs.dev)
    acc = {}
    with torch.no_grad():
        for i in range(0, len(idx), args.batch):
            b = vs.batch(idx[i:i + args.batch])
            st, sc, L = run_batch(net, gs, b, args, stored, two_pass=args.two_pass > 0)
            psi_st = stored[0] * st["psi"] + stored[1]
            rec = {"sse": ((psi_st - b["psi"]) ** 2).sum().item(), "n": psi_st.numel(),
                   "sum": b["psi"].sum().item(), "sum2": (b["psi"] ** 2).sum().item(),
                   "dR": (st["R_axis"] - b["cons"][:, 0]), "dZ": (st["Z_axis"] - b["cons"][:, 1]),
                   "q": torch.stack([sc["q95"], b["q95"]], 1), "bk": torch.stack([sc["betaN_kin"], sc["betaN_mag"], b["betaN"]], 1),
                   "lim": (~st["is_x"]).float().mean().item()}
            for k in SUP + AUX + PHYS_ALL:
                if k in L:
                    rec["L_" + k] = L[k].mean().item()
            rec["func_terms"] = L["func_terms"]
            for k, v in rec.items():
                acc.setdefault(k, []).append(v)
    net.train()
    n = sum(acc["n"]); mu = sum(acc["sum"]) / n
    r2 = 1 - sum(acc["sse"]) / (sum(acc["sum2"]) - n * mu ** 2)
    dR = torch.cat(acc["dR"]); dZ = torch.cat(acc["dZ"])
    rms = lambda v: float(torch.sqrt(torch.nanmean(v ** 2)) * 1e3)
    q = torch.cat(acc["q"]); bk = torch.cat(acc["bk"])
    r2s = lambda p, t: float(1 - ((p - t) ** 2).nanmean() / ((t - t.nanmean()) ** 2).nanmean())
    out = {"r2_psi": r2, "axis_dR_mm": rms(dR), "axis_dZ_mm": rms(dZ), "r2_q95": r2s(q[:, 0], q[:, 1]),
           "r2_bN_kin": r2s(bk[:, 0], bk[:, 2]), "r2_bN_mag": r2s(bk[:, 1], bk[:, 2]),
           **{k: float(np.mean(v)) for k, v in acc.items() if k.startswith("L_")}, "lim_frac": float(np.mean(acc["lim"])),
           "func_terms": {c: float(v) for c, v in zip(CONS, torch.stack(acc["func_terms"]).mean(0))}}
    return out


_INP = {"scal.0.weight", "tok.phi.0.weight", "prof.f.0.weight", "harm_net.0.weight"}
_HEAD = {"head.weight", "prof.f.4.weight", "harm_net.4.weight"}


def param_role(name: str, p) -> str:
    """muon_opt roles of the PegsNet parameters: first layer of each input path (scalar embeddings, Thomson tokens, profile
    net, coil-picture head) = inp; output layers (head outputs, profile values, harmonics) = head; other 2D = hidden."""
    if p.ndim != 2:
        return "other"
    if name in _INP:
        return "inp"
    if name in _HEAD:
        return "head"
    return "hidden"


def save(net, wgt, args, path):
    torch.save({"net": net.state_dict(), "kendall": wgt.state_dict(),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warm", type=int, default=2, help="epochs of psi/q95/betaN only")
    ap.add_argument("--stride", type=int, default=1, help="frame stride on top of the cache's own stride")
    ap.add_argument("--train-tag", default="_s2p7", help="cache of the training shots (build_cache --tag)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--val-shots", type=int, default=64)
    ap.add_argument("--frames-per-epoch", type=int, default=0, help="0 = all kept frames")
    ap.add_argument("--no-phys", action="store_true", help="supervised losses only (ablation)")
    ap.add_argument("--phys", default=",".join(PHYS), help="active label-free terms (comma list from %s)" % PHYS_ALL)
    ap.add_argument("--weighting", default="fixed", choices=["kendall", "fixed"])
    ap.add_argument("--w-phys", type=float, default=0.3, help="fixed weighting: weight of each label-free term")
    ap.add_argument("--s-min", type=float, default=-8.0, help="kendall: lower clamp of the log-variances (weight <= exp(-s_min)/2)")
    ap.add_argument("--aux", default="", help="auxiliary supervised terms from build_aux.py caches: comma list of harm,prof")
    ap.add_argument("--w-aux", type=float, default=1.0, help="fixed weighting: weight of each auxiliary term")
    ap.add_argument("--w-func", type=float, default=1.0, help="fixed weighting: weight of the scorer-functional term")
    ap.add_argument("--w-band", type=float, default=1.0, help="fixed weighting: weight of the LCFS displacement term")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--warm-bound", type=float, default=0.45, help="warm-start shift limit, units of a")
    ap.add_argument("--stretch-bound", type=float, default=0.5, help="warm-start log-stretch limit")
    ap.add_argument("--vstab", action="store_true", help="vertical stabilisation in every Picard step (v2: harmful in training)")
    ap.add_argument("--p-share", type=float, default=0.7, help="first-pass pressure share of the caches (sets the profile net's start)")
    ap.add_argument("--val-tag", default="_p7", help="cache tag of the holdout shots")
    ap.add_argument("--cons-scale", default="std", choices=["a", "std"], help="functional-loss scales: v1 ('a') or per-term std")
    # architecture (model.ARCH_KEYS; stored in the checkpoint, predict.py rebuilds from them)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=0, help="residual blocks after the two-layer trunk")
    ap.add_argument("--prof-width", type=int, default=64)
    ap.add_argument("--harm-scale", type=float, default=0.02, help="bound of the harmonic correction at the plasma edge, units of mu0 Ip R0")
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--two-pass", type=float, default=0.0, help="probability of training a step on inputs re-embedded from the model's own "
                    "equilibrium (second embedding pass); > 0 also evaluates / predicts with two passes")
    ap.add_argument("--vac-pic", type=int, default=0, help="n: external-field head fed with the exact n x n coil-flux picture (0 = pegs head)")
    ap.add_argument("--harm-fixed", type=int, default=0, help="1: harmonics about the machine centre instead of the first-pass axis")
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--multi", default="", help="synthetic training machines (comma list from %s), labelled single-frame records" % (MULTI_MACHINES,))
    ap.add_argument("--mix", type=float, default=0.2, help="fraction of training steps drawn from the --multi machines")
    ap.add_argument("--multi-tag", default="_p7", help="cache tag of the multi-machine first-pass caches (cache/upegs/<m>_synth<tag>)")
    ap.add_argument("--multi-val", type=float, default=0.05, help="held-out fraction of each multi machine for validation")
    ap.add_argument("--init", type=Path, default=None, help="warm start from this checkpoint's network weights")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--mast-share", type=float, default=0.0, help="ch2/syn: share of the --multi steps given to MASTSYN* (0 = by set size)")
    ap.add_argument("--mast-cons-mult", default="1.0", help="ch2/syn: multiplier(s) of the MASTSYN functional-loss scales (one value or 7 comma-separated)")
    ap.add_argument("--feat2", type=int, default=0, help="ch2/syn: 1 = append the feat2.NAMES2 physics features to the scalar embeddings")
    ap.add_argument("--ts-noise", type=float, default=0.0, help="ch2/g4: Gaussian noise (sd in units of each feature's training-set sd) on the "
                    "Thomson-derived scalar inputs of synthetic-MAST batches (Chris: 0.6 best on real MAST)")
    ap.add_argument("--tok-noise", type=float, default=0.0, help="ch2/g4: per-channel log-noise sd on the Te / ne token features of synthetic-MAST batches")
    ap.add_argument("--tok-cal", type=float, default=0.0, help="ch2/g4: per-frame log-calibration sd (Te and ne independently) on synthetic-MAST tokens")
    ap.add_argument("--w-betaN", type=float, default=1.0); ap.add_argument("--w-q95", type=float, default=1.0); ap.add_argument("--w-psi", type=float, default=1.0)
    ap.add_argument("--bn-mag-w", type=float, default=0.5, help="ch2/g4: weight of the magnetic (model-pressure) betaN route (0.5 = original)")
    ap.add_argument("--bn-kin-detach", action="store_true", help="ch2/g4: kinetic betaN loss trains only the ratio head")
    ap.add_argument("--ratio-max", type=float, default=4.0, help="ch2/g4: upper bound of the learned total / electron pressure ratio")
    ap.add_argument("--mast-harm-off", action="store_true", help="ch2/g4: harmonic correction off for MAST-geometry training batches")
    ap.add_argument("--exclude-stems", default="", help="ch2/g4: file of synthetic record stems to leave out of training (origin-disjoint check)")
    ap.add_argument("--mtok-drop", type=float, default=0.0, help="ch2/g4: Thomson-token dropout probability in synthetic-MAST batches")
    ap.add_argument("--ts-noise-d3d", type=float, default=0.0, help="ch2/g4: --ts-noise level for DIII-D batches (0 = clean)")
    ap.add_argument("--tok-drop", type=float, default=0.0, help="m2: probability of dropping each Thomson token in DIII-D training batches (augmentation)")
    g = muon_opt.add_args(ap)
    g.add_argument("--muon-cycle-mom", type=int, default=0, help="1: OneCycle also cycles the Muon momentum (0.85-0.95) like AdamW's beta1")
    ap.add_argument("--resume-from", type=Path, default=None, help="ml/muon: continue an interrupted run from this checkpoint (network + "
                    "loss-weight state); the lr schedule is fast-forwarded to --start-epoch, optimiser moments restart from zero")
    ap.add_argument("--start-epoch", type=int, default=0, help="first epoch to train when resuming (= completed epochs)")
    args = ap.parse_args()
    import model as _M; _M.RATIO_MAX = args.ratio_max     # ch2/g4
    global BN_MAG_W, BN_KIN_DETACH; BN_MAG_W = args.bn_mag_w; BN_KIN_DETACH = args.bn_kin_detach
    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import embed as E
    E.P_SHARE = args.p_share
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    m = d3d()
    gs = GSLayer(m, dev)
    stored = (float(m.spec.stored_sign), float(m.spec.stored_offset))
    tr, ho = holdout_split()
    tr = tr[: args.max_shots or None]
    if args.smoke:     # code / timing check only: train on holdout shots 64.. (every-frame cache), validate on 0..63
        tr, args.train_tag = ho[64: 64 + (args.max_shots or 64)], args.val_tag
    t0 = time.time()
    aux = [a_ for a_ in args.aux.split(",") if a_]
    for a_ in aux:
        assert a_ in AUX, a_
    ts = FrameSet("DIII-D", "train", tr, dev, stride=args.stride, tag=args.train_tag, aux=bool(aux))
    vs = FrameSet("DIII-D", "train", ho[: args.val_shots], dev, stride=4, tag=args.val_tag, aux=bool(aux))
    if args.feat2:
        import feat2 as F2
        F2.extend(ts, gs); F2.extend(vs, gs)
        print("feat2 (DIII-D train):", F2.describe(ts.d["X"][:, -len(F2.NAMES2):]), flush=True)
    if aux and not (ts.has_aux and vs.has_aux):
        raise SystemExit(f"--aux {aux} requested but aux caches are missing (train {ts.has_aux}, val {vs.has_aux})")
    print(f"train {len(tr)} shots {len(ts):,} frames, val {args.val_shots} shots {len(vs):,} frames; loaded in "
          f"{time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB; range-masked targets "
          f"train {ts.n_bad} val {vs.n_bad}", flush=True)
    global CONS_SCALE, PSI_UNIT2, DIVERT
    if args.cons_scale == "std":
        c = ts.d["cons"].double()
        CONS_SCALE = torch.sqrt(torch.nanmean((c - torch.nanmean(c, 0)) ** 2, 0)).float()
        print("functional-loss scales (std over training frames):", dict(zip(CONS, CONS_SCALE.tolist())), flush=True)
    # ---- synthetic training machines (labelled single frames; exact coils, no vessel)
    multi = [m_ for m_ in args.multi.split(",") if m_]
    # ch2/syn rule guard (user, 2026-10-07): synthetic MAST (MASTSYN*) and the other-machine synthetic sets may be trained on;
    # the held-out validation sets never
    bad = [m_ for m_ in multi if m_.upper().startswith("MASTVAL") or m_.upper() == "MASTG4H"]
    if bad:
        raise SystemExit(f"refusing to train on validation sets {bad}")
    msets = []
    scales = {"DIII-D": CONS_SCALE}
    units = {"DIII-D": 1.0}            # psi-loss scale (median mu0 Ip R0 / REF_UNIT)^2 per machine
    rng = np.random.RandomState(args.seed + 1)
    for mname in multi:
        assert mname in MULTI_MACHINES, mname
        import variant as V_
        mm = V_.apply_wall(multi_machine(mname))          # ch2/g4: INTR_WALL in training too (no-op unless set)
        gs_m = GSLayer(mm, dev)
        st_all = S.stems(mname, "synth")
        if args.exclude_stems:                         # ch2/g4 origin-disjoint check (diag/origin_split.py)
            _ex = set(open(args.exclude_stems).read().split())
            n0 = len(st_all); st_all = [s_ for s_ in st_all if s_ not in _ex]
            print(f"  {mname}: --exclude-stems drops {n0 - len(st_all):,} of {n0:,} records", flush=True)
        import os as _os
        if _os.environ.get("G4_SMOKE_MULTI"):          # ch2/g4 smoke test: only records whose cache exists, first N
            from build_cache import cache_dir as _cd
            st_all = [s_ for s_ in st_all if (_cd(mname, "synth", args.multi_tag) / f"{s_}.npz").exists()][: int(_os.environ["G4_SMOKE_MULTI"])]
        pm = rng.permutation(len(st_all)); nv = max(8, int(round(args.multi_val * len(st_all))))
        st_val = [st_all[i] for i in sorted(pm[:nv])]; st_tr = [st_all[i] for i in sorted(pm[nv:])]
        ts_m = FrameSet(mname, "synth", st_tr, dev, tag=args.multi_tag, aux=bool(aux))
        vs_m = FrameSet(mname, "synth", st_val, dev, tag=args.multi_tag, aux=bool(aux))
        if args.feat2:
            F2.extend(ts_m, gs_m); F2.extend(vs_m, gs_m)
            print(f"feat2 ({mname}):", F2.describe(ts_m.d["X"][:, -len(F2.NAMES2):]), flush=True)
        cm = ts_m.d["cons"].double()
        sc_m = torch.sqrt(torch.nanmean((cm - torch.nanmean(cm, 0)) ** 2, 0)).float().clamp(min=1e-4) if args.cons_scale == "std" else None
        if sc_m is not None and mname.startswith("MASTSYN") and str(args.mast_cons_mult) not in ("1", "1.0"):
            # ch2/syn: real MAST frames sit in a narrow window, so the leaderboard's R2 of each scorer functional punishes cm-level
            # errors that the synthetic spread (the default per-term scale) hides -> smaller scales = more weight on MASTSYN functionals
            m_ = torch.tensor([float(x) for x in str(args.mast_cons_mult).split(",")], device=sc_m.device)
            sc_m = sc_m * (m_ if len(m_) == len(sc_m) else m_[0])
            print(f"  {mname}: functional-loss scales x{args.mast_cons_mult} -> {[round(v, 4) for v in sc_m.tolist()]}", flush=True)
        scales[mname] = sc_m
        units[mname] = float((torch.median(_MU0 * ts_m.d["ip"] * ts_m.d["fp_R_geo"]) / REF_UNIT) ** 2)
        msets.append((mname, ts_m, vs_m, gs_m, (1.0, 0.0)))
        print(f"  {mname}: {len(ts_m):,} train / {len(vs_m):,} val frames, K={gs_m.K} cells, sym {gs_m.sym}; psi-loss unit^2 {units[mname]:.3f}; "
              f"range-masked {ts_m.n_bad}", flush=True)
    mweights = np.array([len(x[1]) for x in msets], np.float64); mweights = mweights / mweights.sum() if len(msets) else mweights
    if args.mast_share > 0 and len(msets) > 1:
        # ch2/syn: fixed share of the multi-machine steps for the synthetic MAST set(s); the other machines split the rest by size
        is_m = np.array([x[0].startswith("MASTSYN") for x in msets])
        if is_m.any() and (~is_m).any():
            w = mweights.copy()
            w[is_m] = args.mast_share * w[is_m] / w[is_m].sum(); w[~is_m] = (1 - args.mast_share) * w[~is_m] / w[~is_m].sum()
            mweights = w
    # ch2/g4: Thomson-derived scalar input columns and their training-set sd per machine (noise augmentation)
    _n = E.NAMES
    TS_COLS = torch.tensor(list(range(_n.index("log_beta_pe"), _n.index("ts_shift") + 1)), device=dev)
    def _col_sd(X):
        x = X[:, TS_COLS].double()
        mu = torch.nanmean(x, 0)
        return torch.sqrt(torch.nanmean((x - mu) ** 2, 0)).float().nan_to_num(0.0)
    TS_SD = {x[0]: _col_sd(x[1].d["X"]) for x in msets}
    TS_SD["DIII-D"] = _col_sd(ts.d["X"])
    if args.ts_noise > 0 or args.tok_noise > 0 or args.tok_cal > 0 or args.mtok_drop > 0 or args.ts_noise_d3d > 0:
        print(f"Thomson augmentation: ts_noise {args.ts_noise} (DIII-D {args.ts_noise_d3d}) on {len(TS_COLS)} scalar columns "
              f"{_n[int(TS_COLS[0])]}..{_n[int(TS_COLS[-1])]}, tok_noise {args.tok_noise}, tok_cal {args.tok_cal}, mtok_drop {args.mtok_drop}", flush=True)

    def augment_thomson(b, mname, level):
        """Noise on the network's Thomson INPUTS only (b['X'] Thomson columns, b['tok'] Te/ne features); the raw Thomson used by the
        label-free physics terms and every label stay untouched."""
        X = b["X"].clone()
        if level > 0:
            X[:, TS_COLS] = X[:, TS_COLS] + level * TS_SD[mname] * torch.randn(X.shape[0], len(TS_COLS), device=X.device)
        b["X"] = X
        if mname == "DIII-D":
            return b
        tok = b["tok"].clone()
        B, N = tok.shape[:2]
        if args.tok_noise > 0 or args.tok_cal > 0:
            e_te = args.tok_noise * torch.randn(B, N, device=tok.device); e_ne = args.tok_noise * torch.randn(B, N, device=tok.device)
            c_te = args.tok_cal * torch.randn(B, 1, device=tok.device); c_ne = args.tok_cal * torch.randn(B, 1, device=tok.device)
            tok[..., 3] = tok[..., 3] * torch.exp(e_te); tok[..., 4] = tok[..., 4] * torch.exp(e_ne)
            tok[..., 5] = tok[..., 5] + e_te + c_te; tok[..., 6] = tok[..., 6] + e_ne + c_ne
            tok[..., 7] = tok[..., 7] + e_te + c_te + e_ne + c_ne
            tok = torch.where(b["tmask"][..., None], tok, torch.zeros_like(tok))
        b["tok"] = tok
        if args.mtok_drop > 0:
            b["tmask"] = b["tmask"] & (torch.rand(b["tmask"].shape, device=tok.device) >= args.mtok_drop)
        return b
    if len(msets):
        print("multi-machine step shares (of all steps): DIII-D %.3f, " % (1 - args.mix) + ", ".join(f"{x[0]} {args.mix * w:.3f}" for x, w in zip(msets, mweights)), flush=True)
    global HARM_SCALE_T
    if "harm" in aux:
        h = ts.d["harm_star"].double()
        HARM_SCALE_T = torch.sqrt(torch.nanmean((h - torch.nanmean(h, 0)) ** 2, 0)).float().clamp(min=1e-5)
        print("harmonic-loss scales (std of the oracle coefficients):", [round(v, 5) for v in HARM_SCALE_T.tolist()], flush=True)
    phys = [] if args.no_phys else [p for p in args.phys.split(",") if p]
    for p_ in phys:
        assert p_ in PHYS_ALL, p_
    net = PegsNet(machines={"DIIID": len(m.spec.columns)}, width=args.width, depth=args.depth, prof_width=args.prof_width,
                  harm_scale=args.harm_scale, d_z=args.d_z, vac_pic=args.vac_pic, harm_fixed=args.harm_fixed, feat2=args.feat2).to(dev)
    if args.init is not None:
        ck = torch.load(args.init, map_location=dev)
        sd = dict(ck["net"])
        own = net.state_dict()
        w_old, w_new = sd.get("scal.0.weight"), own["scal.0.weight"]
        if w_old is not None and w_old.shape != w_new.shape:
            # ch2/syn: checkpoint without the feat2 group -> new input columns (values and missing indicators) start at zero,
            # so the network starts as the checkpoint's function
            n_old, n_new = w_old.shape[1] // 2, w_new.shape[1] // 2
            w = torch.zeros_like(w_new)
            w[:, :n_old] = w_old[:, :n_old]; w[:, n_new:n_new + n_old] = w_old[:, n_old:]
            sd["scal.0.weight"] = w
            sd["in_par"] = own["in_par"]
            print(f"init: scalar input layer padded {n_old} -> {n_new} features (zero columns)", flush=True)
        missing = net.load_state_dict(sd, strict=False)
        print(f"initialised from {args.init}: {missing}", flush=True)
    n_par = sum(p.numel() for p in net.parameters())
    wgt = Weighting(SUP + AUX + PHYS_ALL, args.weighting, args.w_phys, args.s_min, args.w_aux,
                    {"func": args.w_func, "band": args.w_band, "betaN": args.w_betaN, "q95": args.w_q95, "psi": args.w_psi}).to(dev)
    if args.resume_from is not None:
        ck_r = torch.load(args.resume_from, map_location=dev)
        net.load_state_dict(ck_r["net"]); wgt.load_state_dict(ck_r["kendall"])
        print(f"RESUME from {args.resume_from} at epoch {args.start_epoch} (network + loss weights; optimiser moments restart)", flush=True)
    fpe = min(args.frames_per_epoch or len(ts), len(ts))
    steps_per_epoch = fpe // args.batch
    if args.opt == "adamw":
        opt = torch.optim.AdamW([{"params": net.parameters(), "weight_decay": 1e-5},
                                 {"params": wgt.parameters(), "weight_decay": 0.0}], lr=args.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
        opt_split = {"adamw:all": n_par}
    else:
        mu_p, ad_p, opt_split = muon_opt.split_params(net.named_parameters(), param_role, args)
        o_ad = torch.optim.AdamW([{"params": ad_p, "weight_decay": 1e-5},
                                  {"params": wgt.parameters(), "weight_decay": 0.0}], lr=args.lr)
        o_mu = muon_opt.make_muon(mu_p, args)
        s_ad = torch.optim.lr_scheduler.OneCycleLR(o_ad, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
        s_mu = torch.optim.lr_scheduler.OneCycleLR(o_mu, max_lr=args.muon_lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1,
                                                   cycle_momentum=bool(args.muon_cycle_mom))
        opt = muon_opt.Combo([o_mu, o_ad])
        class _Both:
            def step(self):
                s_mu.step(); s_ad.step()
        sched = _Both()
        print(f"optimiser: Muon ({args.muon_adjust}, lr {args.muon_lr:g}, wd {args.muon_wd:g}, momentum {args.muon_mom}"
              f"{' cycled' if args.muon_cycle_mom else ''}, nesterov {bool(args.muon_nesterov)}) + AdamW (lr {args.lr:g}); "
              f"split: {muon_opt.describe(opt_split)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    print(f"network {n_par / 1e6:.2f} M parameters; aux terms {aux}; label-free terms {phys}; weighting {args.weighting}"
          f"{'' if args.weighting == 'kendall' else f' (w_phys {args.w_phys})'}; {steps_per_epoch} steps/epoch", flush=True)
    hist = []
    step = 0
    best = -1.0
    if args.resume_from is not None and args.start_epoch > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(args.start_epoch * steps_per_epoch):     # fast-forward the OneCycle schedule(s)
                sched.step()
        step = args.start_epoch * steps_per_epoch
        hp = args.out / "hist.json"
        if hp.exists():
            hist = json.loads(hp.read_text())[: args.start_epoch]
            best = max(h["r2_psi"] for h in hist) if hist else -1.0
        print(f"schedule fast-forwarded {step} steps; {len(hist)} earlier epochs kept in hist.json", flush=True)
    for ep in range(args.start_epoch, args.epochs):
        active = set(["psi", "q95", "betaN"] + aux) if ep < args.warm else set(SUP + aux + phys)
        perm = torch.randperm(len(ts), device=dev)[: steps_per_epoch * args.batch]
        te = time.time()
        run = {}
        n_skipped, bad_count = 0, {}
        for i in range(steps_per_epoch):
            if msets and rng.random_sample() < args.mix:
                j = int(rng.choice(len(msets), p=mweights))
                mname, ts_m, _, gs_b, stored_b = msets[j]
                b = ts_m.batch(torch.randint(0, len(ts_m), (args.batch,), device=dev))
                if mname.startswith("MAST") and (args.ts_noise > 0 or args.tok_noise > 0 or args.tok_cal > 0 or args.mtok_drop > 0):
                    b = augment_thomson(b, mname, args.ts_noise)
            else:
                mname, gs_b, stored_b = "DIII-D", gs, stored
                b = ts.batch(perm[i * args.batch:(i + 1) * args.batch])
                if args.ts_noise_d3d > 0:
                    b = augment_thomson(b, "DIII-D", args.ts_noise_d3d)
                if args.tok_drop > 0:        # m2: Thomson-channel dropout augmentation of the official DIII-D data (robustness to real dropouts)
                    keep = torch.rand(b["tmask"].shape, device=b["tmask"].device) >= args.tok_drop
                    b["tmask"] = b["tmask"] & keep
            CONS_SCALE, PSI_UNIT2 = scales[mname], units[mname]
            DIVERT = mname.startswith("MAST")
            st, sc, L = run_batch(net, gs_b, b, args, stored_b, two_pass=(args.two_pass > 0 and ep >= args.warm and rng.random_sample() < args.two_pass))
            Lm = {k: v.mean() for k, v in L.items() if k != "func_terms"}
            act = [k for k in Lm if k in active]
            bad_terms = [] if bool(torch.isfinite(torch.stack([Lm[k] for k in act])).all()) else \
                [k for k in act if not torch.isfinite(Lm[k])]
            opt.zero_grad(set_to_none=True)
            if bad_terms:
                # a non-finite loss term would send NaN back through backward: skip the whole step
                for k in bad_terms:
                    bad_count[k] = bad_count.get(k, 0) + 1
                n_skipped += 1; sched.step(); step += 1
                continue
            loss = wgt(Lm, active)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(wgt.parameters()), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                bad_count["grad"] = bad_count.get("grad", 0) + 1
                n_skipped += 1; sched.step(); step += 1
                continue
            opt.step(); sched.step(); step += 1
            for k, v in Lm.items():
                # speed-up: summed on the GPU in float64 (same values as the Python float sum), read back when printed
                run[k] = run[k] + v.detach().double() if k in run else v.detach().double()
            if (i + 1) % args.log_every == 0:
                run = {k: float(v) for k, v in run.items()}
                nrun = max(1, args.log_every)
                msg = " ".join(f"{k} {run[k] / nrun:.3g}" for k in SUP + AUX + PHYS_ALL if k in run)
                print(f"ep {ep} it {i + 1}/{steps_per_epoch} {time.time() - te:.0f}s | {msg} | omega {float(st['omega']):.2f}"
                      f" | skipped {n_skipped} {bad_count if bad_count else ''}", flush=True)
                run = {}
            if i >= 500 and n_skipped > 0.01 * (i + 1):       # abort only on a sustained rate (a few isolated skips are tolerated)
                raise RuntimeError(f"epoch {ep}: {n_skipped}/{i + 1} steps skipped for non-finite losses / gradients {bad_count}")
        CONS_SCALE, PSI_UNIT2 = scales["DIII-D"], units["DIII-D"]
        DIVERT = False
        ev = evaluate(net, gs, vs, args, stored)
        ev.update(epoch=ep, time_s=time.time() - te, wall_s=time.time() - t_start, opt_split=opt_split, kendall={n: float(s) for n, s in zip(wgt.names, wgt.s)},
                  n_skipped=n_skipped, bad_terms=bad_count)
        if msets:
            ev["multi"] = {}
            for mname, _, vs_m, gs_b, stored_b in msets:
                CONS_SCALE, PSI_UNIT2 = scales[mname], units[mname]
                DIVERT = mname.startswith("MAST")
                em = evaluate(net, gs_b, vs_m, args, stored_b)
                ev["multi"][mname] = {k: em.get(k) for k in ("r2_psi", "axis_dR_mm", "axis_dZ_mm", "r2_q95", "r2_bN_mag", "L_psi", "L_func", "L_divert", "lim_frac")} | {"func_terms": em["func_terms"]}
            print("   multi val: " + " | ".join(f"{m_} R2psi {v['r2_psi']:.4f} dR {v['axis_dR_mm']:.0f} dZ {v['axis_dZ_mm']:.0f} mm q95 {v['r2_q95']:.2f} bN {v['r2_bN_mag']:.2f} lim {v['lim_frac']:.3f} div {v.get('L_divert') or 0:.3g}"
                                               for m_, v in ev["multi"].items()), flush=True)
        hist.append(ev)
        print(f"== epoch {ep} ({time.time() - te:.0f}s): R2psi {ev['r2_psi']:.5f} axis dR {ev['axis_dR_mm']:.1f} dZ {ev['axis_dZ_mm']:.1f} mm "
              f"q95 {ev['r2_q95']:.3f} bN kin {ev['r2_bN_kin']:.3f} mag {ev['r2_bN_mag']:.3f} | skipped {n_skipped} | "
              + " ".join(f"{k[2:]} {v:.3g}" for k, v in ev.items() if k.startswith("L_")), flush=True)
        print("   func terms:", {k: round(v, 3) for k, v in ev["func_terms"].items()}, flush=True)
        bad_p = [n for n, p in list(net.named_parameters()) + list(wgt.named_parameters()) if not torch.isfinite(p).all()]
        if bad_p:
            raise RuntimeError(f"non-finite weights after epoch {ep}: {bad_p[:5]}; model.pt not overwritten")
        save(net, wgt, args, args.out / "model.pt")
        if ev["r2_psi"] > best:
            best = ev["r2_psi"]; shutil.copy(args.out / "model.pt", args.out / "model_best.pt")
        if args.save_every and (ep + 1) % args.save_every == 0:
            shutil.copy(args.out / "model.pt", args.out / f"model_ep{ep + 1}.pt")
        (args.out / "hist.json").write_text(json.dumps(hist, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
