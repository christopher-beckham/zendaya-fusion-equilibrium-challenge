"""upegs model: physics embeddings -> network -> differentiable Grad-Shafranov layer -> psi(R, Z), q95, betaN.

Learned parts (all small, all dimensionless, none tied to a machine):
    encoder        scalar embeddings (embed.NAMES, soft-clipped, missing indicators) + Thomson point-cloud set
                   encoder (masked mean / max pooling over tokens) -> trunk (width x depth, LayerNorm, SiLU)
    profile net    (psi_N, z) -> P(psi_N), Q(psi_N): p' and FF' shapes, flux functions by construction
    warm start     shift / stretch of the first-pass current (in units of a)
    ext. harmonics vacuum-harmonic correction of the coil flux in units of mu0 Ip R0 (vessel currents, coil errors);
                   bounded by harm_scale at the plasma edge; applied only where the rule allows (forward(harm=...))
    scalars        small q95 correction factor, total / electron pressure ratio
    coil gains     per-coil gain per machine (only machines with labels train them; others stay at 1)
Up-down equivariance is exact: the trunk runs on the input and on its Z-mirror; even outputs are the mean,
odd outputs half the difference (parities from embed.PARITY / physics.HARM_PARITY), and the GS layer is
equivariant on the symmetrised domain.
Lineage: copy of pegs/model.py (2026-10-05 23:10 UTC) + configurable width / depth / harm_scale (net_from_args).
"""
from __future__ import annotations

import math

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import embed as E
from physics import GSLayer, MU0, N_HARM, HARM_PARITY
_WB = {}

N1D = 33
RATIO_MAX = 4.0          # ch2/g4: upper bound of the learned total / electron pressure ratio (train --ratio-max; predict reads it from the checkpoint args)


def prep_scalars(X):
    miss = ~torch.isfinite(X)
    z = 5 * torch.tanh(torch.where(miss, torch.zeros_like(X), X) / 5)
    return torch.cat([z, miss.float()], 1)


class SetEncoder(nn.Module):
    def __init__(self, d_in=E.N_TOK, width=64, d_out=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(2 * width + 1, d_out), nn.SiLU())

    def forward(self, tok, mask):
        h = self.phi(tok)
        m = mask[..., None].float()
        n = m.sum(1)
        mean = (h * m).sum(1) / n.clamp(min=1)
        mx = torch.where(m > 0, h, torch.full_like(h, -1e4)).max(1).values
        mx = torch.where(n > 0, mx, torch.zeros_like(mx))
        return self.out(torch.cat([mean, mx, torch.log1p(n)], 1))


class ProfileNet(nn.Module):
    """(psi_N, z) -> (P, Q) on N1D points; initialised at the first pass's generic profile P = P_SHARE (1 - psi_N),
    Q = 1 - psi_N."""
    def __init__(self, d_z=16, width=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1 + d_z, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
        self.register_buffer("x", torch.linspace(0, 1, N1D))

    def forward(self, z):
        B = z.shape[0]
        inp = torch.cat([self.x[None, :, None].expand(B, -1, 1), z[:, None, :].expand(-1, N1D, -1)], -1)
        o = self.f(inp)
        base = (1 - self.x)[None]
        return E.P_SHARE * base + o[..., 0], base + o[..., 1]


class Block(nn.Module):
    """Pre-norm residual MLP block."""
    def __init__(self, width):
        super().__init__()
        self.n = nn.LayerNorm(width)
        self.f = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, h):
        return h + self.f(self.n(h))


class PegsNet(nn.Module):
    HEADS = [("warm", 4, (1, -1, 1, 1)), ("harm", N_HARM, HARM_PARITY), ("qcorr", 1, (1,)), ("lratio", 1, (1,))]

    def __init__(self, n_feat=len(E.NAMES), width=256, d_z=16, machines=None, depth=0, prof_width=64, harm_scale=0.02,
                 vac_pic=0, harm_fixed=0, feat2=0):
        """depth=0 reproduces the pegs v1-v3 trunk (two plain layers); depth>0 adds residual blocks after it.
        vac_pic=n>0: the external-field (harmonic) head is a separate MLP fed with the exact coil-flux picture (n x n samples
        over the machine domain, units of mu0 Ip R_centre) and the profile latent z; the dimensionless trunk keeps the
        physics heads. harm_fixed=1: harmonics expanded about the machine centre instead of the first-pass axis."""
        super().__init__()
        self.vac_pic, self.harm_fixed, self.feat2 = int(vac_pic), int(harm_fixed), int(feat2)
        in_parity = list(E.PARITY)
        if self.feat2:            # ch2/syn: second physics-feature group appended to X (feat2.NAMES2)
            import feat2 as F2
            n_feat = n_feat + len(F2.NAMES2)
            in_parity += list(F2.PARITY2)
        if self.vac_pic:
            self.harm_net = nn.Sequential(nn.Linear(self.vac_pic ** 2 + d_z + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                          nn.Linear(256, N_HARM))
            nn.init.zeros_(self.harm_net[-1].weight); nn.init.zeros_(self.harm_net[-1].bias)
        self.scal = nn.Sequential(nn.Linear(2 * n_feat, width), nn.SiLU(), nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU())
        self.tok = SetEncoder()
        trunk = [nn.Linear(width + 128, width), nn.SiLU(), nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()]
        trunk += [Block(width) for _ in range(depth)]
        if depth > 0:
            trunk += [nn.LayerNorm(width)]
        self.trunk = nn.Sequential(*trunk)
        self.d_z = d_z
        self.harm_scale = float(harm_scale)
        n_out = d_z + sum(h[1] for h in self.HEADS)
        self.head = nn.Linear(width, n_out)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.prof = ProfileNet(d_z, prof_width)
        par = [1] * d_z
        for _, n, p in self.HEADS:
            par += list(p)
        self.register_buffer("out_par", torch.tensor(par, dtype=torch.float32))
        self.register_buffer("in_par", torch.tensor(in_parity, dtype=torch.float32))
        tp = torch.ones(E.N_TOK); tp[1] = -1
        self.register_buffer("tok_par", tp)
        self.gains = nn.ParameterDict({k: nn.Parameter(torch.zeros(n)) for k, n in (machines or {}).items()})
        self.log_omega = nn.Parameter(torch.tensor(0.0))

    def _raw(self, X, tok, tmask):
        h = torch.cat([self.scal(prep_scalars(X)), self.tok(tok, tmask)], 1)
        return self.head(self.trunk(h))

    def heads(self, X, tok, tmask, equivariant=True):
        o = self._raw(X, tok, tmask)
        if equivariant:
            om = self._raw(X * self.in_par, tok * self.tok_par, tmask)
            o = 0.5 * (o + om * self.out_par)        # even: mean; odd: half difference
        out, i = {"z": o[:, :self.d_z]}, self.d_z
        for name, n, _ in self.HEADS:
            out[name] = o[:, i:i + n]; i += n
        return out

    def gain(self, machine):
        k = machine.replace("-", "")
        return 1 + 0.05 * torch.tanh(self.gains[k]) if k in self.gains else None


ARCH_KEYS = ("width", "depth", "prof_width", "harm_scale", "d_z", "vac_pic", "harm_fixed", "feat2")
ARCH_DEFAULTS = {"width": 256, "depth": 0, "prof_width": 64, "harm_scale": 0.02, "d_z": 16, "vac_pic": 0, "harm_fixed": 0, "feat2": 0}


def net_from_args(targs: dict, machines: dict) -> PegsNet:
    """Rebuild a network from a checkpoint's training args (older pegs checkpoints: the v1-v3 architecture)."""
    kw = {k: type(ARCH_DEFAULTS[k])(targs.get(k, ARCH_DEFAULTS[k])) for k in ARCH_KEYS}
    return PegsNet(machines=machines, **kw)


def forward(net: PegsNet, gs: GSLayer, b: dict, n_steps=3, equivariant=True, extra_steps=0, harm=True,
            warm_bound=0.3, stretch_bound=0.3, vstab=False, z_pin=None):
    """b: batch dict on device (see data.py). Returns the final GS state plus model outputs.
    harm=False drops the vacuum-harmonic correction (e.g. on a machine whose vessel was never seen).
    warm_bound / stretch_bound: limits of the warm-start shift (units of a) and log-stretch; vstab: vertical
    stabilisation in every Picard step. Defaults reproduce run v1."""
    h = net.heads(b["X"], b["tok"], b["tmask"], equivariant)
    P, Q = net.prof(h["z"])
    ip, R0, Zc, a = b["ip"], b["fp_R_geo"], b["fp_Z_axis"], b["fp_a"].clamp(min=1e-2)
    unit = MU0 * ip * R0
    psi_ext = gs.vacuum(b["I"], net.gain(gs.m.name))
    if "psi_add" in b:            # ch2/syn: extra known external flux (induced passive-structure currents, predict.py --passive)
        psi_ext = psi_ext + b["psi_add"]
    if getattr(net, "vac_pic", 0):
        # exact coil-field picture in machine coordinates -> external-field head (machine-specific by nature; used only
        # where the rule allows the correction)
        Rc = gs.centre[0]
        pic = (torch.nan_to_num(b["I"]) @ gs.vac_picture_matrix(net.vac_pic)) / (MU0 * ip * Rc)[:, None]
        pic = 5 * torch.tanh(pic / 5)
        hraw = net.harm_net(torch.cat([pic, h["z"], torch.log(ip / 1e6)[:, None]], 1))
    else:
        hraw = h["harm"]
    c = getattr(net, "harm_scale", 0.02) * torch.tanh(hraw) * (1.0 if harm else 0.0)   # flux at the plasma edge, units of mu0 Ip R0
    if getattr(net, "harm_fixed", 0):
        Hb = gs.harmonics_fixed().expand(ip.shape[0], -1, -1)
        unit_h = MU0 * ip * gs.centre[0]
    else:
        Hb, unit_h = gs.harmonics_edge(R0, Zc, a), unit
    psi_ext = psi_ext + unit_h[:, None] * torch.einsum("bm,bmp->bp", c, Hb)
    key = (warm_bound, stretch_bound, ip.device)                  # speed-up: built once per device (no copy per step)
    wb = _WB.get(key)
    if wb is None:
        wb = _WB[key] = torch.tensor([warm_bound, warm_bound, stretch_bound, stretch_bound], device=ip.device)
    w = wb * torch.tanh(h["warm"])
    J0 = b["Jfp"] * ip[:, None]
    centre = torch.stack([b["fp_R_axis"], b["fp_Z_axis"]], 1)
    J0 = gs.shift_cells(J0, w[:, 0] * a, w[:, 1] * a, w[:, 2], w[:, 3], centre=centre)
    J0 = J0 * (ip / J0.sum(1).where(J0.sum(1).abs() > 1e-6, torch.ones_like(ip)))[:, None]
    omega = torch.sigmoid(net.log_omega + 2.0)
    if torch.is_grad_enabled() and psi_ext.is_cuda and not vstab and z_pin is None and os.environ.get("FZ_GRAPH_PICARD", "1") == "1":
        # speed-up: training batches replay the Picard solve from a CUDA graph (same kernels; FZ_GRAPH_PICARD=0 turns it off)
        st = gs.picard_graphed(psi_ext, J0, ip, R0, P, Q, n_steps=n_steps + extra_steps, omega=omega)
    else:
        st = gs.picard(psi_ext, J0, ip, R0, P, Q, n_steps=n_steps + extra_steps, omega=omega, vstab=vstab, z_pin=z_pin)
    st.update(P=P, Q=Q, harm=c, warm=w, qcorr=0.2 * torch.tanh(h["qcorr"][:, 0]),
              ratio=1 + (RATIO_MAX - 1) * torch.sigmoid(h["lratio"][:, 0]), psi_ext=psi_ext, R0=R0, omega=omega)
    return st


FP_KEYS = ["R_axis", "Z_axis", "psi_axis", "psi_b", "R_geo", "Z_geo", "a", "kappa", "tri_top", "tri_bot", "lp",
           "volume", "li", "res", "is_x", "xpt_R", "xpt_Z"]          # = build_cache.FP_KEYS


@torch.no_grad()
def refine_inputs(gs: GSLayer, b: dict, st: dict) -> dict:
    """Second embedding pass: recompute the dimensionless features, Thomson tokens, first-pass geometry and warm-start
    current from the model's own equilibrium `st` (the cache's first pass had 10-18 cm position errors). The per-shot time
    features are copied from the cached inputs. Returns a new batch dict."""
    fp = E.state_to_fp(gs, st, b["I"])
    X, tok, tmask, _, _ = E.frame_features(gs, fp, b["ip"], b["itf"], b["pts"], b["ts_te"], b["ts_ne"])
    idx = [E.NAMES.index(k) for k in E.TIME_FEATS]
    X[:, idx] = b["X"][:, idx]
    X = torch.where(torch.isfinite(X), X, torch.full_like(X, float("nan")))
    b2 = dict(b)
    b2["X"] = X
    N = b["tok"].shape[1]
    b2["tok"] = torch.nan_to_num(tok[:, :N].clamp(-1e4, 1e4), nan=0.0)
    b2["tmask"] = tmask[:, :N] & torch.isfinite(tok[:, :N]).all(-1)
    b2["Jfp"] = st["J"].detach() / b["ip"][:, None]
    for k in FP_KEYS:
        v = fp[k]
        b2[f"fp_{k}"] = v.float() if torch.is_tensor(v) else v
    # guard: frames whose refined equilibrium is degenerate, unconverged (relative GS residual > 0.05) or far from the cached
    # first pass (axis moved by more than half a minor radius) keep their cached inputs
    a0 = b["fp_a"].clamp(min=0.05)
    far = ((b2["fp_R_axis"] - b["fp_R_axis"]).abs() > 0.5 * a0) | ((b2["fp_Z_axis"] - b["fp_Z_axis"]).abs() > 0.5 * a0)
    bad = ~(torch.isfinite(b2["fp_a"]) & (b2["fp_a"] > 0.05) & torch.isfinite(b2["fp_R_geo"]) & fp["poly_ok"] & torch.isfinite(X).any(1)
            & torch.isfinite(fp["res"]) & (fp["res"] < 0.05)) | far
    if bool(bad.any()):
        for k in b2:
            if torch.is_tensor(b2[k]) and b2[k].shape[0] == bad.shape[0] and k in b and torch.is_tensor(b[k]) and b[k].shape == b2[k].shape:
                b2[k] = torch.where(bad.view(-1, *([1] * (b2[k].dim() - 1))), b[k], b2[k])
    return b2


def forward2(net: PegsNet, gs: GSLayer, b: dict, n_steps=3, **kw):
    """Two embedding passes: cached first pass -> equilibrium -> refined inputs -> equilibrium. Returns (st2, b2)."""
    with torch.no_grad():
        st1 = forward(net, gs, b, n_steps=n_steps, **kw)
    b2 = refine_inputs(gs, b, st1)
    return forward(net, gs, b2, n_steps=n_steps, **kw), b2


def pressure_profile(st, gs: GSLayer):
    """Model p(psi_N) (Pa) and (F^2 - F_b^2) / 2 on the N1D grid, from the profiles and the Ip normalisation s:
    a cell carries s (R / R0) P = R p' dA and s (R0 / R) Q = FF' dA / (mu0 R), so p' = s P / (R0 dA) and
    FF' = mu0 R0 s Q / dA (psi physical, Wb/rad); integrals run from the boundary inwards."""
    s, R0 = st["scale"], st["R0"]
    sw = (st["psi_axis"] - st["psi_b"]).clamp(min=1e-9)
    dx = 1.0 / (N1D - 1)
    def inward(f):
        seg = 0.5 * (f[:, 1:] + f[:, :-1]) * dx
        tail = torch.flip(torch.cumsum(torch.flip(seg, [1]), 1), [1])
        return torch.cat([tail, torch.zeros_like(tail[:, :1])], 1) * sw[:, None]
    p = inward(st["P"] * (s / (R0 * gs.area))[:, None])
    ff = inward(st["Q"] * (MU0 * R0 * s / gs.area)[:, None])
    return p, ff


def forward_kwargs(targs: dict) -> dict:
    """Forward-pass options stored in a checkpoint's training args (older checkpoints: the v1 defaults)."""
    return {"warm_bound": float(targs.get("warm_bound", 0.3)), "stretch_bound": float(targs.get("stretch_bound", 0.3)),
            "vstab": bool(targs.get("vstab", False))}
