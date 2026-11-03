"""Label-free test-time refinement (any machine; used for MAST, where no labels may be used).

Warm start: the trained network's heads for a batch of frames. The per-frame head outputs (profile latent, warm-start
shift / stretch, vacuum-harmonic coefficients, pressure ratio) become free variables and take a few Adam steps on
the label-free losses only (GS consistency, Ampere, Thomson Te / ne collapse, Thomson edge, kinetic pressure),
plus a quadratic pull back to a prior so the refinement stays a small correction. Network weights are not changed
and no target of any kind enters.

harm_mode   "net"   harmonics on, prior = the network's values (only sensible on a machine the net was trained on)
            "off"   harmonics off (pegs --no-harm)
            "free0" harmonics on, start at zero, prior pulls to zero: a label-free per-frame fit of the low-order
                    external field (vessel currents, coil-current errors) from Thomson consistency and GS structure,
                    the same rule on every machine without labels
"""
from __future__ import annotations

import torch

from model import forward, pressure_profile
from physics import HARM_PARITY
from train import PHYS, losses, model_scalars

PRIOR = 1.0


ODD = {"warm": [1], "harm": [i for i, p in enumerate(HARM_PARITY) if p < 0]}   # Z-odd head components


def refine(net, gs, b, n_steps, iters, lr, weights=None, harm=True, harm_mode=None, prior=PRIOR, even_only=True, **fkw):
    """even_only: keep the Z-odd head components (vertical shift, odd harmonics) at the network's values. Midplane Thomson
    carries no up-down information, so freeing them lets the vertical position drift (synthetic SPARC: Z_axis R2 -1e3)."""
    if harm_mode is None:
        harm_mode = "net" if harm else "off"
    with torch.no_grad():
        h0 = net.heads(b["X"], b["tok"], b["tmask"])
        if harm_mode == "free0":
            h0["harm"] = torch.zeros_like(h0["harm"])
    free = {k: v.detach().clone().requires_grad_(True) for k, v in h0.items()}
    opt = torch.optim.Adam(free.values(), lr=lr)
    masks = {}
    if even_only:
        for k, idx in ODD.items():
            m = torch.ones_like(h0[k]); m[:, idx] = 0.0
            masks[k] = m
    w = weights or {k: 1.0 for k in PHYS}
    use_harm = harm_mode != "off"
    orig = net.heads
    try:
        net.heads = lambda *a, **k: free
        for _ in range(iters):
            st = forward(net, gs, b, n_steps=n_steps, harm=use_harm, **fkw)
            p, ff = pressure_profile(st, gs)
            st["_p"] = p
            sc = model_scalars(gs, st, b, p, ff)
            L = losses(gs, st, b, sc, 1.0, 0.0, labelled=False)
            tot = sum(w[k] * torch.nan_to_num(L[k], nan=0.0, posinf=1e3).mean() for k in w if k in L)
            tot = tot + prior * sum(((free[k] - h0[k]) ** 2).mean() for k in free)
            opt.zero_grad(set_to_none=True)
            tot.backward()
            for k, m in masks.items():
                if free[k].grad is not None:
                    free[k].grad.mul_(m)
            opt.step()
            with torch.no_grad():
                for k, m in masks.items():
                    free[k].mul_(m).add_(h0[k] * (1 - m))
        with torch.no_grad():
            st = forward(net, gs, b, n_steps=n_steps, harm=use_harm, **fkw)
            p, ff = pressure_profile(st, gs)
            st["_p"] = p
            sc = model_scalars(gs, st, b, p, ff)
    finally:
        net.heads = orig
    return st, sc
