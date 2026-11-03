"""Optimiser switch shared by the two trainers of ml/muon (pixel NN train_px.py, physics net upegs/train.py).

Muon (torch.optim.Muon, torch 2.11) only takes 2D weight matrices. The usual split is used: Muon for the hidden 2D weight
matrices, AdamW for everything else (biases, LayerNorm, scalars, 1D gains) and, unless asked otherwise, for the input layers
and the output heads. Each trainer supplies a `role(name, param)` function returning one of
    "hidden"  2D matrix between two hidden representations      -> Muon
    "inp"     first layer of an input path                       -> Muon only with --muon-inp 1
    "head"    output layer                                       -> Muon only with --muon-head 1
    "other"   anything else                                      -> AdamW
Muon details (torch 2.11): momentum buffer B = mu B + (1 - mu) g, Nesterov update (1 - mu) g + mu B, 5 Newton-Schulz steps
in bfloat16, decoupled weight decay p <- p (1 - lr wd) with the UNADJUSTED lr, then p <- p - lr_adj O with
lr_adj = lr sqrt(max(1, out/in)) ("original", Keller Jordan) or lr 0.2 sqrt(max(out, in)) ("match_rms_adamw", Moonshot:
update RMS 0.2 lr, so AdamW learning rates carry over).
"""
from __future__ import annotations

import torch


def add_args(ap, adam_lr_help="AdamW lr (also the AdamW part of --opt muon)"):
    g = ap.add_argument_group("optimiser (ml/muon)")
    g.add_argument("--opt", choices=["adamw", "muon"], default="adamw")
    g.add_argument("--muon-lr", type=float, default=1e-3, help="peak lr of the Muon parameter group (before the shape adjustment)")
    g.add_argument("--muon-adjust", choices=["original", "match_rms_adamw"], default="match_rms_adamw")
    g.add_argument("--muon-wd", type=float, default=0.0, help="Muon decoupled weight decay (multiplied by the unadjusted lr)")
    g.add_argument("--muon-mom", type=float, default=0.95)
    g.add_argument("--muon-nesterov", type=int, default=1)
    g.add_argument("--muon-ns-steps", type=int, default=5)
    g.add_argument("--muon-inp", type=int, default=0, help="1: input layers also get Muon")
    g.add_argument("--muon-head", type=int, default=0, help="1: output heads also get Muon")
    g.add_argument("--muon-hidden", type=int, default=1, help="0: hidden matrices stay on AdamW (control: Muon on in/out only)")
    return g


def split_params(named, role, args):
    """-> (muon params, adamw params, summary dict of parameter counts per (optimiser, role))."""
    mu, ad, summ = [], [], {}
    for n, p in named:
        if not p.requires_grad:
            continue
        r = role(n, p)
        use = p.ndim == 2 and ((r == "hidden" and args.muon_hidden) or (r == "inp" and args.muon_inp)
                               or (r == "head" and args.muon_head))
        (mu if use else ad).append(p)
        key = f"{'muon' if use else 'adamw'}:{r}"
        summ[key] = summ.get(key, 0) + p.numel()
    return mu, ad, summ


def make_muon(params, args):
    return torch.optim.Muon(params, lr=args.muon_lr, weight_decay=args.muon_wd, momentum=args.muon_mom,
                            nesterov=bool(args.muon_nesterov), ns_steps=args.muon_ns_steps, adjust_lr_fn=args.muon_adjust)


class Combo:
    """Several optimisers driven as one (zero_grad / step); schedulers are kept by the caller."""

    def __init__(self, opts):
        self.opts = [o for o in opts if o is not None]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]


def describe(summ):
    return ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in sorted(summ.items()))
