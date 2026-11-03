"""Auxiliary per-frame targets for the DIII-D training frames (labels only, DIII-D allowed):

    python build_aux.py --tag _s2p7 [--shard 0 --n-shards 4]          # training shots, every 2nd frame (cache stride)
    python build_aux.py --tag _p7 --subset holdout                     # holdout shots, every frame

For every cached frame (same fidx as the pegs cache):
    harm  (T, 8)  least-squares coefficients of the model's 8 vacuum harmonics (physics.GSLayer.harmonics_edge about the
                  FIRST-PASS R_geo / Z_axis / a, in units of mu0 Ip R0, edge-normalised) fitted on the domain cells to
                  psi_true_phys - psi_coils - G J_true, with J_true = -Delta* psi_true / (mu0 R) inside the true LCFS
                  (ceiling.py: this correction lifts the true-current representation from S 0.971 to 0.9986)
    rms0, rms1    rms residual on the domain before / after the fit (Wb/rad)
    coef  (T, 5)  EFIT profile coefficients pc0, pc1, fc0, fc1, fc2 from xg labels_d3d (J = R (pc0 + pc1 x) +
                  (fc0 + fc1 x + fc2 x^2) / (mu0 R)), NaN where the label is unusable
-> $R1_ROOT/cache/upegs/aux_d3d<tag>/<stem>.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_cache import cache_dir                   # noqa: E402
from data import holdout_split, NN_CACHE, REFS, LABELS   # noqa: E402
from embed import _inside                           # noqa: E402
from machine import d3d, MU0                        # noqa: E402
from physics import GSLayer, N_HARM                 # noqa: E402

OUT = Path(f"{R1_ROOT}/cache/upegs")


def gs_op(psi, R, dR, dZ):
    """Delta* on (B, nZ, nR) tensors (5-point stencil, zero at the border)."""
    out = torch.zeros_like(psi)
    Rp = 0.5 * (R[1:-1] + R[2:]); Rm = 0.5 * (R[1:-1] + R[:-2])
    c = psi[:, 1:-1, 1:-1]
    out[:, 1:-1, 1:-1] = (R[None, None, 1:-1] * ((psi[:, 1:-1, 2:] - c) / Rp - (c - psi[:, 1:-1, :-2]) / Rm) / dR ** 2
                          + (psi[:, 2:, 1:-1] - 2 * c + psi[:, :-2, 1:-1]) / dZ ** 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_s2p7")
    ap.add_argument("--subset", default="train", choices=["train", "holdout"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = d3d()
    gs = GSLayer(m, dev)
    cdir = cache_dir("DIII-D", "train", args.tag)
    out = OUT / f"aux_d3d{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tr, ho = holdout_split()
    stems = (tr if args.subset == "train" else ho)[: args.max_shots or None][args.shard::args.n_shards]
    todo = [s for s in stems if not (out / f"{s}.npz").exists()]
    print(f"{len(stems)} shots, {len(todo)} to do -> {out}", flush=True)
    R = gs.R.double(); dom = gs.dom.reshape(-1)
    Gt = gs.G.double()                         # (K, P)
    Mt = gs.M.double()                         # (C, P)
    RRk = gs.RR.reshape(-1)[gs.flat].double()
    pts_grid = torch.stack([gs.RR.reshape(-1), gs.ZZ.reshape(-1)], 1)[None]       # (1, P, 2)
    t0 = time.time()
    for n, s in enumerate(todo):
        c = np.load(cdir / f"{s}.npz")
        fidx = c["fidx"]
        g = np.load(NN_CACHE["train"] / f"{s}.npz")
        r = np.load(REFS / f"{s}.npz")
        T = len(fidx)
        psi_st = torch.tensor(g["psi"][fidx], dtype=torch.float64, device=dev)
        phys = m.from_stored(psi_st)                                             # (T, nZ, nR)
        J = -gs_op(phys, R, gs.dR, gs.dZ) / (MU0 * gs.RR.double()[None])
        lc = torch.tensor(np.nan_to_num(r["lcfs"][fidx], nan=0.0), dtype=torch.float32, device=dev)   # (T, 64, 2)
        lok = torch.tensor(np.isfinite(r["lcfs"][fidx]).all(-1).all(-1), device=dev)
        inside = _inside(lc, pts_grid.expand(T, -1, -1)) & dom[None]
        Jk = torch.where(inside, J.reshape(T, -1), torch.zeros_like(J.reshape(T, -1)))[:, gs.flat] * gs.area
        I = torch.tensor(c["I"], dtype=torch.float64, device=dev)
        ip = torch.tensor(c["ip"], dtype=torch.float64, device=dev).clamp(min=1e4)
        base = I @ Mt + Jk @ Gt
        res = phys.reshape(T, -1) - base
        R0 = torch.tensor(c["fp_R_geo"], dtype=torch.float32, device=dev)
        Z0 = torch.tensor(c["fp_Z_axis"], dtype=torch.float32, device=dev)
        a = torch.tensor(c["fp_a"], dtype=torch.float32, device=dev).clamp(min=1e-2)
        unit = MU0 * ip * R0.double()
        H = gs.harmonics_edge(R0, Z0, a).double() * unit[:, None, None]           # (T, 8, P): flux per unit coefficient
        A = H[:, :, dom]                                                          # (T, 8, D)
        y = res[:, dom]
        AtA = A @ A.transpose(1, 2) + 1e-9 * torch.eye(N_HARM, device=dev, dtype=torch.float64)[None]
        coef = torch.linalg.solve(AtA, (A @ y[:, :, None]))[:, :, 0]
        fit = torch.einsum("tm,tmd->td", coef, A)
        rms0 = y.pow(2).mean(1).sqrt(); rms1 = (y - fit).pow(2).mean(1).sqrt()
        ok = lok & torch.isfinite(coef).all(1) & (Jk.sum(1) / ip > 0.8) & (Jk.sum(1) / ip < 1.25)
        coef = torch.where(ok[:, None], coef, torch.full_like(coef, float("nan")))
        prof = np.full((T, 5), np.nan, np.float32)
        lf = LABELS / f"{s}.npz"
        if lf.exists():
            lab = np.load(lf)
            prof = lab["coef"][fidx].astype(np.float32)
        tmp = out / f"{s}.tmp.npz"
        np.savez(tmp, harm=coef.float().cpu().numpy(), rms0=rms0.float().cpu().numpy(), rms1=rms1.float().cpu().numpy(),
                 iin=(Jk.sum(1) / ip).float().cpu().numpy(), coef=prof, fidx=fidx)
        os.replace(tmp, out / f"{s}.npz")
        if n % 200 == 0:
            print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s  last shot: rms before {rms0.mean():.4f} after {rms1.mean():.4f} Wb/rad; "
                  f"ok {ok.float().mean():.2f}; coef median {np.round(np.nanmedian(coef.cpu().numpy(), 0), 4).tolist()}", flush=True)
    print(f"done {len(todo)} shots in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
