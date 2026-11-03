"""Robust combination of several runs' predictions (same stems / frames), label-free and machine-independent.

    python ensemble_robust.py --out <dir> --preds <preds dir> ... [--mode mean|median|trim] [--dev 0.10]
mean   : plain average (ensemble.py)
median : per-cell median of psi, median of q95 / betaN
trim   : members whose magnetic axis (R_axis, Z_axis stored by predict.py) lies more than --dev metres from the member
         median axis are dropped for that frame (at least 2 members kept: the closest ones), then averaged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def combine(ds, mode, dev):
    psi = np.stack([d["psi"].astype(np.float64) for d in ds])            # (M, T, nZ, nR)
    q95 = np.stack([d["q95"].astype(np.float64) for d in ds]); bn = np.stack([d["betaN"].astype(np.float64) for d in ds])
    if mode == "mean":
        return psi.mean(0), q95.mean(0), bn.mean(0), np.zeros(psi.shape[1])
    if mode == "median":
        return np.median(psi, 0), np.median(q95, 0), np.median(bn, 0), np.zeros(psi.shape[1])
    ax = np.stack([np.stack([d["R_axis"], d["Z_axis"]], -1) for d in ds]).astype(np.float64)   # (M, T, 2)
    med = np.median(ax, 0)                                                                     # (T, 2)
    dist = np.linalg.norm(ax - med[None], axis=-1)                                             # (M, T)
    keep = dist <= dev
    M, T = keep.shape
    for t in range(T):
        if keep[:, t].sum() < 2:
            o = np.argsort(dist[:, t]); keep[:, t] = False; keep[o[:2], t] = True
    w = keep.astype(np.float64)
    wn = w / w.sum(0, keepdims=True)
    out_psi = (psi * wn[:, :, None, None]).sum(0)
    out_q = (np.where(keep, q95, 0) * wn).sum(0); out_b = (np.where(keep, bn, 0) * wn).sum(0)
    return out_psi, out_q, out_b, (M - keep.sum(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--mode", default="trim", choices=["mean", "median", "trim"])
    ap.add_argument("--dev", type=float, default=0.10)
    args = ap.parse_args()
    stems = sorted(p.stem for p in args.preds[0].glob("*.npz"))
    od = args.out / "preds"; od.mkdir(parents=True, exist_ok=True)
    dropped = 0; frames = 0
    for s in stems:
        ds = [np.load(p / f"{s}.npz") for p in args.preds]
        psi, q95, bn, nd = combine(ds, args.mode, args.dev)
        dropped += int(nd.sum()); frames += len(nd)
        assert np.isfinite(psi).all() and np.isfinite(q95).all() and np.isfinite(bn).all(), s
        extra = {k: ds[0][k] for k in ds[0].files if k not in ("psi", "q95", "betaN")}
        np.savez(od / f"{s}.npz", psi=psi.astype(np.float32), q95=q95.astype(np.float32), betaN=bn.astype(np.float32), **extra)
    (args.out / "stems.txt").write_text("\n".join(stems))
    print(f"{len(stems)} shots, {frames} frames combined ({args.mode}) over {len(args.preds)} runs -> {od}; member-frames dropped {dropped} ({dropped / max(1, frames * len(args.preds)):.3%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
