#!/usr/bin/env python3
"""Predict ψ (and the q95/βN heads) for a synthetic held-out set with a checkpoint, on CPU.

    CUDA_VISIBLE_DEVICES= [FEC_MAST_MASK=limiter] python predict_synth.py CKPT SET OUT.npz [--threads 8]

Uses the worktree's model.load_checkpoint and synth_data.SynthSet (imported, unmodified) on the held-out-only folder
SCR/sets/<SET>_ho[_limiter] (build_sets.py). ψ is SynthSet._forward's output: physical units, our sign convention
(axis at the minimum, i.e. the file's flux × −1), (N, 4225) in the set's row order. The heads come from the same net
call (scalars × scal_std + scal_mean, as model.predict_raw). Run with FEC_MAST_MASK equal to the checkpoint's region
(run_all.py sets it from config.json); load_checkpoint refuses nothing but warns on a mismatch, so this script stops.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import SCR, WT_NN  # noqa: E402

sys.path.insert(0, str(WT_NN))
import mast_region  # noqa: E402
from model import load_checkpoint  # noqa: E402
from synth_data import SynthSet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("set")
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    dev = torch.device("cpu")
    net, ck = load_checkpoint(a.ckpt, dev)
    args = ck["args"]
    trained = str(args.get("mast_mask", "envelope"))
    if mast_region.name() != trained:
        raise SystemExit(f"{a.ckpt}: trained with MAST region {trained!r}, process has {mast_region.name()!r}")
    linear = args.get("decoder", "conv") == "linear"
    ss = SynthSet(SCR / "sets" / f"{a.set}_ho", ck["norm"], dev, mast_variant=args.get("mast_variant") or "_sol0.5")
    rows = ss.rows["heldout"]
    norm = ck["norm"]
    scal_mean = torch.as_tensor(norm["scal_mean"]).float()
    scal_std = torch.as_tensor(norm["scal_std"]).float()
    psi = np.empty((len(rows), 4225), np.float32)
    q = np.empty(len(rows), np.float32)
    b = np.empty(len(rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            idx = rows[i:i + a.batch]
            # SynthSet._forward, with the scalar output kept as well
            kw = {}
            if ss.machine == "mast":
                from model import prior_from_table
                kw = dict(coord=ss.coord, prior=prior_from_table(ss.prior_table, ss.ip[idx], ss.rax[idx]), r0=ss.r0)
                if linear:
                    kw["machine"] = "mast"
            psi_n, scal_n = net(ss.coils[idx], ss.picture[idx], ss.ip[idx], ss.rax[idx], **kw)
            phys = psi_n.float() * net.psi_std + net.pixel_mean
            if i == 0:   # identical to SynthSet._forward
                ref = ss._forward(net, idx, linear)
                if not torch.equal(ref, phys):
                    raise SystemExit(f"forward differs from SynthSet._forward: {float((ref - phys).abs().max())}")
            sc = scal_n.float() * scal_std + scal_mean
            psi[i:i + len(idx)] = phys.numpy()
            q[i:i + len(idx)] = sc[:, 0].numpy()
            b[i:i + len(idx)] = sc[:, 1].numpy()
    z = np.load(SCR / "sets" / f"{a.set}_ho{mast_region.suffix()}" / "arrays.npz")
    if not np.isfinite(psi).all():
        raise SystemExit("non-finite ψ")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, psi=psi, q95_head=q, betaN_head=b, case=z["case"], frame=z["frame"])
    # pooled R²ψ in our convention, as SynthSet.evaluate (a cross-check against the scorer's R²ψ)
    y = z["psi"].astype(np.float64)
    r2 = 1 - ((psi - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"predicted {len(rows)} frames in {time.time() - t0:.0f} s; region {mast_region.name()}; "
          f"pooled R2psi (our sign) {r2:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
