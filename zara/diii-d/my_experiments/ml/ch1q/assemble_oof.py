"""Assemble the per-fold out-of-fold map features (qcurve_oof.py outputs) into one file aligned with the training export.

    python assemble_oof.py --prefix oof     (folds oof0..oof4 -> mapf_oof.npz)   |   --prefix poof (production folds)
Every training shot must appear in exactly one fold file. Output: Q (N, 23), E (N, 11), S (N, 2), levels, extra, stems (all 7,041
sorted stems, frame order of $FEQ_CACHE/ml/ch1s/inp_r0_train.npz).
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
from pathlib import Path

import numpy as np

OUT = Path(f"{FEQ_CACHE}/ml/ch1q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    D = np.load(f"{FEQ_CACHE}/ml/ch1s/inp_rfull_train.npz")   # (package) same frame order as inp_r0 (all training shots, sorted)
    stems = [str(s) for s in D["stems"]]
    nT = np.bincount(D["shot"], minlength=len(stems))
    off = np.r_[0, np.cumsum(nT)]
    N = off[-1]
    Q = np.full((N, 23), np.nan); E = np.full((N, 11), np.nan); S = np.full((N, 2), np.nan)
    seen = np.zeros(len(stems), int)
    sidx = {s: i for i, s in enumerate(stems)}
    for i in range(a.k):
        f = np.load(OUT / f"qc_{a.prefix}{i}.npz")
        o = 0
        for s in f["stems"]:
            j = sidx[str(s)]
            n = nT[j]
            Q[off[j]:off[j + 1]] = f["Q"][o:o + n]; E[off[j]:off[j + 1]] = f["E"][o:o + n]; S[off[j]:off[j + 1]] = f["S"][o:o + n]
            seen[j] += 1
            o += n
        assert o == len(f["Q"]), f"fold {i}: frame count mismatch"
        levels, extra = f["levels"], f["extra"]
    assert (seen == 1).all(), f"shots seen {np.bincount(seen)}"
    out = OUT / f"mapf_{a.prefix}.npz"
    np.savez(out, Q=Q, E=E, S=S, levels=levels, extra=extra, stems=np.array(stems))
    ok = E[:, -1] == 1
    print(f"{len(stems)} shots, {N:,} frames -> {out}; ok fraction {ok.mean():.4f}; finite q(0.95) {np.isfinite(Q[:, 19]).mean():.4f}")


if __name__ == "__main__":
    main()
