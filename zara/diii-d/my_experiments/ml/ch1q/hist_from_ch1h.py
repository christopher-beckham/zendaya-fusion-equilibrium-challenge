"""Convert the ml/ch1h Group H per-shot cache ($FEQ_CACHE/ml/ch1h/hist/<split>/<stem>.npz, read-only) into the
ml/ch1q aligned layout: $FEQ_CACHE/ml/ch1q/hist_ch1h_{train,test}.npz (H, names, stems, t in the frame order of the
ml/ch1s input exports). Optionally appends the ml/ch1q features (--with-q) so both sets can be used together.

    python hist_from_ch1h.py [--with-q]
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
from pathlib import Path

import numpy as np

SRC = Path(f"{FEQ_CACHE}/ml/ch1h/hist")
OUT = Path(f"{FEQ_CACHE}/ml/ch1q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-q", action="store_true")
    ap.add_argument("--src", type=Path, default=SRC, help="Group H cache root (default: ml/ch1h's)")
    ap.add_argument("--split", default="", help="only this new inputs-only split (ch1h cache hist/<split>, ch1q export "
                    "inp_rfull_<split>.npz, ch1q history hist_<split>.npz) -> hist_ch1hq_<split>.npz")
    a = ap.parse_args()
    src = a.src
    parts = [(a.split, a.split)] if a.split else [("train", "diii_d_train"), ("test", "diii_d_public_test")]
    for part, sp in parts:
        exp = np.load(OUT / f"inp_rfull_{part}.npz" if a.split else f"{FEQ_CACHE}/ml/ch1s/inp_rfull_{part}.npz")
        stems = [str(s) for s in exp["stems"]]
        names = json.loads((src / sp / "meta.json").read_text())["names"]
        H, T = [], []
        for s in stems:
            with np.load(src / sp / f"{s}.npz") as d:
                H.append(d["H"].astype(np.float32)); T.append(d["t"].astype(np.float64))
        H = np.concatenate(H); T = np.concatenate(T)
        assert len(T) == len(exp["t"]) and np.allclose(T, exp["t"]), f"{part}: frame order differs"
        tag = "ch1h_"
        if a.with_q:
            Q = np.load(OUT / f"hist_{part}.npz")
            assert np.allclose(Q["t"], T)
            H = np.concatenate([H, Q["H"]], 1); names = list(names) + [f"q_{n}" for n in Q["names"]]
            tag = "ch1hq_"
        np.savez(OUT / f"hist_{tag}{part}.npz", H=H, names=np.array(names), stems=np.array(stems), t=T)
        print(f"{part}: {len(stems)} shots, {len(T):,} frames, {H.shape[1]} features -> hist_{tag}{part}.npz; "
              f"finite fraction {np.isfinite(H).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
