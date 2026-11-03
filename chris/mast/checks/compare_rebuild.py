"""Compare a rebuilt run (retrain.py) with the original: per-epoch training metrics and the final weights.

    python chris/mast/checks/compare_rebuild.py RUN [--rebuild NAME] [--orig-ckpt FILE]

Original: /fsx/chris.beckham/github/fec-multimachine/my_experiments/nn/runs/RUN (frozen; read only), checkpoint
best_mast.pt by default (score_ep030.pt for runs without best_mast.json: pass --orig-ckpt). Rebuild: last.pt in
chris/mast/multimachine/my_experiments/nn/runs/NAME (default RUN-rebuild).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
ORIG_RUNS = Path("/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn/runs")
NEW_RUNS = HERE / "multimachine" / "my_experiments" / "nn" / "runs"
KEYS = ("train_loss", "train_psi", "r2_psi", "r2_qb", "proxy", "seconds")


def rows(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "train_loss" in r:                    # training rows (score.py appends its own rows)
            out[r["epoch"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rebuild", default=None)
    ap.add_argument("--orig-ckpt", default="best_mast.pt")
    a = ap.parse_args()
    o_dir, n_dir = ORIG_RUNS / a.run, NEW_RUNS / (a.rebuild or f"{a.run}-rebuild")
    o, n = rows(o_dir / "metrics.jsonl"), rows(n_dir / "metrics.jsonl")
    print(f"{'epoch':>5} " + " ".join(f"{k:>24}" for k in KEYS))
    for ep in sorted(n):
        cells = []
        for k in KEYS:
            x, y = o.get(ep, {}).get(k), n[ep].get(k)
            cells.append(f"{x:>11.6g} {y:>11.6g}" if x is not None and y is not None else f"{'-':>24}")
        print(f"{ep:>5} " + " ".join(cells))
    so = torch.load(o_dir / a.orig_ckpt, map_location="cpu", weights_only=False)
    sn = torch.load(n_dir / "last.pt", map_location="cpu", weights_only=False)
    print(f"checkpoint epochs: original {so.get('epoch')} ({a.orig_ckpt}), rebuild {sn.get('epoch')} (last.pt)")
    wo, wn = so["model"], sn["model"]
    assert wo.keys() == wn.keys(), "different parameter names"
    worst, num, den = 0.0, 0.0, 0.0
    for k in wo:
        if not wo[k].dtype.is_floating_point:
            continue
        d = (wo[k].double() - wn[k].double())
        worst = max(worst, d.abs().max().item())
        num += d.pow(2).sum().item(); den += wo[k].double().pow(2).sum().item()
    print(f"weights: max |difference| {worst:.3e}, relative RMS difference {(num / den) ** 0.5:.3e}, "
          f"identical: {all(torch.equal(wo[k], wn[k]) for k in wo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
