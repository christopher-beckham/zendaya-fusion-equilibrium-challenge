"""ml/ch1s shared helpers: holdout splits, frame tables, pooled R2 (scorer definition), reference prediction sets.

Split rule = c1/nn/score_holdout.py / train_c1.py: 10 % of the sorted parquet stems, np.random.RandomState(seed).permutation.
Pooled R2 over every finite label of the split (the scorer: finite labels only, non-finite predictions -> fold mean).
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CH1 = HERE.parent / "ch1"
MUON = HERE.parent / "muon"
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"
CACHE = Path(f"{FEQ_CACHE}/nn/diii_d_train")
TEST_CACHE = Path(f"{FEQ_CACHE}/nn/diii_d_public_test")
OUTC = Path(f"{FEQ_CACHE}/ml/ch1s")

# production recipe (gb1 = gradient loss lam 1.0, Muon wd 0.1) holdout members per split
REF_MEMBERS = {
    0: [CH1 / "runs" / r / "preds" for r in ("s1_gb1", "gb1_s2", "gb1_s3", "gb1_s4", "gb1_s5", "gb1_s6")],
    42: [CH1 / "runs" / f"s42_gb1_s{k}" / "preds" for k in range(1, 7)],
}
REF_ENS = {0: CH1 / "runs" / "ens_gb1_6" / "preds", 42: CH1 / "runs" / "s42_gb1_6" / "preds"}
V6_ENS = {0: CH1 / "runs" / "v6_12" / "preds", 42: CH1 / "runs" / "s42_v6_12" / "preds"}
PROD_MEMBERS = [CH1 / "runs" / f"prod_gb1_s{k}" / "preds_test" for k in range(1, 7)]


def all_stems():
    return [f.stem for f in sorted(DATA_DIR.glob("*.parquet"))]


def split(seed: int, frac: float = 0.10):
    """(train stems, holdout stems) exactly as train_c1.split_stems / score_holdout.split_stems."""
    stems = all_stems()
    if frac <= 0:
        return stems, []
    perm = np.random.RandomState(seed).permutation(len(stems))
    hold = set(perm[: int(round(frac * len(stems)))].tolist())
    return [s for i, s in enumerate(stems) if i not in hold], [s for i, s in enumerate(stems) if i in hold]


def test_stems():
    return sorted(p.stem for p in TEST_CACHE.glob("*.npz") if p.stem not in ("basis", "meta"))


def frame_table(stems, cache: Path = CACHE, with_labels: bool = True):
    """Per-frame metadata (+ labels) of a list of shots, concatenated in stem order."""
    rec = {k: [] for k in ("shot", "k", "T", "t", "ip", "bt")}
    if with_labels:
        rec["q95"], rec["betaN"] = [], []
    for i, s in enumerate(stems):
        with np.load(cache / f"{s}.npz") as d:
            T = len(d["t"])
            rec["shot"].append(np.full(T, i)); rec["k"].append(np.arange(T)); rec["T"].append(np.full(T, T))
            rec["t"].append(d["t"]); rec["ip"].append(d["ip"]); rec["bt"].append(d["bt"])
            if with_labels:
                rec["q95"].append(d["q95"]); rec["betaN"].append(d["betaN"])
    return {k: np.concatenate(v).astype(np.float64) if k not in ("shot", "k", "T") else np.concatenate(v)
            for k, v in rec.items()}


def load_scalars(pdir: Path, stems):
    """(N, 2) q95 / betaN of a preds directory (npz per shot), concatenated in stem order."""
    out = []
    for s in stems:
        with np.load(Path(pdir) / f"{s}.npz") as p:
            out.append(np.column_stack([p["q95"], p["betaN"]]).astype(np.float64))
    return np.concatenate(out)


def r2(y, p):
    m = np.isfinite(y)
    pp = np.where(np.isfinite(p[m]), p[m], y[m].mean())
    return 1.0 - np.sum((pp - y[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)


def r2qb(Y, P):
    """(q95 R2, betaN R2, mean) for (N, 2) labels / predictions."""
    a, b = r2(Y[:, 0], P[:, 0]), r2(Y[:, 1], P[:, 1])
    return a, b, 0.5 * (a + b)


def fmt(Y, P):
    a, b, c = r2qb(Y, P)
    return f"q95 {a:.5f}  betaN {b:.5f}  qb {c:.6f}"


def sse_red(Y, P, Pref):
    """Relative squared-error change of P vs Pref for q95, betaN and the qb-weighted sum (qb SSE = mean of SSE/SST)."""
    out = []
    for j in range(2):
        m = np.isfinite(Y[:, j])
        out.append(np.sum((P[m, j] - Y[m, j]) ** 2) / np.sum((Pref[m, j] - Y[m, j]) ** 2) - 1)
    a0, b0, c0 = r2qb(Y, Pref)
    a1, b1, c1 = r2qb(Y, P)
    out.append((1 - c1) / (1 - c0) - 1)
    return out


def write_preds(pdir: Path, stems, tab_shot, P, extra=None):
    """Per-shot npz with q95 / betaN only (float32), the layout ml/ch1 uses for its preds (minus psi)."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(P).all(), "non-finite scalar predictions; nothing written"
    for i, s in enumerate(stems):
        m = tab_shot == i
        d = {"q95": P[m, 0].astype(np.float32), "betaN": P[m, 1].astype(np.float32)}
        if extra is not None:
            d.update({k: v[m] for k, v in extra.items()})
        np.savez(pdir / f"{s}.npz", **d)
