"""Fold lists for out-of-fold (OOF) psi maps.

Holdout experiments (design 'B', shared by both splits): R = all training shots outside H0 u H42 (the two 704-shot holdouts);
K = 5 folds over R (RandomState(777)). Fold net i trains on R \\ F_i and predicts F_i plus A_i, a fifth of H0 u H42
(round-robin). Every shot thus gets one OOF map from a network that saw neither its own shot nor any holdout shot of either split.
Production ('P'): K = 5 folds over all 7,041 shots; fold net i trains on the rest and predicts its fold.
-> lists/oof{i}_train.txt, lists/oof{i}_pred.txt, lists/poof{i}_train.txt, lists/poof{i}_pred.txt, lists/folds.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1s"))
from s_common import all_stems, split  # noqa: E402

K = 5
L = HERE / "lists"
stems = all_stems()
h0 = set(split(0)[1]); h42 = set(split(42)[1])
hh = sorted(h0 | h42)
rest = [s for s in stems if s not in h0 and s not in h42]
rng = np.random.RandomState(777)
pr = rng.permutation(len(rest)); ph = rng.permutation(len(hh))
fold_r = np.empty(len(rest), int); fold_r[pr] = np.arange(len(rest)) % K
fold_h = np.empty(len(hh), int); fold_h[ph] = np.arange(len(hh)) % K
info = {"n_all": len(stems), "n_h0": len(h0), "n_h42": len(h42), "n_union": len(hh), "n_rest": len(rest), "folds": []}
for i in range(K):
    tr = [s for s, f in zip(rest, fold_r) if f != i]
    pdl = sorted([s for s, f in zip(rest, fold_r) if f == i] + [s for s, f in zip(hh, fold_h) if f == i])
    (L / f"oof{i}_train.txt").write_text("\n".join(tr) + "\n"); (L / f"oof{i}_pred.txt").write_text("\n".join(pdl) + "\n")
    info["folds"].append({"train": len(tr), "pred": len(pdl)})
pp = np.random.RandomState(778).permutation(len(stems))
fold_p = np.empty(len(stems), int); fold_p[pp] = np.arange(len(stems)) % K
for i in range(K):
    tr = [s for s, f in zip(stems, fold_p) if f != i]; pdl = [s for s, f in zip(stems, fold_p) if f == i]
    (L / f"poof{i}_train.txt").write_text("\n".join(tr) + "\n"); (L / f"poof{i}_pred.txt").write_text("\n".join(pdl) + "\n")
info["prod_folds"] = [int((fold_p == i).sum()) for i in range(K)]
(L / "folds.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info))
