# Handoff: does the rt_nonoise last-epoch spike reach real MAST?

For an agent on the RTX PRO 6000 VM, which has Zara's full `/fsx` (including the MAST public-test input cache that the
A100 bundle lacks). The question is about generalisation, not speed: one member of r1, trained in the A100 rebuild,
ended on an epoch that is badly wrong on some DIII-D validation frames. Does the same failure show up in its predictions
for real MAST shots?

## Ground rules

- No MAST labels, ever. Everything below compares predictions with predictions.
- Do not change the code. Ask Chris before any code change, upload, or deletion of something you did not create.
- Never print or commit tokens.

## What happened

The A100 rebuild of r1 (`RESULTS.md` in this folder) trained all six runs with `train_r1.sh full`. The last run,
`rt_nonoise` (10 epochs, starting from this chain's `h_muon_mo`), tracked Zara's recorded run until its last epoch:

| Epoch (0-indexed) | DIII-D held-out R2psi | DIII-D psi loss | Synthetic MAST R2psi (MASTG4RT / MASTG4V3RT) | Zara's R2psi |
|---|---|---|---|---|
| 5 | 0.99798 (best) | 0.786 | | 0.99747 |
| 7 | 0.99771 | 0.889 | 0.9880 / 0.9907 | 0.99779 |
| 8 | 0.99730 | 1.04 | 0.9886 / 0.9906 | 0.99783 |
| **9 (last)** | **-3.12316** | **1,570** | **0.9886 / 0.9906** | 0.99784 |

At epoch 9 every other DIII-D held-out number stays normal (axis dR 16.6 mm, dZ 9.1 mm, R2 q95 0.948, R2 βN kinetic
0.975, limited fraction 0.0054), no batch was skipped, and the weights are finite (`train.py` checks before saving).

What this means:

- R2psi in `evaluate()` (`starter/my_experiments/ch2/g4/fz/train.py:317`) is pooled over every pixel of about 4,096
  DIII-D validation frames: 1 − total squared error / total variance. A few frames with an enormous psi error are enough
  to drive it below zero. The most likely reading is that the equilibrium solve returned a hugely wrong psi for a small
  number of DIII-D frames, while the rest are fine. Which frames, and why, is not known.
- The synthetic MAST validation is identical to epoch 8. Synthetic MAST is not real MAST, so this does not settle
  the question.
- `predict.py` loads `<run>/model.pt`, which is the last epoch. An r1 built from this chain uses the epoch-9 model.

## Why retraining will not reproduce it

Training is seeded (`seed 0`, `train.py:444`) but not deterministic on the GPU. On the A100, two `rt_ft` runs with the
same code, seed, machine and starting model already differed after one epoch (R2psi 0.99717 against 0.99730). A retrain
of `rt_nonoise` will almost certainly not spike at the same place. Use the uploaded weights for this question.

Spikes of this kind are not specific to the A100 or to the faster code. Zara's own runs with her original code show them
(her `rt_ft` falls to R2psi 0.6964 at epoch 5, her `rt_mix` to 0.9755 at epoch 1; both recover the next epoch). Across
her 74 epochs and the 74 here, 3 clear one-epoch dips after early training: roughly 2% per epoch, so roughly 1 rebuild in
8 would end with one on a kept epoch (6 kept epochs: the 4 members, `h_muon_mo` and `w3_ft`). Three events is a small
sample; read this as "not rare", not as a rate.

## The weights

The checkpoints the A100 rebuild saved for this run. They were shared privately for this check and are not published;
the result is in `RT_NONOISE_SPIKE_RESULTS.md`.

| File | Weights after (0-indexed epoch) | DIII-D held-out R2psi | SHA256 |
|---|---|---|---|
| `model.pt` | 9, last (what `predict.py` loads) | -3.12316 | `058281f2f9c3920e233e5c68b70c8bfa2495162fb9d95ea35f5fb5be5de6bea6` |
| `model_ep8.pt` | 7 | 0.99771 | `c5fe8ac6fa5d9eab03c0c99d5b47e7cb144d8cb2a573065bdf60a24f22cb12f0` |
| `model_best.pt` | 5 (best R2psi) | 0.99798 | `58a6ea6abb020b988a72d79a999de5e8a6c2bb4113912ae3d4e6db5cbf24c7d7` |
| `model_ep4.pt` | 3 | 0.99717 | `aa514cc23ed8d9f0d3ef2ccfdbe9fd76ddc3dad0a87480b02f1e7ddf5e8a3db3` |

Also `args.json` (the run's arguments), `hist.json` (per-epoch numbers, the same file as `logs/full/rt_nonoise/hist.json`
here) and `SHA256SUMS`.

`model_epN.pt` holds the weights after the N-th epoch, which is 0-indexed epoch N − 1 (`train.py:722` names it `ep + 1`,
`save_every` 4). The epoch-8 weights, just before the spike, were not saved; epoch 7 is the closest healthy checkpoint.

## Steps

### 1. Get the weights

Put the six files of "The weights" in a folder `spike_ck/` and check them against the SHA256 above (`sha256sum -c`). All six
files must report `OK`.

### 2. One folder per checkpoint

`predict.py` always loads `<run>/model.pt`, so give each checkpoint a folder with a `model.pt` link:

```bash
for c in ep9:model.pt ep7:model_ep8.pt ep5:model_best.pt ep3:model_ep4.pt; do
  n=${c%%:*}; mkdir -p ck/$n; ln -sfn $PWD/spike_ck/${c#*:} ck/$n/model.pt
done
```

### 3. Predict the MAST public test with each checkpoint

Use r1's code in this package (`zara/mast/r1/starter/my_experiments/ch2/g4/fz/predict.py`) and the same flags for
every checkpoint. If Zara's `r1_chain.sh` is available, use its member flags; otherwise the defaults are fine, because only
the difference between checkpoints matters. Run from the code folder (it imports its neighbours):

```bash
FZ=<repo>/zara/mast/r1/starter/my_experiments/ch2/g4/fz
for n in ep9 ep7 ep5 ep3; do
  (cd $FZ && $PY predict.py --run $OLDPWD/ck/$n --machine MAST --split test --out $OLDPWD/ck/$n/mast) 2>&1 | tail -2
done
```

`predict.py` reads inputs only (`labels=False`). It needs `/fsx/zara.ercan/cache/mast/mast_public_test/<shot>.npz`; on the
A100 that folder held only `basis.npz`, which is why this check could not run there. Note the "fallback frames" count in
the last line of each run: frames whose psi came out non-finite and were replaced by the first-pass equilibrium.

### 4. Compare

Per frame, the psi difference between two checkpoints relative to the frame's own psi variation:

```python
import glob, os
import numpy as np

def load(n):
    return {os.path.basename(f): np.load(f)["psi"].astype(np.float64) for f in sorted(glob.glob(f"ck/{n}/mast/preds/*.npz"))}

P = {n: load(n) for n in ("ep9", "ep7", "ep5", "ep3")}

def rel(a, b):
    """Per-frame RMS difference of psi over the grid / std of b's psi over the grid; returns (values, (shot, frame))."""
    v, where = [], []
    for k in P[b]:
        x, y = P[a][k], P[b][k]
        d = np.sqrt(((x - y) ** 2).mean((1, 2)) / y.var((1, 2)).clip(min=1e-30))
        v.append(d); where += [(k, i) for i in range(len(d))]
    return np.concatenate(v), where

for a, b in (("ep9", "ep7"), ("ep5", "ep7"), ("ep3", "ep7")):
    d, where = rel(a, b)
    top = np.argsort(d)[::-1][:5]
    print(f"{a} vs {b}: frames {len(d)}  median {np.median(d):.3g}  p99 {np.quantile(d, .99):.3g}  max {d.max():.3g}  "
          f"top {[where[i] for i in top]}")
```

How to read it:

- `ep5 vs ep7` and `ep3 vs ep7` are the baseline: how much two healthy checkpoints of the same run differ.
- If `ep9 vs ep7` looks like the baseline, including its maximum, the spike is DIII-D-only and does not affect r1's MAST
  predictions.
- If a handful of frames in `ep9 vs ep7` are far beyond the baseline's maximum, the failure reaches real MAST. List those
  shots and frames, and look at what they share (plasma current, limited or diverted, coil currents near the edge of
  the training range, Thomson coverage).

### 5. Optional: find the bad DIII-D frames

DIII-D training labels are allowed. Predicting the DIII-D hold-out with `ep9` and `ep7` and comparing each against the
labels shows which frames blew up and whether the blow-up happens again when the model is rerun (it may have been a
one-off of that evaluation pass):

```bash
(cd $FZ && $PY predict.py --run $OLDPWD/ck/ep9 --machine DIII-D --split train --subset holdout --out $OLDPWD/ck/ep9/d3d)
```

Check first that `holdout_split()` in `data.py` gives the same shots as the validation set used in training.

## Report

Write the findings next to this file (for example `RT_NONOISE_SPIKE_RESULTS.md` in a `checks-...` branch), with the
numbers from step 4, the "fallback frames" counts, and, if the failure reaches MAST, the list of affected shots and frames.

## Possible fixes (code changes: Chris decides)

Only relevant if step 4 shows that the failure reaches MAST, or if a rebuild must be dependable regardless:

- keep `model_best.pt` (best DIII-D validation R2psi; no MAST labels involved) instead of the last epoch;
- average the weights of the last few epochs;
- after training, retrain any member whose final validation R2psi falls below a threshold.
