# Rebuilds of Chris's 30 training runs

Each run retrained from scratch with `retrain.py` (frozen `train_mm.py`, stopped at the checkpoint's epoch, no scoring jobs), one RTX PRO 6000 per run. Synthetic-MAST S: `ensemble_synth.py --sets g4_20000`, the scorer the original runs used. Originals: the g4_20000 score recorded in `best_mast.json`; for the four runs without one (`score_ep030.pt`) and the one run selected on another set (W4-v4only, `g4_v4`), the original checkpoint scored the same way.

| Run | Stop epoch | Wall clock | Peak GPU memory | Synthetic-MAST S original | Rebuild | Difference |
|---|---|---|---|---|---|---|
| `scr-C1-z9cond-seed0` | 12 | 32.1 min | 47.4 GB | 0.850431 | 0.850424 | -0.000007 |
| `scr-C2-z9cond-seed1` | 15 | 38.6 min | 47.4 GB | 0.850090 | 0.850062 | -0.000028 |
| `scr-C3-condonly` | 6 | 12.1 min | 40.1 GB | 0.744942 | 0.744932 | -0.000010 |
| `scr-C6-z9-condw1` | 9 | 26.8 min | 51.9 GB | 0.851851 | 0.851731 | -0.000120 |
| `scr-E1-20k5k-ema-shape` | 45 | 73.0 min | 43.1 GB | 0.859139 | 0.859073 | -0.000066 |
| `scr-E3-20k5k-ema-shape-seed1` | 40 | 64.9 min | 43.1 GB | 0.858684 | 0.858669 | -0.000015 |
| `scr-E5-20k5k-ema-shape-seed2` | 40 | 65.3 min | 43.1 GB | 0.859324 | 0.859499 | +0.000175 |
| `scr-E6-20k5k-ema-shape2x` | 40 | 64.7 min | 43.1 GB | 0.860234 | 0.860119 | -0.000115 |
| `scr-F1-final-g4x5000` | 30 | 47.1 min | 39.5 GB | 0.833022 | 0.830018 | -0.003004 |
| `scr-F1-z15mix-s0` | 12 | 31.9 min | 48.2 GB | 0.851842 | 0.851996 | +0.000154 |
| `scr-F2-final-g4x5000-seed1` | 30 | 47.1 min | 39.5 GB | 0.826858 | 0.825569 | -0.001289 |
| `scr-F2-z15mix-s1` | 12 | 31.8 min | 48.2 GB | 0.850529 | 0.850576 | +0.000047 |
| `scr-F3-only` | 9 | 18.4 min | 40.1 GB | 0.738107 | 0.738572 | +0.000465 |
| `scr-F6-z15mix-s2` | 12 | 32.7 min | 48.2 GB | 0.851362 | 0.851387 | +0.000025 |
| `scr-G1-g4x5000-noise0.3` | 30 | 46.2 min | 39.5 GB | 0.829605 | 0.830334 | +0.000729 |
| `scr-G4-g4x5000-clip1` | 30 | 46.3 min | 39.6 GB | 0.829087 | 0.828658 | -0.000429 |
| `scr-V1-v4only` | 9 | 16.4 min | 39.7 GB | 0.802192 | 0.801804 | -0.000388 |
| `scr-V2-20k5kv4` | 12 | 29.5 min | 44.4 GB | 0.856836 | 0.856850 | +0.000014 |
| `scr-V7C-v7only` | 9 | 17.1 min | 38.8 GB | 0.785740 | 0.785225 | -0.000515 |
| `scr-W1-20k5kv4-seed1` | 12 | 29.0 min | 44.4 GB | 0.856539 | 0.856623 | +0.000084 |
| `scr-W2-20k5kv4-seed2` | 12 | 29.6 min | 44.4 GB | 0.856640 | 0.856673 | +0.000033 |
| `scr-W3-20k5kv4-seed3` | 15 | 36.4 min | 44.4 GB | 0.856894 | 0.856800 | -0.000094 |
| `scr-W4-20k5kv4-seed4` | 12 | 29.4 min | 44.4 GB | 0.857194 | 0.857070 | -0.000124 |
| `scr-W4-v4only-ep12-seed1` | 9 | 16.3 min | 39.7 GB | 0.804742 | 0.804766 | +0.000024 |
| `scr-W5-20k5kv4-seed5` | 15 | 35.3 min | 44.4 GB | 0.856309 | 0.856328 | +0.000019 |
| `scr-W6-20k5kv4-seed6` | 15 | 36.1 min | 44.4 GB | 0.856569 | 0.856504 | -0.000065 |
| `scr-mastZ15-v6mix-seed0` | 15 | 37.1 min | 44.8 GB | 0.856910 | 0.856960 | +0.000050 |
| `scr-mastZ15-v6mix-seed1` | 12 | 30.2 min | 44.8 GB | 0.856477 | 0.856516 | +0.000039 |
| `scr-mastZ15-v6mix-seed2` | 12 | 30.0 min | 44.8 GB | 0.857009 | 0.856882 | -0.000127 |
| `scr-mastZ15-v6mix-seed3` | 12 | 29.1 min | 44.8 GB | 0.856510 | 0.856519 | +0.000009 |

Across the 30 runs: mean difference -0.000151, largest absolute difference 0.003004. The organisers allow 0.01 S between a retrained entry and the submitted score.

## Training time per model and per blend (wall clock of the training commands)

| Model | Runs | Training time |
|---|---|---|
| Z | 1 | 0.78 h |
| Z2 | 1 | 0.77 h |
| Z3 | 4 | 3.11 h |
| Z6b | 5 | 5.25 h |
| Z7 | 4 | 2.07 h |
| Z8 | 2 | 0.55 h |
| Z9qb | 7 | 3.75 h |
| Z15 | 4 | 2.11 h |
| Z17 | 1 | 0.20 h |
| Z18 | 3 | 1.62 h |
| FA | 3 | 1.61 h |
| FC | 1 | 0.31 h |
| V7C | 1 | 0.28 h |

Shared runs count once per blend (Z3 shares two runs with Z and Z2, Z9qb holds Z7's four, Z6b needs Z's run for its q95). Chris's part only; Zara's members are not included.

| Blend | Chris's runs | Chris's training time |
|---|---|---|
| One 24 h budget (B16+FC) | 10 | 5.59 h |
| 24 h per challenge (B16+V7C) | 11 | 5.77 h |
| Maximum | 29 | 17.80 h |
