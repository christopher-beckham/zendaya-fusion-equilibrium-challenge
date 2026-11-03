# Chris's MAST models

The 13 models that Zara's team blends use. Each blend reads the model's finished prediction file
(`mast_public_test.npz`), so everything below up to and including post-processing belongs to the model.

| Model | Used in blend | Final prediction folder (`submissions/` on `/fsx`) |
|---|---|---|
| Z15 | B16+FC, B16+V7C, maximum | `mastZ15-mm-v6mix-x4qb-q95cc0932-bnV2-limiter-c1` |
| Z18 | B16+FC, B16+V7C, maximum | `mastZ18-mm-Z9dataPlusCond-x3qb-q95cc0932-bnV2-limiter-c1` |
| Z17 | B16+V7C | `mastZ17-mm-C3-condgsolveonlyqb-q95cc0932-bnV2-limiter-c1` |
| Z | B16+FC, B16+V7C, maximum | `mastZ-mm-F1-g4x5000-limiter-ep030-c1` |
| Z2 | B16+FC, B16+V7C, maximum | `mastZ2-mm-G1-g4x5000-noise03-limiter-ep030-c1` |
| FC | B16+FC, maximum | `mastFC-mm-gsolve_cond3_1000onlyqb-q95cc0932-bnV2-limiter-c1` |
| V7C | B16+V7C, maximum | `mastV7C-mm-V7onlyqb-q95cc0932-bnV2-limiter-c1` |
| Z3 | maximum | `mastZ3-mm-ens4-F1F2G1G4-limiter-ep030-c1` |
| Z6b | maximum | `mastZ6b-mm-E6E1E3E5-qF1-limiter-c1` |
| Z7 | maximum | `mastZ7-mm-V2W123-20k5kv4-limiter-c1` |
| Z8 | maximum | `mastZ8-mm-V1W4-v4only-limiter-c1` |
| Z9qb | maximum | `mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1` |
| FA | maximum | `mastFA-mm-Z15dataPlusgsolve_cond3_1000-x3qb-q95cc0932-bnV2-limiter-c1` |

The local copies of the Z, Z2, Z3, Z6b, Z7 and Z8 prediction files were removed from `/fsx`; the blends read the copies
fetched back from Hugging Face.

## Training runs

Run records (`config.json`, `best_mast.json`, `metrics.jsonl`) are in `run_records/<run>/`. `config.json` holds every
argument, so each run is `python my_experiments/nn/train_mm.py` with those arguments, run from `multimachine/`.

| Model | Runs | Checkpoint used | Training caches (`--synth`) |
|---|---|---|---|
| Z | `scr-F1-final-g4x5000` | `score_ep030.pt` (45 epochs scheduled) | `zara_g4_5000` |
| Z2 | `scr-G1-g4x5000-noise0.3` | `score_ep030.pt` (45 scheduled) | `zara_g4_5000` |
| Z3 | `scr-F1-final-g4x5000`, `scr-F2-final-g4x5000-seed1`, `scr-G1-g4x5000-noise0.3`, `scr-G4-g4x5000-clip1` | `score_ep030.pt` each | `zara_g4_5000` |
| Z6b | `scr-E6-20k5k-ema-shape2x`, `scr-E1-20k5k-ema-shape`, `scr-E3-20k5k-ema-shape-seed1`, `scr-E5-20k5k-ema-shape-seed2` (maps, βN); q95 from Z | `best_mast.pt` (epochs 40, 45, 40, 40 of 45) | `zara_g4_20k5k` |
| Z7 | `scr-V2-20k5kv4`, `scr-W1/W2/W3-20k5kv4-seed1..3` | `best_mast.pt` (12, 12, 12, 15 of 20) | `zara_g4_20k5kv4` |
| Z8 | `scr-V1-v4only`, `scr-W4-v4only-ep12-seed1` | `best_mast.pt` (9 of 20, 9 of 12) | `zara_g4_5000v4` |
| Z9qb | Z7's four runs + `scr-W4/W5/W6-20k5kv4-seed4..6` | `best_mast.pt` (12, 15, 15 for the extra three) | `zara_g4_20k5kv4` |
| Z15 | `scr-mastZ15-v6mix-seed0..3` | `best_mast.pt` (15, 12, 12, 12 of 20) | `zara_g4_mixv6` |
| Z17 | `scr-C3-condonly` | `best_mast.pt` (6 of 20) | `gsolve_cond1000` |
| Z18 | `scr-C1-z9cond-seed0`, `scr-C2-z9cond-seed1`, `scr-C6-z9-condw1` | `best_mast.pt` (12, 15, 9 of 20) | C1, C2: `zara_g4_20k5kv4cond`; C6: `zara_g4_20k5kv4` + `gsolve_cond1000` |
| FA | `scr-F1-z15mix-s0`, `scr-F2-z15mix-s1`, `scr-F6-z15mix-s2` | `best_mast.pt` (12 of 20 each) | `zara_g4_mixv6_gsolve_cond3_1000` |
| FC | `scr-F3-only` | `best_mast.pt` (9 of 20) | `gsolve_cond3_1000` |
| V7C | `scr-V7C-v7only` | `best_mast.pt` (9 of 20) | `zara_g4_v7` |

Every run also trains on the official DIII-D training data (through the feature cache in `/fsx/chris.beckham/fusion_cache`)
and on Zara's synthetic DIII-D set (`zara_v1_d3d`).

`best_mast.pt` is the scored epoch (every 3 epochs) with the highest synthetic-MAST S among those with DIII-D S ≥ 0.85,
saved with the averaged (EMA) weights where EMA is on. The epochs in the table come from each run's `best_mast.json`.

To check: Z, Z2, Z3 (9 October) and Z6b's E runs (11 October) predate the last edits to `train_mm.py`, `synth_data.py`
and `shape_loss.py` before they were committed. Whether the frozen code reproduces them is not yet verified.

## From checkpoints to the prediction file

All scripts are in `starter/my_experiments/nn/`; prediction itself runs `multimachine/my_experiments/nn/predict.py`.

| Pipeline | Models | Steps |
|---|---|---|
| A: `build_candidate.sh` | Z, Z2 | predict MAST from one checkpoint → `postproc_qb.py` (q95 from the map, βN = 1.92 × Thomson electron βN) → βN × 1.2 → float32 |
| B: `build_ensemble.sh` then A's post-processing | Z3, Z7, Z8 | mean of the members' ψ, q95 and βN, then as A |
| Z6b | Z6b | B on the four E runs → `mastZ6maps-…`, then `splice_q95.py` takes q95 from Z |
| `build_Z9.sh` | Z9qb | B on seven runs, then the q95 rule and V2 βN as in C |
| C: `build_qb.sh` | Z15, Z17, Z18, FA, FC, V7C | B → `q95_study/H_v4/apply_cc.py --factor 0.932` (q95 × 0.932 where the boundary touches the centre column) → `q95_study/F_betaN/apply_bn_variant.py --variant V2` (βN = 2.6088 × geometric electron βN) |

All predictions use the halved-solenoid MAST inputs (`_sol0.5`) and the limiter mask (`FEC_MAST_MASK=limiter`).
Each candidate folder also holds a DIII-D file from Chris's Challenge 1 model; the team blends do not use it.
