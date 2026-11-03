# Zara's DIII-D package (Challenge 1, and the DIII-D file of Challenge 2)

Team zenithon-zendaya, Fusion Equilibrium Challenge (NeurIPS 2026). This package rebuilds and reruns the DIII-D pipelines of our
Final submissions: flux map ψ(R, Z), q95 and βN for every EFIT time slice of a DIII-D shot, from coil currents, plasma current,
toroidal field and Thomson profiles only. Three recipes share the code:

| Recipe | Used in | Trained components | Training, one RTX PRO 6000 (80 GB cap) | Development board DIII-D S |
|---|---|---|---|---|
| **v13** | Challenge 1 submission (DIII-D file only) | 12 ψ networks (60 epochs), 8 functional heads, 3 sequence heads, q95 / βN recipe R3c (5 out-of-fold ψ networks, 12 MLP, 6 sequence and 6 variant-D scalar networks) | 14 h 46 min measured end to end | 0.996946 |
| **Ch1-safe** | Challenge 1 alternative with a wider budget margin | 12 ψ networks and 8 functional heads on a 40-epoch schedule, 3 sequence heads, q95 / βN recipe R3c-lite (5 out-of-fold ψ networks, 6 MLP and 6 sequence scalar networks) | about 10.3 h (from the per-step times of the v13 rebuild) | 0.996945 |
| **Ch1-min** | Challenge 2 submission (its DIII-D file; Challenge 2 needs both files) | 6 ψ networks (v13 seeds 1-6, 30-epoch schedule), their own shape outputs as calibration targets, 3 sequence heads, 3 variant-D q95 / βN networks | GPU 2.05 h measured (22 Oct) + CPU caches about 1.1 h | 0.996881 |

## Layout

The tree follows the starter kit, so every script finds its neighbours by relative path.

| Path | Content |
|---|---|
| `env.sh` | paths and interpreters (`FEQ_ROOT`, `FEQ_CACHE`, `FEQ_PY`, `FEQ_PY_STARTER`, `FEQ_WORKERS`) |
| `train_ch1.sh`, `predict_ch1.sh` | v13: training entrypoint (from scratch) and inference entrypoint (any inputs-only split) |
| `train_ch1safe.sh`, `predict_ch1safe.sh` | Ch1-safe: the v13 steps with the 40-epoch schedule and R3c-lite |
| `train_ch1min.sh`, `predict_ch1min.sh` | Ch1-min: a subset of the v13 steps with the same commands |
| `checks/` | comparison with a submitted file (`compare_sub.py`), bound on the change in S (`noise_bound.py`; `noise_bound_rtx98.json`: 98 Final shots on the RTX PRO 6000), the labelled-shot test of GPU rounding (`score_lab.py`, `lab_noise.*`, `noise_lab.*`), reference step times, the retrain check of v13 |
| `fetch_weights.py`, `weights_manifest.json` | downloads the shipped checkpoints from `zenithon-zendaya/diii-d-model-weights` (pinned commit, SHA256 per file) into `my_experiments/ml/*/runs/<name>/` |
| `prepare_inference.sh` | CPU step before shipped-weights inference: per-shot training cache the q95 / βN networks read |
| `link_data.sh` | points `challenge_data/data/<split>/` at the official data downloaded by `../../data/fetch_official.py`; an optional second argument is a separate download of the Final dataset (the folder holding `data/diii_d_private_test/`) |
| `fusion_scoring/` | the challenge scorer's modules (unmodified; the calibration step uses its LCFS and functional code) |
| `my_experiments/c1/` | input features, C1 physics + Ridge base model, per-shot input caches |
| `my_experiments/ml/ch1h/` | Group H long-history inputs; trainer of the ψ networks and sequence heads (`train_c1h.py`) |
| `my_experiments/ml/ch1c/` | trainer of the functional heads (`train_c1c.py`); label-side functionals of the true maps |
| `my_experiments/ml/ch1q/`, `ch1s/` | q95 / βN networks and blends (R3c for v13, D3 for Ch1-min) |
| `my_experiments/ml/ch1k/` | calibration of the ψ map (rule `rules/c4.json`) |
| `my_experiments/ml/ch1/` | ensemble mean, submission file builder |
| `my_experiments/ml/final/` | inference from saved weights (`infer_c1h.py`), R3c prediction script, network list, CUDA memory cap (`capmem.py`) |
| starter files | `data_fixes.py` (applied when the raw files are read), `validate_submission.py`, `push_predictions.py`, `submission_skeleton.py` (starter commit 823f6ee) |

## Requirements

- One NVIDIA GPU with 80 GB (A100 80GB class), 16 CPU cores, 64 GB RAM.
- Disk for `$FEQ_CACHE`: about 150 GB for training; about 75 GB per 900-shot split for v13 inference (each network's per-shot
  predictions are about 3 GB) and about 25 GB for Ch1-min. `FEQ_CLEAN=1` deletes the per-network predictions once the file is
  written.
- Environments `fusion-eq-gpu` and `fusion-equilibrium-starter` from `../environment/` (see its README).

```bash
python ../../data/fetch_official.py --root <data> --splits diii_d_train diii_d_public_test diii_d_private_test
bash link_data.sh <data>
export FEQ_CACHE=<cache> FEQ_PY=<envs>/fusion-eq-gpu/bin/python FEQ_PY_STARTER=<envs>/fusion-equilibrium-starter/bin/python
```

## Train (from scratch)

```bash
bash build_caches.sh ch1 && bash train_ch1.sh               # v13
bash build_caches.sh ch1safe && bash train_ch1safe.sh       # Ch1-safe
bash build_caches.sh ch1min && bash train_ch1min.sh         # Ch1-min
```

`build_caches.sh` is the CPU preprocessing of a recipe (steps 1 and 2 below, and the q95 / βN inputs that need no trained
network: x_b targets, the input export of the Ridge base map and first-pass features, history features, out-of-fold lists);
every step in it runs with no GPU visible. `train_*.sh` then trains the networks (steps 3 and 4) and stops at once if the
preprocessing of its recipe has not run. Two CPU steps read a trained network's output (the q-curves of the out-of-fold ψ
networks' maps and the map features assembled from them), so they stay in the training entrypoint, with no GPU visible. In
`train_ch1.sh` (v13) they run in the background while the GPU trains the ψ networks and heads, which do not read them, so the
GPU does not wait for them; `train_ch1safe.sh` runs them between its GPU steps. All steps use the 7,041 official DIII-D training
shots (no holdout, no other data). Each step is skipped when
its output exists, so an interrupted run resumes; wall-clock time per step goes to `$FEQ_CACHE/train_times.tsv`
(`train_times_ch1safe.tsv`, `train_times_ch1min.tsv`), with its kind (CPU or GPU) in the last column. The public test inputs
are needed for training too (the q95 / βN input export has a public-test part).

1. Input caches (per-shot features, Group A physics inputs, Group H long-history inputs) and label-side targets (the scorer's
   functionals and LCFS extremes of the true training maps; boundary fraction of the true maps). CPU, `build_caches.sh`.
2. C1 physics + Ridge base model (vacuum field of the coils plus a linear map of the inputs). CPU, `build_caches.sh`.
3. ψ networks (residual MLP 2048 × 4 on top of the Ridge map, Muon optimiser; v13: 12 networks, 60 epochs, seeds 1-6 weight
   decay 0.1, 7-12 weight decay 0.2; Ch1-min: seeds 1-6 on a 30-epoch schedule), v13's 8 functional heads, 3 sequence heads. GPU.
4. q95 / βN networks. v13 (R3c): 5 out-of-fold ψ networks (map features of the training frames; `train_ch1.sh` trains them
   first), 12 MLP, 6 sequence and 6 variant-D scalar networks. Ch1-min (D3): 3 of the variant-D networks, whose wall-fraction inputs for training frames come from
   the true maps (no out-of-fold networks). GPU.

## Predict (from the shipped weights)

```bash
python fetch_weights.py                       # checkpoints -> my_experiments/, inference artifacts -> $FEQ_CACHE
bash prepare_inference.sh                     # CPU, about 5 min: training cache the q95 / βN networks read
bash predict_ch1.sh diii_d_private_test       # -> $FEQ_CACHE/pred/diii_d_private_test/sub/diii_d_private_test.npz
bash predict_ch1safe.sh diii_d_private_test   # -> $FEQ_CACHE/pred/ch1safe/diii_d_private_test/sub/diii_d_private_test.npz
bash predict_ch1min.sh diii_d_private_test    # -> $FEQ_CACHE/pred/ch1min/diii_d_private_test/sub/diii_d_private_test.npz
```

Keys `shot_XXXX_psirz`, `shot_XXXX_q95`, `shot_XXXX_betaN`; shots in sorted parquet order (the Hugging Face stream order); ψ
float32. Every step is a fixed rule applied to all frames; there are no per-shot or per-frame manual edits. The trained networks
predict on the GPU in bf16, as when the submitted files were made, and the split's input export for the q95 / βN networks also
runs on the GPU; the input caches, the mean map, the q95 / βN blend, the calibration and the submission file are CPU steps. Prediction is not training: forward passes at inference do not count toward the 24 GPU-hours
(organisers' Final guide of 19 October, section 6).

Inference also reads training-side files that training writes into `$FEQ_CACHE`: the Ridge base, the feature layouts of the
training caches, the training labels and training-frame features from which the q95 / βN networks recompute their
normalisation when they load (including the out-of-fold map features), and the official training split. A from-scratch
rebuild makes slightly different copies of the fitted ones, so the package ships the copies the weights were trained
against (`artifacts/` of the weights repository, 5.2 GB, placed by `fetch_weights.py`); `prepare_inference.sh` rebuilds the
per-shot training cache on CPU (byte-identical). The official training data must be in `challenge_data/data/diii_d_train/`.

## Data provenance

- DIII-D: only the official challenge training data (`diii_d_train`), its inputs and its EFIT labels (ψ, q95, βN, LCFS).
  Label-side training targets (the scorer's functionals of the true maps, the boundary fraction) are computed from those labels.
  No external data, no synthetic data, no pretrained models.
- Test splits: inputs only. Nothing is fitted or tuned on test inputs.

## Checks

All on the AWS cluster, one NVIDIA RTX PRO 6000 Blackwell (96 GB) per job, environments built fresh from `../environment/`.

| Check | Result |
|---|---|
| Environments built from the package files (empty package cache, no pip cache) | `pip list` identical to the pinned lists; see `../environment/README.md` |
| Input caches of the public test rebuilt in the new environment | byte-identical to the caches the models were run on (875 of 875 files) |
| Three v13 networks (ψ, context head, sequence head) predicted from the shipped weights in the new environment | byte-identical (874 of 874 shots each) |
| Shipped weights on Hugging Face (`zenithon-zendaya/diii-d-model-weights` @ `db0846d6`) | 146 files (73 networks + training logs), 19.2 GB; SHA256 of every file at that commit equals the local file |
| Smoke test of `train_ch1min.sh` → `predict_ch1min.sh` and `train_ch1safe.sh` → `predict_ch1safe.sh` from an empty cache (300 training shots, 60 test shots, 1 epoch per network) | every step ran; complete, finite files (`final/slurm/pkg_smoke_new.sbatch`) |
| `train_ch1min.sh`, every network from scratch on one GPU under the 80 GB cap (CPU caches from the 16 Oct rebuild) | GPU steps 2.05 h: ψ networks 713-736 s each, sequence heads 313-316 s, variant-D networks 685-709 s |
| Shipped weights + shipped inference artifacts → Final inputs, 98-shot subset (every 9th shot), all three recipes, against the per-shot predictions the submitted files were built from | v13: 28 of 98 shots byte-identical, largest single-frame difference ψ 0.013, q95 0.007, βN 0.005; Ch1-min: 30 of 98, ψ 0.0056, q95 0.014, βN 0.0055; Ch1-safe: 29 of 98, ψ 0.0040, q95 0.011, βN 0.0036 (GPU reductions in the q95 / βN networks; their values feed the map calibration) |
| Per-shot training cache rebuilt on CPU against the one the networks were trained on | identical (21 of 21 sampled files) |
| v13 smoke test and one-GPU rebuild (16-17 October) | `train_ch1.sh` from an empty cache: 14 h 46 min, peak CUDA memory 69.0 GB under the 80 GB cap; rebuilt weights against the submitted v13 file on the public test: ψ R² 0.999998878 |
| v13 shipped weights → all 879 Final shots, 27 October: clean copy of the package, environments from the pinned specs, official data checked against Hugging Face's SHA256 (7,041 + 874 + 879 files), weights and inference artifacts from `zenithon-zendaya/diii-d-model-weights` @ `c2e8c4a3` (156 of 156 files match the manifest), the root inference entrypoint `bash predict.sh OUT` | byte-identical to the submitted file: SHA256 `3af1ebe0…` both, 2,637 of 2,637 arrays; organisers' validator OK; score difference with the official scorer code 0; 1 h 52 min |
| Input export (`r3c_export`) on the CPU against the GPU export, full training set and public test (16 cores, 9 min, 14 GB RAM) | Ridge features, physics inputs and every index array identical; first-pass features identical NaN pattern, 1 frame in 1.56 million off by more than 1e-3 of a column's spread (two GPU exports differ in 13) |
| Smoke test of `build_caches.sh ch1` → `train_ch1.sh` → `predict_ch1.sh` (300 training shots, 1 epoch), every GPU process sampled every 2 s | no process on the GPU during `build_caches.sh`; only the network trainers on the GPU during `train_ch1.sh`; 52 GPU and 24 CPU steps, each marked correctly; the background CPU steps start when the last out-of-fold network ends and finish before the first network that reads them ("the GPU waited 0 s"); a failing background step and a failing GPU step both stop `train_ch1.sh` with an error and leave no process behind |
