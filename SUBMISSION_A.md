# Submission A: Challenge 1 (DIII-D), v13

Team zenithon-zendaya, Fusion Equilibrium Challenge Final. Submission A holds one file, `diii_d_private_test.npz` (879 Final
DIII-D shots), and is entered for Challenge 1 only. It has no MAST file.

| | |
|---|---|
| Model | v13 (`zara/diii-d/`, recipe `ch1`) |
| Development board, DIII-D S | 0.996946 (public test; the Final fold is scored blind) |
| Training entrypoint | `bash prepare.sh A` (CPU preprocessing, no GPU visible), then `bash train.sh A` (the networks, on one GPU) |
| Inference entrypoint | `bash predict.sh A OUT_DIR` (shipped weights) or `bash predict.sh A OUT_DIR --rebuilt` (retrained) |
| Training data | the official DIII-D training split only (7,041 shots); no other machine, no synthetic data (section "Data provenance") |
| GPU training time | **20.9 GPU-hours on one A100-SXM4-80GB**, measured twice on two GCP machines: 20.9 (23-25 October) and 20.77 with this package's `train.sh A` (30 October to 1 November); 12.8 GPU-hours on one RTX PRO 6000 |
| CPU time | training, on the A100 machines: about 5.5 to 6 h of preprocessing (`prepare.sh A`) before the first network, and about 2.5 h alongside GPU training; prediction: about 1.5 h on 16 cores |

## The model

All networks are trained on the 7,041 training shots, with no holdout.

| Component | Count | Trained by | Used at prediction |
|---|---|---|---|
| C1 physics + Ridge base map (vacuum field of the coils plus a linear map of the inputs) | 1 | `build_caches.sh`, CPU | yes |
| ψ networks (residual MLP 2048 × 4 on the Ridge map, Muon, 60 epochs; seeds 1-6 weight decay 0.1, 7-12 weight decay 0.2) | 12 | `train_ch1.sh`, GPU | yes |
| Functional heads (scog × 3, scog_ctx × 3, auxg × 2): calibration targets | 8 | `train_ch1.sh`, GPU | yes |
| Sequence heads: calibration targets | 3 | `train_ch1.sh`, GPU | yes |
| q95 / βN recipe R3c: out-of-fold ψ networks (map features of the training frames) | 5 | `train_ch1.sh`, GPU | no |
| q95 / βN recipe R3c: MLP, sequence and variant-D scalar networks | 12 + 6 + 6 | `train_ch1.sh`, GPU | yes |

52 networks are trained and 47 predict. The steps after the networks are fixed rules applied to every frame, with no per-shot
or per-frame edits: the ψ map is the mean of the 12 ψ networks; q95 and βN are a fixed blend of the scalar networks, the
q profile of the map and the ψ ensemble (weights in `zara/diii-d/my_experiments/ml/final/r3c_predict.sh`); the map is then
calibrated by rule c4 (`zara/diii-d/my_experiments/ml/ch1k/rules/c4.json`: elongation stretch about the X-point, axis shift,
li calibration with a range guard). Details: `zara/diii-d/README.md`.

## Compute (rule 6)

**Declared: 20.9 GPU-hours** of training on one A100-SXM4-80GB, measured on two occasions, a week apart, on two GCP machines
with the same GPU, driver (535.216.01) and CPU type (Intel Xeon @ 2.20 GHz):

| Networks | Count | 23-25 October | 30 October to 1 November |
|---|---|---|---|
| Machine | | `a2-ultragpu-1g`: one A100, 12 vCPUs, 170 GB RAM; nothing else ran | `a2-ultragpu-2g`: two A100s, 24 vCPUs shared with B's line; this training had its GPU to itself |
| Code | | an earlier version of the training scripts | this package: `prepare.sh A`, then `train.sh A` |
| ψ networks | 12 | 7.67 h | 7.85 h |
| Sequence heads | 3 | 0.47 h | 0.42 h |
| Functional heads | 8 | 4.27 h | 4.11 h |
| Out-of-fold ψ networks (q95 / βN recipe) | 5 | 2.52 h | 2.64 h |
| q95 / βN MLPs | 12 | 3.45 h | 3.38 h |
| q95 / βN sequence networks | 6 | 0.75 h | 0.71 h |
| Variant-D networks | 6 | 1.70 h | 1.67 h |
| Input export | | about 2 min (on the GPU then) | on the CPU, in `prepare.sh A` |
| **GPU total** | **52** | **20.9 h** | **20.77 h** |
| Peak GPU memory | | 70.5 GB | 72.2 GB in `nvidia-smi` |

Records: `checks/a100-dedicated/README.md` (23-25 October, every step) and `checks/a100-final/README.md`, section "Training
time: two A100 measurements" (30 October to 1 November, with `train_times.tsv` and the logs).

All steps of the recipe, on the A100 machines and on our RTX machine (a node of our AWS cluster: one RTX PRO 6000 Blackwell
capped at 80 GB, 16 cores):

| Part | Kind | A100, 23-25 October (12 vCPUs) | A100, 30 October to 1 November (24 shared vCPUs) | RTX machine |
|---|---|---|---|---|
| `build_caches.sh ch1` (`prepare.sh A`): input caches, label-side targets, C1 physics + Ridge base, input export, history features, out-of-fold lists | CPU only, no GPU visible | about 5.5 h | 5 h 52 min | 1 h 28 min (27 October) |
| `train_ch1.sh` (`train.sh A`): the 52 networks, one after another | GPU | **20.9 h** | **20.77 h** | 12.8 h (16-17 October) |
| `train_ch1.sh`: q-curves and map features of the 5 out-of-fold networks (`r3c_oofqc_*`, `r3c_mapf`) | CPU, in the background while the GPU trains networks that do not read them | 2.62 h, as separate steps then | 2.50 h | about 1 h (27 October) |
| `predict.sh` on the 879 Final shots (shipped weights) | GPU: the networks' forward passes (8 min) and the q95 / βN step (30 min, GPU and CPU); everything else CPU (about 1 h 22 min) | | 8 h 22 min, alongside two trainings on the shared CPUs | about 2 h on 16 cores (two runs on 27 October: 1 h 52 min and 2 h 0 min); peak memory about 125 GB (Slurm accounting of the 27 October run) |

Prediction time and memory are not capped. The CPU steps of `predict.sh` run `NCPU` worker processes (`settings.sh`, default
every core); fewer workers use less memory.

Only GPU work counts toward the 24 GPU-hours. Forward passes at inference do not count (organisers' Final guide of
19 October, section 6). The CPU preprocessing has its own entrypoint, `bash prepare.sh A`, with no GPU visible; it can run on
a machine without a GPU. `bash train.sh A` then trains the networks on the A100 and prints their GPU time at the end. Every
step writes its wall-clock time and its kind (CPU or GPU) to `$FEQ_CACHE/train_times.tsv`.

## Data provenance

DIII-D only; the submission has no MAST file.

| Data | Source | Use |
|---|---|---|
| `diii_d_train` (7,041 shots, inputs and labels) | official challenge data, `Sophelio/fusion-equilibrium-challenge` (pinned revision in `data/fetch_official.py`) | the only data any component is fitted to |
| `diii_d_public_test` (874 shots, inputs only) | same | read by the input export in `build_caches.sh`, which writes a training and a public-test part; nothing is fitted to it |
| `diii_d_private_test` (879 Final shots, inputs only) | official Final data, `ganatma/fusion-equilibrium-challenge-final` | prediction only |

No other machine, no synthetic data, no external dataset and no pre-trained model. Every network starts from random weights
(seeded) and is trained by `train.sh A`.

## Files

| Item | Where |
|---|---|
| Submitted file | `diii_d_private_test.npz`, SHA256 `3af1ebe0ee31cef2c0cd827434850e3a9fc5063b42fc2ebb2c873f73d9c1be27` |
| Predictions repository (pointer route) | `zenithon-zendaya/fec-final-sub1` @ `807b5104fa53619102cb728c980c5f481db9b499` |
| Shipped weights | `zenithon-zendaya/diii-d-model-weights` @ `c2e8c4a3` (pinned with SHA256 per file in `zara/diii-d/weights_manifest.json`; downloaded by `zara/diii-d/fetch_weights.py`, 24.9 GB, which also holds the networks of the other DIII-D recipes) |
| Pinned environments | `zara/environment/fusion-eq-gpu.yml` (networks) and `zara/environment/fusion-equilibrium-starter.yml` (input caches, validator), with their pip and conda lists |
| Official data | `data/fetch_official.py` (Hugging Face, pinned revisions, SHA256 per file) |

## Reproduce on a new machine

Step by step in `REPRODUCE.md` (steps 0-2, then 4-6 with the letter A). In short: one A100 80 GB, 12 or more cores, about
128 GB RAM (less with fewer workers, `REPRODUCE.md`) and 400 GB of disk; `bash prepare.sh A` and `bash train.sh A` retrain every network from scratch, and `bash predict.sh A OUT_DIR`
regenerates the submitted file from the shipped weights (`--rebuilt` for the retrained ones).

## Checks

| Check | Result |
|---|---|
| Shipped weights → all 879 Final shots, 27 October: clean copy of the package (`git archive`), the inference entrypoint with the shipped weights, environments from the pinned specs, official data (7,041 + 874 + 879 files) and weights (156 of 156 files) checked against Hugging Face's SHA256, one RTX PRO 6000 | byte-identical to the submitted file (SHA256 `3af1ebe0…`, 2,637 of 2,637 arrays); validator OK; score difference with the official scorer code 0; 1 h 52 min |
| v13 from an empty cache on one RTX PRO 6000 under the 80 GB cap, 16-17 October | 14 h 46 min end to end, peak CUDA memory 69.0 GB; ψ R² 0.999998878 against the submitted predictions on the public test |
| v13 retrained from scratch and compared with the submitted file, 27-28 October: `build_caches.sh ch1` and `train_ch1.sh` from an empty cache on one RTX PRO 6000 (80 GB cap, 12 CPUs, 160 GB RAM), then `predict_ch1.sh` on the 879 Final shots. Code: an earlier version of this package that differs from this one only in shell plumbing and documentation | the retrained networks are not bit-identical (0 of 2,637 arrays identical); ψ R² 0.9999990 against the submitted file; the official scorer with the submitted file as labels gives S 0.999716; label-free bound on the change in S: **0.0016**, against the 0.01 allowed (`zara/diii-d/checks/noise_bound.py`, Development-board R² as stand-ins; result `zara/diii-d/checks/retrain_final_v13.json`). GPU steps 13.1 h (one network was rerun after the disk filled) |
| **Shipped weights on an A100**, 30-31 October: `REPRODUCE.md` steps 1-4 on a new GCP machine with two A100-SXM4-80GB (driver 535), whose 24 vCPUs were shared with two trainings (`checks/a100-final/`) | validator OK; not byte-identical (0 of 2,637 arrays): the A100 rounds the networks' forward passes differently from the RTX PRO 6000 the submitted file was made on (the CPU steps do not depend on the number of workers). Largest difference ψ 0.011, q95 0.032, βN 0.013; ψ R² 0.99999994. Estimated change in S **1e-5** (9.9e-6: the change measured on 101 labelled shots predicted twice, scaled by the size of the differences); strict label-free bound 4.7e-4. `predict.sh A` 8 h 22 min on the shared CPUs |
| **Retrained on an A100**, 30 October to 1 November, on the same machine: `prepare.sh A` (5 h 52 min on the shared CPUs), `train.sh A` on its own GPU (52 networks, 20 h 46 min), then `predict.sh A --rebuilt` (3 h 15 min) and `steps/5_compare.sh` (`checks/a100-final/`) | GPU training **20.77 GPU-hours** (20.9 on 23-25 October, `checks/a100-dedicated/`). Retrained file: validator OK; 0 of 2,637 arrays identical; largest difference ψ 0.018, q95 0.37, βN 0.059; ψ R² 0.9999988 against the submitted file; label-free bound on the change in S **1.5e-3**, against the 0.01 allowed (`zara/diii-d/checks/noise_bound.py`, Development-board R² as stand-ins), as for the RTX retrain above |
| Smoke test of `build_caches.sh ch1` → `train_ch1.sh` → `predict_ch1.sh` (300 training shots, 1 epoch), GPU processes sampled every 2 s, 27 October | no process on the GPU during `build_caches.sh`; only the network trainers on the GPU during `train_ch1.sh`; 52 GPU and 24 CPU steps, each marked correctly; "the GPU waited 0 s"; a failing step stops `train_ch1.sh` with an error and leaves no process behind |
| Input export on the CPU against the earlier GPU export, full training set and public test | Ridge features, physics inputs and index arrays identical; first-pass features: 1 frame in 1.56 million off by more than 1e-3 of a column's spread (two GPU exports differ in 13) |
| Final inputs in a separate download of the Final dataset, against Hugging Face and the copy the submitted file was built from | 879 DIII-D files match the published SHA256 and are byte-identical to that copy |
| Predictions repository with only its read token | private; holds exactly `diii_d_private_test.npz` at the root at the commit above; LFS SHA256 equals the submitted file |

More rows, per recipe: `zara/diii-d/README.md`.
