# Submission B: Challenge 2 (DIII-D + MAST), MAST with r1

Team zenithon-zendaya, Fusion Equilibrium Challenge Final. Submission B holds two files: `diii_d_private_test.npz` (879 Final
DIII-D shots, recipe Ch1-min) and `mast_private_test.npz` (1,205 Final MAST shots, a fixed blend of 8 of Chris Beckham's MAST models and Zara's r1). It is entered for Challenge 2.

| | |
|---|---|
| DIII-D model | Ch1-min (`zara/diii-d/`, recipe `ch1min`) |
| MAST model | blend of FA, FC, V7C, Z, Z15, Z2, Z3, Z8 (`chris/mast/`) and r1 (`zara/mast/r1/`), weights `blend/weights_chris8r1.json` |
| Development board | DIII-D S 0.996881 (Ch1-min, public test). The MAST blend was built after the Development phase and was not scored there; its weights come from the Development-phase scores of our own submissions (`blend/derive/`, section "Development feedback and checkpoint selection") |
| GPU use | Network training, and the networks' forward passes at prediction. Every data file and cache, for training and for prediction (including r1's first-pass caches), is built with no GPU visible |
| Training entrypoint | `bash prepare.sh B` (CPU preprocessing, no GPU visible), then `bash train.sh B` (the networks, on one GPU) |
| Inference entrypoint | `bash predict.sh B OUT_DIR` (shipped weights) or `bash predict.sh B OUT_DIR --rebuilt` (retrained) |
| GPU training time | **17.84 GPU-hours on one A100-SXM4-80GB**, measured twice on GCP machines: 17.84 (23-25 October) and 17.68 with this package's `train.sh B` (30 October to 1 November); 11.8 on one RTX PRO 6000 |
| CPU time | DIII-D preprocessing (`prepare.sh B`): 4.83 h on a 12-vCPU A100 machine; CPU-only data generation: section "CPU-only data generation" |

## The model

### DIII-D: Ch1-min

All networks are trained on the 7,041 official training shots, with no holdout.

| Component | Count | Trained by |
|---|---|---|
| C1 physics + Ridge base map | 1 | `zara/diii-d/build_caches.sh ch1min`, CPU |
| ψ networks (v13's seeds 1-6, 30-epoch schedule) | 6 | `zara/diii-d/train_ch1min.sh`, GPU |
| Sequence heads (calibration targets) | 3 | same |
| Variant-D q95 / βN networks (`prod_xbmix_D`, `prod_ctx5_D`, `prod_ctx5sm_D`) | 3 | same |

The map is the mean of the 6 ψ networks; the ψ networks' own shape outputs serve as calibration targets. Details:
`zara/diii-d/README.md` and the header of `zara/diii-d/train_ch1min.sh`.

### MAST: blend of B's 8 models and r1

| Model | Training runs | ψ map | q95 | βN |
|---|---|---|---|---|
| V7C | `scr-V7C-v7only` | 0.4038 | 0.1117 | |
| Z15 | `scr-mastZ15-v6mix-seed0` to `seed3` | 0.1681 | 0.2416 | |
| FC | `scr-F3-only` | 0.1491 | 0.2137 | |
| Z3 | `scr-F1-final-g4x5000`, `scr-F2-final-g4x5000-seed1`, `scr-G1-g4x5000-noise0.3`, `scr-G4-g4x5000-clip1` | 0.1237 | 0.1658 | |
| r1 | `h_muon_mo`, `w3_ft`, `rt_ft`, `rt_mix`, `rt_ftw3`, `rt_nonoise` (`zara/mast/r1/recorded/`) | 0.1224 | | 0.2875 |
| Z | `scr-F1-final-g4x5000` | 0.033 | | |
| Z2 | `scr-G1-g4x5000-noise0.3` | | 0.1274 | |
| FA | `scr-F1-z15mix-s0`, `scr-F2-z15mix-s1`, `scr-F6-z15mix-s2` | | 0.1064 | 0.7125 |
| Z8 | `scr-V1-v4only`, `scr-W4-v4only-ep12-seed1` | | 0.0335 | |

Weights `blend/weights_chris8r1.json` (blend search on Development-phase scores, `blend/derive/` and section "Development
feedback and checkpoint selection"; r1's scores there are those of its Development-phase networks, see below).

### r1 (Zara Ercan's MAST model)

The average of four networks (`rt_ft`, `rt_mix`, `rt_ftw3`, `rt_nonoise`) fine-tuned from one base network (`h_muon_mo`, via
`w3_ft` for `rt_ftw3`), each run with a per-frame inboard field correction (11 settings) picked by the selector `rt_ft`, then fixed
post-processing (`zara/mast/r1/predict/README.md`). Training code with Chris's speed-ups: `zara/mast/r1/` (`README.md`,
`SPEEDUP.md`: bit-identical boundary search).

The shipped networks were trained with this code on training caches built on the CPU (`zara/mast/r1/cpu_caches/`): Chris
Beckham, 26-27 October, the six runs of `train_r1.sh full` on one RTX PRO 6000. The first-pass caches of the Final inputs are
also built on the CPU (`predict_r1.sh`). r1's Development-phase networks (same recipe and run arguments) scored S 0.675 alone on
the Development board and were inside five scored blends. On Zara's internal validation set (below) the shipped networks
score S 0.8404 against 0.8387 for the Development-phase networks.

**Zara's internal validation set.** 3,816 held-out cases of three of Zara's FreeGSNKE sets (`gen4` 20000: 2,560 cases; 5000:
616; 5000v3: 640), none of them used in training. The inputs are the challenge's own `mast_public_test` inputs: coil
currents, plasma current and toroidal field (scaled by a few per cent per case), and the Thomson profiles as shipped. The
equilibrium labels are solved by FreeGSNKE (`zara/datagen/`). No MAST label is used: the public-test inputs are used
without their targets, as rule 3 allows. The set ranks models of one family; it does not predict the Final score.

## Compute (rule 6)

**Declared: 17.84 GPU-hours** of training on one A100-SXM4-80GB, measured on two occasions, a week apart, on GCP machines with
the same GPU, driver (535.216.01) and CPU type (Intel Xeon @ 2.20 GHz):

| Part | Kind | 23-25 October | 30 October to 1 November | RTX machine |
|---|---|---|---|---|
| Machine | | `a2-ultragpu-1g`: one A100, 12 vCPUs, 170 GB RAM, one training at a time | `a2-ultragpu-2g`: two A100s, 24 vCPUs shared with A's line; this training had its GPU to itself | one RTX PRO 6000 Blackwell capped at 80 GB, 16 cores (AWS) |
| Code | | earlier versions of the training scripts | this package: `prepare.sh B`, then `train.sh B` | |
| `zara/diii-d/train_ch1min.sh` (12 networks) | GPU | **3.23 h** | 3.18 h | 2.05 h |
| `chris/mast/train.sh` (15 runs) | GPU | **10.96 h** (10.99 h on a second machine at the same time) | 11.07 h | 7.96 h |
| r1: `zara/mast/r1/scripts/train_r1.sh full` (6 runs) | GPU | **3.65 h** (25 October, a separate run on the same machine type) | 3.43 h | 1.80 h |
| **GPU total** | | **17.84 h** | **17.68 h** | 11.8 h |
| Peak GPU memory | | 59.6 GB | 61.0 GB in `nvidia-smi` | |

Records: `checks/a100-dedicated/fec-B/README.md` (Ch1-min and Chris's runs, 23-25 October),
`zara/mast/r1/checks/a2-ultragpu-1g/RESULTS.md` (r1, 25 October) and `checks/a100-final/README.md`, section "Training time:
two A100 measurements" (30 October to 1 November, every run side by side with the first measurement).

The other steps, which count 0 GPU-hours:

| Part | Kind | A100 machines | RTX machine |
|---|---|---|---|
| `zara/diii-d/build_caches.sh ch1min` (`prepare.sh B`) | CPU only | 4.83 h (12 vCPUs, 23-25 October); 7 h 41 min with all of `prepare.sh B` on 6 workers alongside other work (30 October) | about 1.3 h |
| r1: CPU-built training caches (`zara/mast/r1/cpu_caches/`) | CPU only, no GPU visible | built beforehand on CPU nodes (below) and downloaded | |
| `predict.sh B` on the Final shots (shipped weights), both files | GPU: the networks' forward passes; everything else CPU | 7 h 23 min on the 2×A100 machine of 30-31 October, on 6 workers alongside a training (`checks/a100-final/`) | 2 h 47 min (30 October, shared node). Each machine on its own, 16 cores (27 October): DIII-D 2 h 17 min, peak memory about 117 GB; MAST 1 h 42 min, peak memory about 28 GB (Slurm accounting) |
| r1's first-pass cache of the Final MAST inputs, only if it is rebuilt instead of downloaded (`predict_r1.sh`) | CPU only, no GPU visible | | about 2.1 h on 8 shards of 12 cores, under 1 GB of memory per shard (27 October) |

Prediction time and memory are not capped. The CPU steps of `predict.sh` run `NCPU` worker processes (`settings.sh`, default
every core); fewer workers use less memory. r1's training caches are built on the CPU, as for the shipped networks, so they
count 0 GPU-hours.

Only GPU work counts toward the 24 GPU-hours; forward passes at inference do not count.

### CPU-only data generation

Everything below ran with no GPU, before training; none of it counts toward the 24 GPU-hours. Each generated dataset is
downloaded by `REPRODUCE.md` step 3, pinned by SHA256. The code that generated each set: `zara/datagen/README.md` (Zara's
FreeGSNKE sets) and `chris/mast/gsolve/SOURCES.md` (gsolve).

| Data | Used by | Hardware | Wall-clock | CPU time |
|---|---|---|---|---|
| Zara's FreeGSNKE synthetic sets: 500 DIII-D cases, MASTSYN (generations 1 and 3), gen4 5000, 20000, 5000v3, 5000v4, 1500v6, 1500v7 (5-16 October; per set: `chris/mast/DATASETS.md`) | every MAST model of the blend | AWS g7e.48xlarge nodes (Intel Xeon Platinum 8559C, 96 CPUs), up to 93 single-threaded solver processes per node | 39.4 h in all | about 4,760 CPU-hours |
| Chris's gsolve set `cond3_1000` (17 October) | FC, FA | one node of our AWS cluster, at most 40 cores (CPU model not logged) | 81 min | 14.4 process-hours (12 processes × 3 threads) |
| Training caches of those sets and the merged caches (`data/fetch.py`) | Chris's models | any CPU | a few minutes | |
| r1's first-pass training caches (26-27 October) | r1 | AWS g7e nodes (Intel Xeon Platinum 8559C), 48 CPUs per job | about 6 h (7.0 h back to back) | 47 process-hours, 265 CPU core-hours |
| DIII-D preprocessing of Ch1-min (`zara/diii-d/build_caches.sh ch1min`, run by `prepare.sh B`) | Ch1-min | the A100 machine of 23-25 October (12 vCPUs) | 4.83 h | |

In all, Zara's FreeGSNKE sets hold 281,841 labelled synthetic equilibria. The six gen4 sets are organised as shots: 42,730
short synthetic shots (4,856 held out) holding 234,767 equilibria, 5.5 per shot on average. MASTSYN (46,574) and the DIII-D
set (500) are single equilibria. The MAST sets take their inputs from the 1,206 shots of `mast_public_test`, the DIII-D set
from 704 shots of `diii_d_train`. Chris's gsolve set `cond3_1000` holds 6,581 shots (364 held out) with 49,220 equilibria.

## Data provenance

| Data | Source | Use |
|---|---|---|
| `diii_d_train` (7,041 shots, inputs and labels) | official data, `Sophelio/fusion-equilibrium-challenge` | Ch1-min; every MAST model also trains on it |
| `diii_d_public_test`, `mast_public_test` (inputs only) | same | input export (Ch1-min); MAST coil pictures and test inputs read by Chris's training monitors; the generators of the synthetic MAST data read MAST inputs only |
| `diii_d_private_test`, `mast_private_test` (Final inputs only) | official Final data, `ganatma/fusion-equilibrium-challenge-final` | prediction only |
| r1's training data | official DIII-D training data and synthetic MAST equilibria (FreeGSNKE, generated on CPU from inputs only: the challenge's own `mast_public_test` inputs, no MAST label; two of the four fine-tune caches, `mastg4rt` and `mastg4v3rt`, give the networks the shipped Thomson profiles of those shots as input, the other two the profiles the generator rendered from its own solution); first-pass caches built on the CPU (`zara/mast/r1/cpu_caches/`); the shipped networks were trained on these | r1 |
| Synthetic MAST and DIII-D equilibria (FreeGSNKE gen4 sets by Zara, gsolve sets by Chris, `zara_v1_d3d`) | generated on CPU from inputs only; pinned in `zenithon-zendaya/mast-synthetic-data` (`data/manifest.json`) | training of the MAST models (which model uses which: `chris/mast/DATASETS.md`) |

No network is trained on MAST equilibrium reconstructions or EFIT-derived MAST quantities, no external dataset or pre-trained
model is used, and every network starts from random weights and is trained by `train.sh B`. Two choices draw on MAST label
information: the blend weights, the member pool and the blend entered in the Final come from the Development-phase scores
of our own submissions, and all 15 runs of Chris's models stop at the epoch chosen by the official scoring code on
synthetic data, which applies the organisers' MAST envelope mask (next section). The starter kit's MAST demo shot is read only to copy its column schema when
building a 500-case synthetic test set; no training set of B uses it (`zara/datagen/README.md`).

### Development feedback and checkpoint selection

B draws on MAST label information in two places: the scores and Detailed Results that the Development phase returned for
our own submissions, and `fusion_scoring/masks/mast_envelope.npz` inside the official scoring code. Both are
described here.

**Blend weights.** `blend/weights_chris8r1.json` was computed once on the CPU on 27 October by `blend/derive/`
(`code/fpb.py`, `code/search.py`) and has not changed since.

- Inputs: the Development-phase scores of our own submissions, member models and team blends (`blend/derive/inputs/`), and
  the members' `mast_public_test` prediction files, compared only with each other. All of these submissions were model
  predictions; none was a constant, shifted or rescaled prediction. No MAST label file is read.
- Method: the error of a linear blend equals the weighted errors of its members minus half their weighted disagreement with
  each other. From this identity, the score of one submitted team blend and the scores of its members give the total sum of
  squares of the public-fold labels about their mean, one number each for ψ, q95 and βN (`code/fpb.py`, lines 182-190). The
  equivalent numbers for the seven Consistency terms were chosen by leave-one-out on the submitted team blends, and one
  coefficient for D_LCFS was fitted on them. With these numbers, `fpb.py` computes the Development-board R² of ψ, q95 and βN
  of any blend of scored members exactly and estimates the other terms; `search.py` kept the weights with the highest
  estimated S, and `code/budget2.py` used the same estimates, with each member's training cost, to choose the pool. The
  blend itself was not submitted in the Development phase.
- What these numbers are: ten pooled numbers, each the overall spread of the public-fold labels of one quantity, and one
  coefficient. None is per shot or per frame, and no label value is reconstructed. They convert the differences between our
  own prediction files into changes of the Development-board score.
- FC and V7C were never submitted alone. Their ψ, q95, D_LCFS and Consistency scores were solved in the same way from the
  scores of two submitted team blends that contain them (`inputs/board_s3.json`, entries marked `pinned`; `code/pin.py`).
  Their βN score is not known, so they carry no βN weight.
- Choosing the Final entry: the same estimates were used on 27 October to choose which blend to enter in the Final,
  estimated S 0.8157 for this blend against 0.8155 and 0.8118 for the two alternatives.
- Calibration blends: `b2` contains our model `hyb`, whose q95 used the MAST envelope mask. `hyb` is not in B, and `fpb.py`
  keeps it out of the q95 pool; `b2` is one of the calibration points.

**Stopping epochs of Chris's models.**

- All 15 runs stop at an epoch chosen by synthetic-MAST S. FA, FC, V7C, Z15 and Z8 (11 runs) use `best_mast.pt`: of the
  epochs scored every 3 epochs, the one with the highest synthetic-MAST S among those with DIII-D S of at least 0.85
  (`chris/mast/MODELS.md`). The retrained runs stop at the same epochs (`chris/mast/retrain.py` reads each run's
  `best_mast.json`). The other 4 runs (`scr-F1-final-g4x5000`, `scr-F2-final-g4x5000-seed1`, `scr-G1-g4x5000-noise0.3`,
  `scr-G4-g4x5000-clip1`, which make up Z, Z2 and Z3) were scored only at epochs 15, 30 and 45 (`score_every: 15` in their
  run records). After training, epoch 30 had the highest synthetic-MAST S of the three for `scr-F1-final-g4x5000` (Z), and
  all four use epoch 30.
- Synthetic-MAST S is the starter kit's scoring code, unchanged, run on held-out synthetic MAST equilibria with their
  synthetic labels (`chris/mast/starter/my_experiments/nn/synth_S/scorer.py`). As on the platform, it bounds the boundary
  search with `mast_envelope.npz` on both the predicted and the reference maps. Apart from `hyb` in the calibration blend
  `b2` (above), this is the only use of the envelope that can affect B's predictions.
- No network reads the envelope. Every run record has `mast_mask: limiter`, so every run trains and predicts on the limiter
  region, and `mast_region.py` raises if anything opens `mast_envelope.npz` while the limiter is selected. The cache scripts
  `chris/mast/build_caches.sh` and `build_mast_test_inputs.sh` also write two feature files on the envelope region
  (`features_v5_<split>.npy`, `features_v5_<split>_sol0.5.npy`); no model reads them, and `build_mast_sol_half.py` uses them
  only for a row-layout check and a printed comparison.
- After submission, `zara/mast/r1/checks/score_bound/` runs the same official scoring code on our own prediction files to
  bound the effect of rebuilding; it does not change any prediction.

## Files

| Item | Where |
|---|---|
| Submitted files | `diii_d_private_test.npz` SHA256 `86e6a3726ff4d7a050bf7560a2d6df19c49a4ca1ce817f704da7492ac0acc153`; `mast_private_test.npz` SHA256 `3300302e7c69325554ee182b103bc6defeec07495d284180c333d07963f13fb3` |
| Predictions repository (pointer route) | `zenithon-zendaya/fec-final-sub5` @ `3a2a2a269053df11ac3c22ab5fdb184c5dfda6b7` |
| Shipped weights | DIII-D: `zenithon-zendaya/diii-d-model-weights` @ `c2e8c4a3` (`zara/diii-d/weights_manifest.json`, `zara/diii-d/fetch_weights.py`); MAST: `zenithon-zendaya/mast-model-weights` @ `8616a376` (`data/weights_manifest.json`, `data/fetch_weights.py`); r1: `zenithon-zendaya/zara-r1-weights` @ `5ca03a54` (networks trained on the CPU-built caches; `zara/mast/r1/predict/r1_files.json`, `fetch_r1.py`), and r1's first-pass cache of the Final MAST inputs, `beckhamc/zara-r1-cpu-caches` @ `d9be687e` (built on the CPU from the Final inputs; `zara/mast/r1/predict/r1_final_firstpass.json`) |
| Pinned environments | `zara/environment/` (DIII-D) and `chris/environment/` (MAST) |
| Data | `data/fetch_official.py` (official, SHA256 per file) and `data/fetch.py` (synthetic MAST caches, SHA256 per file) |

## Reproduce on a new machine

Step by step in `REPRODUCE.md` (steps 0-6 with the letter B). In short: one A100 80 GB, 12 or more cores, about 128 GB RAM (less with fewer
workers, `REPRODUCE.md`) and 550 GB of disk; `bash prepare.sh B` and `bash train.sh B` retrain every network from scratch on r1's CPU-built caches, and `bash predict.sh B OUT_DIR`
regenerates both submitted files from the shipped weights (`--rebuilt` for the retrained ones).

## What to expect when B is retrained

Retrained networks are not bit-identical to the shipped ones, so the retrained files differ from the submitted ones, and
for MAST the difference is visible. We cannot score either file against the Final labels. We retrained every network of B
from scratch with this package on an A100 (`train.sh B`, 30 October to 1 November, `checks/a100-final/`) and measured:

- **DIII-D (Ch1-min):** the retrained file against the submitted one: ψ R² 0.9999981, largest single-frame difference 0.026
  in ψ, 0.42 in q95 and 0.096 in βN. Label-free bound on the change in S: **1.9e-3**, against the 0.01 allowed. The bound
  assumes every difference moves the prediction away from the labels, with the Development-board errors standing in for
  the Final ones.
- **MAST, the whole blend:** the retrained file against the submitted one: ψ R² 0.99980, largest single-frame difference
  0.24 in ψ, 6.7 in q95 and 1.40 in βN. Most of the difference comes from r1 (ψ R² 0.9965 against its shipped version).
  Two trainings of the same model make different errors frame by frame and score alike,
  so these differences do not by themselves mean a change in S, and no MAST labels exist to measure it on the Final shots.
  We measured it on labelled synthetic MAST data instead: the shipped and the retrained networks, each through the package's
  `predict.sh B` and the same blend, on the held-out parts of 15 synthetic MAST sets (11,360 cases from two different
  solvers, label columns removed before prediction). S changed by **−0.0033 to +0.0007** from set to set, and by −0.0004
  pooled; ψ, 55 % of S, by at most 0.0005 on any set. The retrained blend scores slightly lower on the gsolve sets, the
  hardest for these models, and the drop grows as S falls (−0.0017 at S 0.76 to −0.0033 at S 0.69).
- **r1's four members, averaged** (without r1's prediction-time field correction and post-processing), on Zara's internal
  validation set (section "r1 (Zara Ercan's MAST model)"): S 0.8416 retrained against 0.8404 shipped; each of the four
  members within 0.0025 of its shipped version.

Earlier retrains agree: Ch1-min retrained on an RTX PRO 6000 gave ψ R² 0.9999982 against the submitted DIII-D file; each of
Chris's 30 MAST training runs retrained on RTX PRO 6000 GPUs scored within 0.003 of the original on synthetic MAST; r1
retrained from rebuilt training caches scored 0.8405 against 0.8404 (rows in "Checks").

A label-free bound on the Final MAST shots, of the kind used for DIII-D, is too loose to settle the 0.01 check for MAST: for
the file with only r1 retrained it is already 0.05 to 0.09, because it assumes every difference moves the prediction away
from the labels. The synthetic sets measure the change directly. They cannot give the Final score, and their equilibria are
synthetic, but on every one of them, including the gsolve sets on which these models score lowest (S 0.69 to 0.76), the
change stays within a third of the tolerance.

## Checks

| Check | Result |
|---|---|
| **Shipped weights, clean copy** (27 October, job 11011): `PARTS=mast bash predict.sh` on all Final MAST shots from a clean copy (`git archive`) of an earlier version of this package, whose r1 prediction scripts differ from this one only in file paths, Zara's environments built from the pinned specs, Chris's `fusion-eq` environment, Final data checked against Hugging Face's SHA256, Chris's shipped weights downloaded from `mast-model-weights` @ `8616a376` and r1's from `zara-r1-weights` @ `5ca03a54` (`fetch_r1.py`), all checked by SHA256, Final MAST inputs rebuilt with `chris/mast/build_mast_test_inputs.sh` (coil pictures and linear-decoder matrices from Chris's cache, SHA256 equal to his recorded rebuild, `chris/mast/checks/HASHES.md`), r1's input cache rebuilt by `predict_r1.sh`, one RTX PRO 6000 | **`mast_private_test.npz` byte-identical to the submitted file**: 3,615 of 3,615 arrays (`3300302e…`); `predict.sh` 1 h 41 min |
| r1's first-pass cache of the Final inputs: `predict_r1.sh` reuses it when complete, so the clean-copy run read the existing one. Rebuilt separately from the raw Final inputs on the CPU (jobs 10953, 10954) | byte-identical to the caches used: input cache 1,206 of 1,206 files, first-pass cache 1,205 of 1,205 files. A second run of r1's chain on these caches (job 10995) is byte-identical at every stage |
| r1's first pass on 10 Final MAST shots with other settings, against the submitted run's cache (job 11384, 30 October) | 12 threads with the shots grouped differently: 8 of 10 files byte-identical; 4 threads: 0 of 10; 1 thread: 3 of 10. The files that differ do so by about 1e-5 m in the magnetic-axis position and 1e-3 in elongation |
| r1's first pass of all 1,205 Final MAST shots rebuilt with 8 shards of 4 threads (submitted run: 12), then r1's chain with the shipped networks and the blend with Chris's shipped member files, one RTX PRO 6000 (job 11386, 30 October) | 2 of 1,205 first-pass files byte-identical. `mast_private_test.npz` against the submitted file: largest difference 0.0031 in ψ and 0.071 in βN, q95 identical; label-free bound on the change in S 2.5e-4 (`zara/mast/r1/checks/score_bound/`; Development-board R² as stand-ins; 1.1e-3 with no assumption). A rebuilt first pass is therefore not guaranteed within 1e-4, and `REPRODUCE.md` step 4 downloads the submitted run's cache |
| Chris's 8 MAST models: member files predicted from the shipped weights (`predict_model.py`) against the member files the blend reads | byte-identical, 3,615 of 3,615 arrays each |
| r1 member file: the shipped networks (the files on `zara-r1-weights` @ `5ca03a54`, SHA256 checked at upload) through r1's prediction chain on the 1,205 Final MAST shots, one RTX PRO 6000, code differing from this package only in file paths | the r1 file the blend reads (SHA256 `e8dd0012…`) |
| Blend: `blend/run_blend.py` with `blend/weights_chris8r1.json` on those member files | the submitted `mast_private_test.npz` (`3300302e…`) |
| Organisers' validator (`validate_submission.py`) on both files | OK: DIII-D 879 shots, MAST 1,205 shots |
| DIII-D file, shipped weights, clean copy of the package (27 October, job 10966), inference entrypoint, one RTX PRO 6000 | byte-identical to the submitted file: 2,637 of 2,637 arrays (`86e6a372…`) |
| `REPRODUCE.md` steps 1-5 for B with the shipped weights, in a sandbox that hides our cluster's own data folders, so that everything comes from the package's downloads (30 October), one RTX PRO 6000 | both files byte-identical to the submitted ones: DIII-D 2,637 of 2,637 arrays (`86e6a372…`), MAST 3,615 of 3,615 arrays (`3300302e…`); validator OK on both; `predict.sh B` 2 h 47 min on a shared node |
| **Shipped weights on an A100**, 30-31 October: `REPRODUCE.md` steps 1-4 on a new GCP machine with two A100-SXM4-80GB (driver 535), whose 24 vCPUs were shared with two trainings (`checks/a100-final/`) | validator OK on both files; neither is byte-identical, because the A100 rounds the networks' forward passes differently from the RTX PRO 6000 the submitted files were made on. DIII-D: 0 of 2,637 arrays identical, largest difference ψ 0.031, q95 0.025, βN 0.009, ψ R² 0.99999992; estimated change in S **1e-5** (7.5e-6, method as for A in `checks/a100-final/README.md`), strict label-free bound 5.7e-4. MAST: 0 of 3,615 arrays identical, largest difference ψ 0.003, q95 0.115, βN 0.033, ψ R² 0.9999999995; estimated change in S **2e-5** on average, 6e-5 one standard deviation above and 1.0e-4 two above, strict label-free bound 1.5e-3 (method in `checks/a100-final/README.md`: no labelled MAST data exists, so the estimate assumes the A100's rounding differences are unrelated to the errors against the labels; tool `zara/mast/r1/checks/score_bound/`). `predict.sh B` 7 h 23 min on the shared CPUs |
| **Retrained on an A100**, 30 October to 1 November, on the same machine: `prepare.sh B` (7 h 41 min on the shared CPUs), `train.sh B` on its own GPU (17 h 41 min), then `predict.sh B --rebuilt` and `steps/5_compare.sh` (`checks/a100-final/`) | GPU training **17.68 GPU-hours**: Ch1-min 3.18, Chris's 15 runs 11.07, r1 3.43 (17.84 on 23-25 October). Both retrained files pass the validator; 0 arrays identical. DIII-D: largest difference ψ 0.026, q95 0.42, βN 0.096; ψ R² 0.9999981; label-free bound on the change in S **1.9e-3** (`zara/diii-d/checks/noise_bound.py`). MAST: largest difference ψ 0.24, q95 6.7, βN 1.40; ψ R² 0.99980; per member against its shipped version, ψ R² 0.9988 to 0.99993 for Chris's 8 models and 0.9965 for r1. This run found that `predict.sh B --rebuilt` read Chris's retrained checkpoints from the shipped checkpoints' folder; the run pointed it at the right folder, and this package passes that folder itself |
| **r1 retrained on the A100, on Zara's internal validation set** (3,816 held-out FreeGSNKE cases; 1 November, `checks/a100-final/r1_validation/`) | S of the four-member average 0.8416, against 0.8404 for the shipped networks. Per member, retrained against shipped: `rt_ft` 0.8379 / 0.8356, `rt_mix` 0.8369 / 0.8348, `rt_ftw3` 0.8374 / 0.8372, `rt_nonoise` 0.8295 / 0.8293. `rt_nonoise` again ended on a poor last epoch on DIII-D held-out frames, in a part of the network r1 does not use on MAST (`zara/mast/r1/checks/a2-ultragpu-1g/RT_NONOISE_SPIKE_RESULTS.md`); no frame needed r1's fallback |
| **The whole MAST blend on 15 labelled synthetic MAST sets**, shipped against retrained-on-A100 (1 November, our AWS cluster, `checks/a100-final/synthetic_blend/`): the held-out parts of every synthetic MAST set we have, 11,360 cases (FreeGSNKE: Zara's internal validation, `g4_20000`, `g4_5000`, `g4_v3`, `g4_v4`, `g4_v4b`, `g4_v2`, `g4_v6`, `g4_v7`; gsolve: `gs1000`, `gst1000`, `gsc1000`, `gsc2z1000`, `gsc3_1000`, `gsx8`), label columns removed, laid out as the Final MAST split; `predict.sh B` (MAST part) of this package with the shipped weights and with `--rebuilt`; the competition's scoring code against the labels, one score per set | Change in S from shipped to retrained: −0.0033 to +0.0007 per set (FreeGSNKE sets −0.0018 to +0.0007; gsolve sets −0.0033 to −0.0017), −0.0004 pooled. ψ term at most 0.0005 in S on any set; the largest changes are in q95 and βN on the gsolve sets. Per-set table: `checks/a100-final/README.md`, section "Retrained models" |
| Ch1-min retrained from scratch with an earlier version of this package's DIII-D code (22 October, one RTX PRO 6000, GPU steps 2.05 h), then predicted on the 879 Final DIII-D shots | ψ R² 0.9999982 against the submitted DIII-D file (`86e6a372…`); largest single-frame difference 0.024 in ψ, 0.61 in q95, 0.10 in βN |
| Chris's 30 MAST training runs retrained from scratch (`chris/mast/retrain.py`, one RTX PRO 6000 per run), scored on synthetic MAST with the scorer the original runs used | S within 0.003 of the original for every run, mean difference −0.00015; the 15 runs behind this blend: −0.0030 to +0.0007 (`chris/mast/checks/rebuilds/SUMMARY.md`) |
| r1's training caches rebuilt and r1 retrained, in a sandbox that hides our cluster's own data folders (30-31 October): `build_r1_caches.sh` on the CPU (an earlier version with the same builders, shard counts and threads; the 640 reference files missing from the bundle built with `build_refs.py` as in its step 0), about 6.3 h on 48 cores; then `train_r1.sh full` on one RTX PRO 6000; then r1's chain on the 1,205 Final MAST shots with the submitted first-pass cache, blended with Chris's 8 shipped member files | Caches: 167,014 of 168,184 files byte-identical to the caches behind the shipped r1 networks. All MAST caches and the DIII-D holdout caches are identical; 592 of 6,337 DIII-D `_s2p7` first-pass files and 578 of their auxiliary-target files differ, all from how the shipped `_s2p7` cache was built (`zara/mast/r1/cpu_caches/README.md`): 299 of the 472 shots its first run built on another node, and 293 of the 4,139 shots its second run solved in a different group of 8 than a run in one go would; the 1,726 shots it solved in the same group are all identical. Training: 1.8 GPU-hours. Zara's internal validation set (3,816 held-out FreeGSNKE cases, section "r1 (Zara Ercan's MAST model)"): retrained r1 S 0.8405, shipped r1 0.8404. Final MAST file against the submitted one: ψ R² 0.99996, largest difference 0.20 in ψ and 1.16 in βN, q95 identical. Label-free bound on the change in S (`zara/mast/r1/checks/score_bound/`): 0.050 with Development-board R² as stand-ins, 0.090 with no assumption. This bound is a worst case: on the validation set, where S can be computed, the retrained r1 scores 0.0001 above the shipped one (0.840469 against 0.840365) |

The retraining tolerance of 0.01 in S is scored with the Final labels, which we do not have. Section "What to expect when B
is retrained" sums up what we measured instead: label-free bounds for DIII-D, and for MAST the change in S on labelled
synthetic data, where the label-free bound of a retrain is much larger than the change it stands for.
