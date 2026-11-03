# Checks against the frozen code

## Reading this log

Each section below states what was compared with what and the outcome, and names the raw result file (in this folder)
that the comparison printed. `HASHES.md` holds the SHA256 of both sides of every byte-level comparison, so an
IDENTICAL can be confirmed later from the fingerprints, after the scratch copies on `/fsx` are gone.

| Claim | Kind of check | Evidence |
|---|---|---|
| Code copied unchanged from the frozen trees | `cmp` per file | `../SOURCES.md`, `../gsolve/SOURCES.md` |
| Synthetic data downloads and merged caches | SHA256 per file against `data/manifest.json` | `data/fetch.py` output; `rebuild_gsolve_caches.result.txt` |
| Official data copies | SHA256 per file against Hugging Face's published values | `step2/step2_5_official_data.result.txt` |
| Derived caches rebuild from the official data | `cmp` per file, all shards | `step2/step2_4_caches.result.txt`, `HASHES.md` |
| Path and split edits leave outputs unchanged | `cmp` of rebuilt candidate files | `step2/step2_1_*`, `step2_2_*`, `step2_3_predict_public.result.txt`, `HASHES.md` |
| 13 models from the original checkpoints | `cmp` against the files the blends read | `step2/step2_3_predict_public.result.txt`, `HASHES.md` |
| Final-input predictions | organisers' validator; ranges; SHA256 of each file | `step2/step2_3_predict_private.result.txt` |
| 30 runs retrained | synthetic-MAST S, original against retrained | `rebuilds/SUMMARY.md` |
| Retrained models and blends | pooled R² and largest difference, no labels | `step1_models_and_blends.result.txt` |
| Shipped weights | SHA256 per file; predictions from the download `cmp`-identical | `step2/step2_6_weights.result.txt`, `HASHES.md` |
| From-scratch dry run | environment, statistics, training, prediction | `step2/step2_8_dryrun.result.txt`, `HASHES.md` |

To re-run a check, use the script named in its section (`postproc_check.sh`, `predict_all_public.sh`,
`rebuild_gsolve_caches.py`, `compare_rebuild.py`, `compare_predictions.py`, `summarize_rebuilds.py`); they write to
scratch folders on `/fsx`. `hash_log.py` regenerates `HASHES.md` (it reports files that no longer exist as missing).
`predict_model.py --keep-steps` recreates the intermediate files a check compared.

| Check | Script | Result |
|---|---|---|
| The two gsolve training caches rebuild from the downloaded `gsolve_raw/<set>/gen4_layout` with `synth_cache_seq.py` and `synth_shape_targets.py` (unchanged) | `rebuild_gsolve_caches.py` (Slurm: `rebuild_gsolve_caches.sbatch`) | `rebuild_gsolve_caches.result.txt` (job 9650): `arrays.npz` and `shape_targets.npz` identical to the `/fsx` originals for cond1000 and cond3_1000; `manifest.json` identical except the two recorded source paths |

The merged caches are checked by `data/fetch.py` itself on every run.

## Rebuilds of single training runs

`chris/mast/retrain.py RUN` reruns a recorded run with the frozen `train_mm.py` up to the epoch its checkpoint came
from (`--stop-epoch`, same learning-rate schedule), with the scoring jobs off. `compare_rebuild.py RUN` compares
training metrics and weights; `synthS_*-rebuild.json` is the synthetic-MAST S of the rebuilt checkpoint from the same
scorer the original run used (`ensemble_synth.py --sets g4_20000`). One RTX PRO 6000 per run, 2 runs side by side.

| Run (model) | Stop epoch | Wall clock | Peak GPU memory | DIII-D val R²ψ (orig → rebuild) | Synthetic-MAST S (orig → rebuild) |
|---|---|---|---|---|---|
| `scr-V7C-v7only` (V7C) | 9 | 17:04 | 39.7 GB | 0.997774 → 0.997810 | 0.785740 → 0.785225 (−0.0005) |
| `scr-F3-only` (FC) | 9 | 18:22 | 41.1 GB | 0.998368 → 0.998362 | 0.738107 → 0.738572 (+0.0005) |

Weights are not identical (relative RMS difference 8 % and 2 %): GPU training is not bit-for-bit repeatable.

## Step 2.1: data and cache locations are settings

`FEC_DATA_ROOT` (the official challenge data) and `FEC_CACHE` (derived caches) replace the fixed `/fsx` paths in the
ten files that defined them; the defaults are the old paths. `retrain.py` passes them to `train_mm.py` when set.

| Check | Result |
|---|---|
| Every changed constant, evaluated from the source with the settings unset and set | old paths when unset, the given paths when set (all ten files) |
| FC's post-processing rebuilt from its kept `-base` folder (`step2/postproc_check.sh`: q95 rule, then βN V2) | byte-identical to the original candidate before the change, after it with the settings unset, and with them pointing at a separate folder of links; fails with them pointing at an empty folder (`step2/step2_1_paths.result.txt`) |
| All 30 `retrain.py` commands through `train_mm.py`'s parser, settings unset | identical to the recorded configs |

## Step 2.2: no code path reads the old worktrees

The scorer and post-processing libraries (`qlib.py`, `bn_lib.py`, `physics_suite.py`), the limiter-mask fallback in
both `mast_region.py` copies and `train_mm.py`'s scorer folder now point at this repository's own copies, found from
each file's location. `retrain.py` passes the repository's copy of the recorded `--pictures` file (byte-identical).

| Check | Result (`step2/step2_2_frozen_paths.result.txt`) |
|---|---|
| FC and Z15 post-processing rebuilt from their `-base` folders, now with only this repository's code | byte-identical to the original candidates |
| One epoch of V7C through `retrain.py` (repository pictures and limiter mask) against epoch 1 of the earlier V7C rebuild | training loss within 5e-6, DIII-D R²ψ within 2e-5, R²q95/βN within 4e-5: the size of GPU run-to-run noise |

Still pointing outside the repository, handled in later steps: `synth_S/paths.py` (in use by the scoring jobs while this
was done), `splice_q95.py` (reads candidate folders; step 2.3), the legacy `build_*.sh` / `run_*.sh` launchers and the
gsolve recipes.

## Step 2.3 (part 1): one prediction command per model, for any MAST test set

`chris/mast/predict_model.py MODEL --split SPLIT` runs every member checkpoint (`models.json`: members in the original
averaging order and the post-processing recipe), averages them and applies the same post-processing as the original
build scripts, writing `<out>/MODEL/<split>.npz`. The MAST test set is the setting `FEC_MAST_SPLIT`
(default `mast_public_test`), read by `predict.predict_mast`, the MAST input builders, `postproc_qb.py`, `apply_cc.py`,
`apply_bn_variant.py` and `physics_suite.py`; caches that never carried the split in their name get a `_<split>`
suffix only for other splits, so every public-test file name is unchanged. `build_features.py --only-config SPLIT` and
`build_thomson_raw.py --mast-only` build one test set without touching the DIII-D caches.

| Check | Result (`step2/step2_3_predict_public.result.txt`) |
|---|---|
| All 13 models predicted on `mast_public_test` from the original checkpoints | byte-identical to the original files (candidate folders, the Hugging Face copies of Z, Z2, Z6b, Z7, Zara's copies of Z3, Z8); the 6 kept `-base` intermediates identical as well |
| FC post-processing alone after the split changes, default settings | byte-identical |

### Step 2.2 (completed): the synthetic-MAST scorer's paths

`synth_S/paths.py` finds the scorer and model code from its own location; the interpreter (`FEC_PYTHON`), the cache
(`FEC_CACHE`) and Zara's raw held-out sets (`FEC_GEN4_ROOT`) are settings. V7C's original checkpoint scored this way
gives exactly the score its original run recorded (`step2/step2_2b_synthS_paths.result.txt`). `extra_sets.json` (extra
practice sets the package does not use) keeps its recorded folders.

## Step 2.4: every derived cache rebuilds from the official data

`chris/mast/build_caches.sh` builds, into an empty `FEC_CACHE`, every cache derived from the official data:
the DIII-D train memmaps and shards (`build_cache.py`), the coil pictures and `features_v5` (`build_features.py`),
raw Thomson (`build_thomson_raw.py`), the linear-decoder matrices (`lindec_build.py`) and the MAST public-test inputs
(`build_mast_test_inputs.sh`). Synthetic caches come from `data/fetch.py`; normalisation statistics from `train_mm.py`.

| Check | Result (`step2/step2_4_caches.result.txt`) |
|---|---|
| Each step run with the commands in `build_caches.sh` into an empty folder, outputs compared with `/fsx/chris.beckham/fusion_cache` | all 29 files (including the 26 GB `psi.npy`) and all 7,041 shards byte-identical; about 20 minutes with 24 workers |

The steps were run one at a time with those commands; one uninterrupted run of `build_caches.sh` is part of the
from-scratch dry run (step 2.8).

## Step 2.5: the official data, pinned

`data/fetch_official.py` downloads the organisers' datasets at fixed revisions (`Sophelio/fusion-equilibrium-challenge`
@ `1e280905b8` for development, `ganatma/fusion-equilibrium-challenge-final` @ `965040c1` for the Final inputs) and
checks every file against the SHA256 Hugging Face publishes; `--verify-local` checks an existing copy.

| Check | Result (`step2/step2_5_official_data.result.txt`) |
|---|---|
| The copies every cache and prediction above was built from | all 10,326 files match: diii_d_train 7,041, diii_d_public_test 874, mast_public_test 1,206, mast_private_test 1,205 |

The full download (about 98 GB for the development splits) has not been run here: /fsx was nearly full.

## Step 2.8: dry run from scratch on the AWS cluster (one model)

Fresh clone, new conda environment from `environment.yml`, derived caches rebuilt from the official data (step 2.4),
synthetic caches downloaded with `data/fetch.py`, then V7C trained (`retrain.py`), scored and predicted
(`predict_model.py`). Result: `step2/step2_8_dryrun.result.txt`.

| Check | Result |
|---|---|
| Environment | builds in 4 min; `pip freeze` identical to the pinned list |
| Normalisation statistics written on first use | byte-identical to the originals |
| V7C trained from scratch | 18:01; synthetic-MAST S 0.7838 against the original's 0.7857 (−0.0020) |
| V7C's original checkpoint through the fresh setup | byte-identical to the original candidate |
| V7C retrained from scratch against the original file | ψ R² 0.99972, q95 R² 0.9936, βN R² 0.9994 |

Not covered yet: the full official-data download (about 98 GB; /fsx was nearly full, so the verified local copy was
used), one uninterrupted `build_caches.sh` run, the other 12 models from scratch, and shipped weights from Hugging Face
(step 2.6, awaiting approval). The environment was built on the same cluster, not a clean machine.

## Step 2.3 (part 2): the Final inputs

The Final MAST inputs (`ganatma/fusion-equilibrium-challenge-final` @ `965040c1`, 1,205 shots) built into their own cache
with `build_mast_test_inputs.sh mast_private_test`, then all 13 models predicted from the original checkpoints.

| Check | Result (`step2/step2_3_predict_private.result.txt`) |
|---|---|
| The organisers' `validate_submission.py` (starter `4b6af06`, `--config mast_private_test`) | all 13 files pass: 1,205 shots, shapes, keys and dtype valid |
| q95, βN and ψ ranges against the same model on the public test | close for every model; no non-finite values; float32 throughout |

The 13 files are at `/fsx/chris.beckham/fec_upload/step2/pred_private/<model>/mast_private_test.npz` (SHA256 in the
result file), ready for the team blend.

## Retrained models and blends against the originals

Every model predicted on `mast_public_test` from the retrained checkpoints, compared with the same model from the
original checkpoints; then Zara's two within-budget blends rebuilt with her `blend_build2.py` and weights, from the
original model files and from the retrained ones (`step1_models_and_blends.result.txt`). No labels are used.

| Comparison | Result |
|---|---|
| Blends from the original model files | array-for-array identical to Zara's submitted blends (B16+FC, B16+V7C) |
| Blends with Chris's 30 retrained runs | against the submitted blends: ψ R² 0.99997–0.999997, q95 R² 0.998–0.9985, βN R² 0.999995 |
| Single models | ψ R² 0.9996–0.999999; βN R² 0.9967–0.99996; q95 R² 0.973–0.99999 (lowest for the single-run models Z and Z6b's q95 taken from Z) |

## Step 2.6: shipped weights, pinned

The 30 checkpoints the 13 models use (2.89 GB) are in the model repository `zenithon-zendaya/mast-model-weights`
at `8616a376`, under `runs/<run>/<checkpoint>`. `data/weights_manifest.json` pins the revision and every file's SHA256;
`data/fetch_weights.py --root DIR` downloads them to `DIR/<run>/<checkpoint>`, the layout `FEC_WEIGHTS` expects.

| Check | Result (`step2/step2_6_weights.result.txt`) |
|---|---|
| Download into an empty folder | 30/30 checkpoints match the manifest |
| V7C and Z15 predicted from the downloaded weights | byte-identical to the original files |
