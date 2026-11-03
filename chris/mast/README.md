# Chris's MAST models

Thirteen models used by the team's MAST blends (Z, Z2, Z3, Z6b, Z7, Z8, Z9qb, Z15, Z17, Z18, FA, FC, V7C), built from
30 training runs. `MODELS.md` describes each model; `models.json` is the machine-readable definition (member runs and
checkpoints in averaging order, post-processing recipe).

## Settings

| Variable | What it points at |
|---|---|
| `FEC_DATA_ROOT` | the official challenge data: a folder holding `data/<split>/` (from `data/fetch_official.py`) |
| `FEC_CACHE` | derived caches (`build_caches.sh`) and `synth/` (from `data/fetch.py --root $FEC_CACHE`) |
| `FEC_WEIGHTS` | folder of run folders holding the shipped checkpoints (default: the original runs on `/fsx`) |
| `FEC_MAST_SPLIT` | the MAST test set; set by `predict_model.py`, `build_mast_test_inputs.sh` |

Environment: `chris/environment/` (conda environment `fusion-eq`, Python 3.11, torch 2.11 + CUDA 12.8).

## From scratch

```bash
export FEC_DATA_ROOT=<data>/development FEC_CACHE=<cache>
python data/fetch_official.py --root <data> --splits diii_d_train diii_d_public_test mast_public_test mast_private_test
python data/fetch.py --root $FEC_CACHE                          # synthetic training caches, checked by SHA256
bash chris/mast/build_caches.sh 24                              # derived caches, ~20 min on 24 CPUs
bash chris/mast/train.sh                                        # all 30 runs, ~17.9 GPU-hours on one RTX PRO 6000
# Final inputs: their own cache folder, holding the same coil pictures
FEC_DATA_ROOT=<data>/final FEC_CACHE=<cache_final> bash chris/mast/build_mast_test_inputs.sh mast_private_test 24
FEC_DATA_ROOT=<data>/final FEC_CACHE=<cache_final> bash chris/mast/predict.sh mast_private_test <out> --rebuilt \
  --weights chris/mast/multimachine/my_experiments/nn/runs
```

With the shipped weights instead of retraining, drop `train.sh` and `--rebuilt`:
`python data/fetch_weights.py --root <weights>` and `export FEC_WEIGHTS=<weights>`.
`<cache_final>/features_v2/` needs `pictures_mast.npz` and `pictures_d3d.npz` from the main cache.

## What has been checked (details in `checks/`)

| Step | Check |
|---|---|
| Code copy | every file byte-identical to the frozen trees it came from (`SOURCES.md`, `gsolve/SOURCES.md`) |
| Synthetic data | downloads and the 5 rebuilt merged caches match `/fsx` by SHA256; the gsolve caches rebuild byte for byte from the raw solver output |
| Derived caches | `build_caches.sh`'s steps rebuild all of them byte for byte from the official data |
| Official data | the copies used match the pinned revisions' published SHA256 |
| Retraining | all 30 runs retrained: synthetic-MAST S within 0.003 of the originals (organisers allow 0.01) |
| Prediction | all 13 models from the original checkpoints byte-identical to the files the team blends read |
| Training time | Chris's part of the within-budget blends: 5.6 h (one budget) and 5.8 h (per challenge) on one RTX PRO 6000 |
