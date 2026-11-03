# A100 run, 23-25 October 2026 (dedicated machine)

Timing report of the run behind the declared GPU-hours (Zara Ercan, 25 October 2026). One GCP machine with one
A100-SXM4-80GB trained, one after another, v13 (the DIII-D model of submission A) and the 15 MAST runs of Chris's models in
submission B, with an earlier version of this package's training scripts. The report also covered combinations of models
that are not under review; this copy keeps the parts on A and B, with the measurements unchanged.

Since this run, two steps of v13's training moved: the out-of-fold q-curves and map features now run on the CPU alongside
GPU training, and the input export runs on the CPU. B's declared time comes from separate runs on the same machine type:
Ch1-min (the DIII-D model of B) with the same 15 MAST runs (3.23 + 10.96 GPU-hours; `fec-B/`), and r1 (3.65;
`zara/mast/r1/checks/a2-ultragpu-1g/`).

Both declarations were measured a second time, on 30 October to 1 November, with this package's `train.sh` on another A100
machine: A 20.77 GPU-hours, B 17.68 (`../a100-final/`, section "Training time: two A100 measurements", which sets the two
measurements side by side).

## Setup of the runs

- GCP us-central1, on-demand, through SkyPilot.
- VM: `a2-ultragpu-1g`, one NVIDIA A100-SXM4-80GB, driver 535.216.01, 12 vCPUs (Intel Xeon 2.2 GHz, 6 cores with 2 threads each), 170 GB RAM, 600 GB disk.
- Environments and data as in `REPRODUCE.md`: pinned conda environments, official data from `Sophelio/fusion-equilibrium-challenge`, synthetic MAST data from `zenithon-zendaya/mast-synthetic-data`. Only the training splits were downloaded.
- Command: the training entrypoint, wrapped in `/usr/bin/time -v`, with `FEQ_WORKERS=12`, `CUDA_MEM_CAP_GB=80`, GPU sampled every 10 s.
- Timing window: from the training entrypoint's start line to its end line. Downloads, environment builds and the MAST derived-cache build (`build_caches.sh`) happen before it.

## Measured

v13, then the MAST blend's 15 runs, one after another from 10:40:57 UTC on 23 October to 02:40:10 UTC on 25 October, exit
status 0. The two parts belong to different submissions (A and B) and count separately toward each one's 24 GPU-hours.

| Part | Wall time | of which GPU steps |
|---|---|---|
| DIII-D v13 (76 steps) | 29.00 h | about 20.9 h |
| MAST blend (15 runs) | 10.99 h | 10.99 h |

### v13 DIII-D, by step type

Each step's kind was checked in its code (section "CPU or GPU: how each step was classified").

| Step group | Kind | Steps | Time | Per step |
|---|---|---|---|---|
| Input caches (`cache_nn`, `cache_px`, `cache_hist`, train and test) | CPU | 6 | 0.70 h | 71 to 932 s |
| `targets_cons`, `targets_geo` | CPU | 2 | 2.36 h | 5,098 s and 3,409 s |
| `ridge_fit` | CPU | 1 | 0.36 h | 1,281 s |
| `r3c_xb`, `r3c_hist*`, `r3c_folds`, `r3c_mapf` | CPU | 8 | about 2.08 h | `r3c_xb` 5,227 s |
| `r3c_oofqc` (5 folds) | CPU | 5 | 2.62 h | 1,887 s |
| **CPU subtotal** | | **22** | **about 8.1 h** | |
| `r3c_export` (Ridge base map and first-pass features of every training frame) | GPU | 1 | about 2 min | 59 s on the RTX PRO 6000; its exact A100 time was not read |
| 12 ψ networks (`psi`, 60 epochs) | GPU | 12 | 7.67 h | 2,301 s |
| 3 sequence heads (`seqhead`) | GPU | 3 | 0.47 h | 565 s |
| 8 functional heads (`head_scog`, `head_scog_ctx`, `head_auxg`) | GPU | 8 | 4.27 h | 1,802 to 2,264 s |
| 5 out-of-fold ψ networks (`r3c_oof`) | GPU | 5 | 2.52 h | 1,815 s |
| 12 q95 / βN MLPs (`r3c_prod_*_hmD`) | GPU | 12 | 3.45 h | about 1,035 s |
| 6 q95 / βN sequence networks (`r3c_prod_seq_hmD`) | GPU | 6 | 0.75 h | 452 s |
| 6 variant-D networks (`r3c_prod_*_D`) | GPU | 6 | 1.70 h | about 1,020 s |
| **GPU subtotal** | | **53** | **about 20.9 h** (20.84 h plus `r3c_export`) | |

The CPU and GPU totals together are the 29.0 h wall time of the v13 part.

GPU utilisation over the whole 40 h window averaged 69% (the CPU steps are in the average). During `psi_1`, the first GPU
step, it averaged 92%. Peak GPU memory over the run was 70.5 GB.

### CPU or GPU: how each step was classified

Checked in the scripts each step runs (`zara/diii-d/train_ch1.sh`) and in every local module they import:

- **CPU**: `preprocess.py`, `preprocess_px.py` (input caches), `hist_features.py`, `ref_train.py` and `geo_train.py`
  (label-side targets), `train_parallel.py` and `export_ridge.py` (Ridge base, fitted on the CPU), `xb_targets.py`,
  `hist_features_ch1h.py`, `hist_feats.py`, `hist_from_ch1h.py`, `make_folds.py`, `qcurve_oof.py` and `assemble_oof.py`. None of
  them, nor the modules they import, uses torch or CUDA (numpy, scipy, pandas and pyarrow only).
- **GPU**: every network training (`train_c1h.py`, `train_c1c.py`, `train_c1q.py`, `train_sq.py`, `train_s.py`), and
  `export_inputs.py` (step `r3c_export` in v13 and Ch1-safe, `d3_export` in Ch1-min), which computes the Ridge base map and
  first-pass features of every training frame with torch on CUDA. The q95 / βN networks are fitted to its output, so it counts
  toward the ceiling.

### MAST blend, 15 runs

All 15 runs are GPU training (`chris/mast/retrain.py`).

| Run | A100 time | RTX PRO 6000 reference |
|---|---|---|
| `scr-F1-z15mix-s0` | 46.5 min | |
| `scr-F2-z15mix-s1` | 44.1 min | |
| `scr-F6-z15mix-s2` | 44.1 min | |
| `scr-F3-only` | 23.4 min | |
| `scr-V7C-v7only` | 21.6 min | 17.1 min |
| `scr-F1-final-g4x5000` | 66.0 min | |
| `scr-mastZ15-v6mix-seed0` | 50.2 min | 37.1 min |
| `scr-mastZ15-v6mix-seed1` | 40.3 min | |
| `scr-mastZ15-v6mix-seed2` | 40.4 min | |
| `scr-mastZ15-v6mix-seed3` | 40.3 min | |
| `scr-G1-g4x5000-noise0.3` | 65.8 min | |
| `scr-F2-final-g4x5000-seed1` | 65.9 min | |
| `scr-G4-g4x5000-clip1` | 65.9 min | |
| `scr-V1-v4only` | 22.5 min | |
| `scr-W4-v4only-ep12-seed1` | 22.5 min | |
| **Total** | **10.99 h** | 7.96 h (sum of the RTX measurements in `chris/mast/checks/rebuilds/SUMMARY.md`) |

## A100 against RTX PRO 6000

RTX numbers are the per-step times measured in the 16 October one-GPU rebuild of v13
(`zara/diii-d/checks/train_times_v13_rtx.tsv`) and in `chris/mast/checks/rebuilds/SUMMARY.md`.

| Item | RTX | A100 (measured) | Ratio |
|---|---|---|---|
| v13 GPU total | 12.8 h | about 20.9 h | 1.63 |
| MAST blend total | 7.96 h | 10.99 h | 1.38 |
| ψ network, 60 epochs | 1,335 s | 2,301 s | 1.72 |
| Sequence head | 315 s | 565 s | 1.79 |
| Functional heads | 1,116 to 1,325 s | 1,802 to 2,264 s | 1.64 overall |
| Out-of-fold ψ network | 1,059 s | 1,815 s | 1.71 |
| q95 / βN MLP (`hmD`) | 715 s | about 1,035 s | 1.45 |
| q95 / βN sequence network | 265 s | 452 s | 1.71 |
| `scr-V7C-v7only` | 17.1 min | 21.6 min | 1.26 |
| `scr-mastZ15-v6mix-seed0` | 37.1 min | 50.2 min | 1.35 |
| CPU steps (caches, targets, Ridge, `xb`, `oofqc`) | | | 3.2 to 4.8 |

The CPU steps are slow because the 12 vCPUs are hyperthreads. The RTX reference used 16 cores.
