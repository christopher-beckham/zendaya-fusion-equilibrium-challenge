# A100 run of B's Ch1-min and MAST training, 23-25 October 2026

The run behind B's declared Ch1-min and MAST GPU-hours (3.23 + 10.96). Copied from Zara Ercan's timing report of 25 October,
written from the run's logs on the machine; the machine has since been deleted, and its `train_times` tables and logs were
not kept. Setup as in `../README.md` (same machine type, run at the same time on its own machine): `bash train.sh B` of an
earlier version of this package's training scripts, wrapped in `/usr/bin/time -v`, with `FEQ_WORKERS=12` and
`CUDA_MEM_CAP_GB=80`, GPU sampled every 10 s. Timing window: from the `train B start` line to the `train B end` line.

| | B |
|---|---|
| Wall time of `train.sh B` | 19 h 01 min, exit status 0 |
| DIII-D recipe | Ch1-min |
| DIII-D wall time | 8.06 h |
| of which GPU steps | 3.23 h |
| of which CPU steps | 4.83 h |
| MAST (15 runs) | 10.96 h |
| GPU-hours (DIII-D GPU steps + MAST) | 14.19 h |
| Peak GPU memory | 59.6 GB |
| Mean GPU utilisation over the window | 67 % |

## Ch1-min, 23 steps

| Step | Kind | A100 time | RTX reference |
|---|---|---|---|
| `cache_nn_train`, `cache_nn_test`, `cache_px_train`, `cache_px_test`, `cache_hist_train` | CPU | 920 + 106 + 730 + 85 + 584 s | 274 + 37 + 231 + 31 + 184 s |
| `targets_cons`, `targets_geo` | CPU | 5,054 + 3,385 s | 1,053 + 730 s |
| `ridge_fit`, `ridge_export` | CPU | 1,258 + 0 s | 383 + 1 s |
| `psi_e30_1` to `psi_e30_6` | GPU | 1,162 to 1,210 s each | about 727 s each |
| `seqhead_1` to `seqhead_3` | GPU | 508 to 514 s each | about 315 s each |
| `d3_export`, `d3_xb` | CPU | 57 + 5,193 s | 59 + 1,100 s |
| `d3_prod_xbmix_D`, `d3_prod_ctx5sm_D`, `d3_prod_ctx5_D` | GPU | 1,033, 1,012, 1,005 s | 645, 714, 709 s |
| **GPU total** | | **3.23 h** | **2.05 h** (ratio 1.58) |
| CPU total | | 4.83 h | about 1.1 h |

The CPU steps are slow because the 12 vCPUs are hyperthreads; the RTX reference used 16 cores. In the current package the
CPU steps run in `prepare.sh B`, and `d3_export` runs on the CPU.

## MAST, 15 runs

10.96 h. Run by run within 1.5 min of the per-run table in `../README.md` (10.99 h, the same 15 runs on the other machine).

## Later measurements of the same parts

`train.sh B` of this package on another A100 machine, 30 October to 1 November (`../../a100-final/`): Ch1-min 3.18
GPU-hours, the same 15 MAST runs 11.07, r1 3.43; 17.68 in all. r1's 3.65 of the declaration:
`zara/mast/r1/checks/a2-ultragpu-1g/`.
