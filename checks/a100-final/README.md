# A100 run, 30 October to 1 November 2026

`REPRODUCE.md` end to end for submissions A and B on a newly created machine with A100 GPUs: the shipped weights (steps
1-4), then every network retrained from scratch and used to predict again (steps 5-6). The run used an earlier version of
this package. Since then, apart from documentation and check records, three things changed: the `--firstpass-only` option
of `zara/mast/r1/predict/fetch_r1.py`, the label-free check tools under `zara/diii-d/checks/` and
`zara/mast/r1/checks/score_bound/`, and one fix this run found in `predict.sh` (section "Retrained models").

**Machine.** GCP a2-ultragpu-2g: two NVIDIA A100-SXM4-80GB (driver 535.216.01), Intel Xeon @ 2.20 GHz with 24 vCPUs, 334 GB
RAM, 1.2 TB disk (`logs/machine.txt`). It ran three copies of the package: the shipped-weights check, the retraining of A
and the retraining of B. A's training had GPU 0 to itself and B's training GPU 1; no other GPU work ran on a GPU while it
trained. The CPUs, memory and disk were shared by all three copies, so the CPU times below are longer than on a machine of
its own.

**Results in short.**

| | A | B |
|---|---|---|
| GPU-hours of training (limit 24) | 20.77 (20.9 measured on 23-25 October) | 17.68 (17.84 measured on 23-25 October) |
| Shipped weights against the submitted files | not byte-identical (A100 against RTX PRO 6000 rounding); estimated change in S 1e-5 | DIII-D 1e-5, MAST 2e-5 (estimated change in S) |
| Retrained models against the submitted files | DIII-D: label-free bound on the change in S 1.5e-3 | DIII-D: label-free bound 1.9e-3. MAST: S changed by −0.0033 to +0.0007 on 15 synthetic MAST sets |

## Shipped weights

`steps/4_shipped_weights.sh`, `predict.sh` and `steps/5_compare.sh` for A and B (`logs/compare_A_shipped.log`,
`logs/compare_B_shipped.log`). The weights matched their manifests by SHA256, and every file passes the organisers' validator.

| File | Arrays byte-identical | Largest difference ψ / q95 / βN | ψ R² against the submitted file | Estimated change in S | Strict bound |
|---|---|---|---|---|---|
| A: `diii_d_private_test.npz` | 0 of 2,637 | 0.011 / 0.032 / 0.013 | 0.99999994 | 9.9e-6 | 4.7e-4 |
| B: `diii_d_private_test.npz` | 0 of 2,637 | 0.031 / 0.025 / 0.009 | 0.99999992 | 7.5e-6 | 5.7e-4 |
| B: `mast_private_test.npz` | 0 of 3,615 | 0.003 / 0.115 / 0.033 | 0.9999999995 | 2e-5 (6e-5 one standard deviation above) | 1.5e-3 |

SHA256 of the three files: `predictions/SHA256SUMS.txt`.

**Why they differ.** The networks' forward passes run on an A100 here and on an NVIDIA RTX PRO 6000 for the submitted files,
and the two GPUs round differently in the last digits. Every CPU step of `predict.sh` handles each shot on its own, with one
BLAS/OpenMP thread, and writes its results in shot order, so the number of CPU workers cannot change any value
(`logs/timing/notes.md`, 15:20 UTC 31 October). On RTX PRO 6000 GPUs the same commands give byte-identical files.

**Estimated change in S (DIII-D).** `zara/diii-d/checks/noise_bound.py` compares each file with the submitted one, shot by
shot, without labels (`logs/timing/noise_bound_*_shipped_a100.json`). Its strict bound assumes every difference moves the
prediction away from the labels. The estimate uses a test with labels instead (`zara/diii-d/checks/lab_noise.sbatch`,
`score_lab.py`): 101 labelled training shots predicted twice on the RTX PRO 6000, which differed by GPU rounding alone, changed
S by 3.5e-7 (v13) and 1.6e-7 (Ch1-min) against the EFIT labels (`lab_noise.json`). The A100 files differ from the submitted
ones 28 times (A) and 47 times (B) more than those two runs differed from each other, in squared distance weighted as in S
(`noise_lab.json` against the A100 files). Scaling the change in S by the same factor gives the estimate. Scaling linearly with
the squared distance overstates the change when the factor is above 1.

**Estimated change in S (MAST).** No labelled MAST data exists for the test above, so the MAST file was compared with the
submitted one by Zara's label-free bound (`zara/mast/r1/checks/score_bound/`; 31 October, on our cluster, on a copy of the
A100 file, SHA256 `424ee809…` as in `predictions/SHA256SUMS.txt`):

    python zara/mast/r1/checks/score_bound/score_bound.py SUBMITTED/mast_private_test.npz A100/mast_private_test.npz \
        --machine mast --json score_bound_B_mast_shipped_a100.json --workers 16
    python zara/mast/r1/checks/score_bound/score_estimate.py score_bound_B_mast_shipped_a100.json

(result `logs/timing/score_bound_B_mast_shipped_a100.json`). It follows the scorer: ψ over the grid, q95 and βN per frame, the
7 shape scalars derived from each file's ψ with the scorer's own code, and the LCFS distance. Its strict bound, with the
Development-board R² as stand-ins for the error against the labels, is 1.5e-3 (2.6e-3 with no assumption on the labels'
spread). The estimate assumes the A100's rounding differences are unrelated to the errors against the labels: a change d then
adds ||d||² to each term's residual on average, with a spread (one standard deviation) of 2 sqrt((1 - R²) ||d||² / SST / n),
each of the n = 1,205 shots counted as one independent unit. Summed with the weights of S, the average change is 2e-5 (1e-5
from the R² terms, plus 7e-6 for the LCFS term, taken at its strict bound); one standard deviation above the average it is
6e-5, two above 1.0e-4. Those two figures are on the high side: shots, not frames, are the independent units, and the terms'
spreads are added rather than combined in quadrature. Most of the spread comes from the triangularity terms of Consistency
(each LCFS is found by a contour search, and a small change in ψ can move it by centimetres on some frames: largest move
55 mm).

## Training time: two A100 measurements

The GPU-hours declared in `SUBMISSION_A.md` and `SUBMISSION_B.md` were measured on 23-25 October, on dedicated A100
machines (`../a100-dedicated/`; r1 in `zara/mast/r1/checks/a2-ultragpu-1g/`). This run measured them again, a week later, on
a different machine, with this package's `train.sh`. Both are GCP machines with A100-SXM4-80GB GPUs, driver 535.216.01 and
Intel Xeon CPUs at 2.20 GHz.

| | First measurement, 23-25 October | Second measurement, this run |
|---|---|---|
| Machine | `a2-ultragpu-1g`: one A100, 12 vCPUs, 170 GB RAM, one training at a time | `a2-ultragpu-2g`: two A100s, 24 vCPUs shared, each training on a GPU of its own |
| Code | earlier versions of the training scripts (the DIII-D CPU steps ran inside the timed training; v13's input export ran on the GPU) | this package: `prepare.sh` (CPU), then `train.sh` (GPU) |

### Submission A (v13)

| Networks | Count | 23-25 October | This run |
|---|---|---|---|
| ψ networks | 12 | 7.67 h | 7.85 h |
| Sequence heads | 3 | 0.47 h | 0.42 h |
| Functional heads | 8 | 4.27 h | 4.11 h |
| Out-of-fold ψ networks | 5 | 2.52 h | 2.64 h |
| q95 / βN MLPs | 12 | 3.45 h | 3.38 h |
| q95 / βN sequence networks | 6 | 0.75 h | 0.71 h |
| Variant-D networks | 6 | 1.70 h | 1.67 h |
| Input export | | about 2 min (on the GPU then) | on the CPU, in `prepare.sh A` |
| **GPU total** | **52** | **20.9 h** (20.84 h plus the export) | **20.77 h** |
| Peak GPU memory | | 70.5 GB | 69.0 GB allocated, 72.2 GB in `nvidia-smi` |

This run: `train.sh A` 20 h 46 min wall-clock (30 October 21:51 to 31 October 18:38 UTC), with the CPU follow-up of the
out-of-fold networks (2.50 h) in the background. Per step: `train_times/retrain-A/train_times.tsv`, each step marked CPU or
GPU; closing summary in `logs/train_A.log`.

### Submission B (Ch1-min, Chris's 15 MAST runs, r1)

| Part | 23-25 October | This run |
|---|---|---|
| Ch1-min, 12 networks | 3.23 h | 3.18 h |
| Chris's 15 MAST runs | 10.96 h (10.99 h on a second machine at the same time) | 11.07 h |
| r1, 6 runs | 3.65 h (25 October, separate run on the same machine type) | 3.43 h |
| **GPU total** | **17.84 h** | **17.68 h** |
| Peak GPU memory | 59.6 GB | 59.1 GB reserved, 61.0 GB in `nvidia-smi` |

This run: `train.sh B` 17 h 41 min wall-clock (31 October 07:52 to 1 November 01:33 UTC). Ch1-min per step:
`train_times/retrain-B/train_times_ch1min.tsv`; Chris's runs and r1: `logs/train_B.log` and `r1_runs/` (per-epoch times in
each run's `hist.json`).

| Chris's MAST run | 23-25 October | This run |
|---|---|---|
| `scr-F1-z15mix-s0` | 46.5 min | 44.8 min |
| `scr-F2-z15mix-s1` | 44.1 min | 43.9 min |
| `scr-F6-z15mix-s2` | 44.1 min | 44.0 min |
| `scr-F3-only` | 23.4 min | 23.1 min |
| `scr-V7C-v7only` | 21.6 min | 21.2 min |
| `scr-F1-final-g4x5000` | 66.0 min | 65.2 min |
| `scr-mastZ15-v6mix-seed0` | 50.2 min | 49.7 min |
| `scr-mastZ15-v6mix-seed1` | 40.3 min | 40.1 min |
| `scr-mastZ15-v6mix-seed2` | 40.4 min | 40.0 min |
| `scr-mastZ15-v6mix-seed3` | 40.3 min | 40.1 min |
| `scr-G1-g4x5000-noise0.3` | 65.8 min | 66.1 min |
| `scr-F2-final-g4x5000-seed1` | 65.9 min | 70.7 min |
| `scr-G4-g4x5000-clip1` | 65.9 min | 69.1 min |
| `scr-V1-v4only` | 22.5 min | 24.2 min |
| `scr-W4-v4only-ep12-seed1` | 22.5 min | 22.2 min |

`scr-F2-final-g4x5000-seed1`, `scr-G4-g4x5000-clip1` and `scr-V1-v4only` ran while `predict.sh A --rebuilt` used the shared
CPUs (31 October 18:38 to 21:53 UTC), which explains their 5-8 % longer times.

| r1 run | 25 October | This run |
|---|---|---|
| `h_muon_mo` | 4,779 s | 4,525 s |
| `w3_ft` | 1,798 s | 1,662 s |
| `rt_ftw3` | 1,237 s | 1,146 s |
| `rt_ft` | 1,781 s | 1,651 s |
| `rt_mix` | 1,828 s | 1,695 s |
| `rt_nonoise` | 1,732 s | 1,659 s |
| **Total** | **13,155 s = 3.65 h** | **12,338 s = 3.43 h** |

### CPU and wall-clock times of this run

| Step | Wall-clock | Note |
|---|---|---|
| Steps 1-3: environments, official data, MAST data | 1 h 12 min | every SHA256 matches |
| `prepare.sh A` | 5 h 52 min | 12 workers |
| `prepare.sh B` | 7 h 41 min | 6 workers, alongside the shipped-weights check |
| `predict.sh A`, shipped weights | 8 h 22 min | 6 workers, alongside both preprocessing runs and A's training |
| `predict.sh B`, shipped weights | 7 h 23 min | 6 workers, alongside A's training |
| `predict.sh A --rebuilt` | 3 h 15 min | 12 workers |
| `predict.sh B --rebuilt` | DIII-D 1 h 43 min, MAST 3 h 14 min | MAST rerun after the fix below |

On one RTX PRO 6000 node with 16 cores of its own, `predict.sh` takes about 2 h per submission (`SUBMISSION_A.md`,
`SUBMISSION_B.md`, section "Compute").

**Disk and memory.** The run's folders (environments, official data, the three copies and their work folders) peaked at
about 650 GB, on 1 November (`logs/timing/du.tsv`, every 30 min). Memory in use peaked at 52 GB of 334 GB
(`logs/timing/sys.csv`, every 30 s).

## Retrained models

Every network was retrained from random weights (`prepare.sh`, `train.sh`), then used to predict the Final shots
(`predict.sh --rebuilt`) and compared with the submitted files (`steps/5_compare.sh`: `logs/compare_A_rebuilt.log`,
`logs/compare_B_rebuilt.log`). SHA256 of every file: `predictions/SHA256SUMS_rebuilt.txt`. Every file passes the organisers'
validator.

| File | Arrays byte-identical | Largest difference ψ / q95 / βN | ψ R² against the submitted file | Label-free bound on the change in S (0.01 allowed) |
|---|---|---|---|---|
| A: `diii_d_private_test.npz` | 0 of 2,637 | 0.018 / 0.37 / 0.059 | 0.9999988 | 1.5e-3 |
| B: `diii_d_private_test.npz` | 0 of 2,637 | 0.026 / 0.42 / 0.096 | 0.9999981 | 1.9e-3 |
| B: `mast_private_test.npz` | 0 of 3,615 | 0.24 / 6.7 / 1.40 | 0.99980 | see below |

**DIII-D.** `zara/diii-d/checks/noise_bound.py` against the submitted file, all 879 shots, with the Development-board R² as
stand-ins for the error against the labels (`logs/timing/noise_bound_*_rebuilt_a100.json`, `logs/bound_*_rebuilt.log`). The
bound assumes every difference moves the prediction away from the labels, so the change in S is at most this. The RTX retrain
of v13 on 27-28 October gave 1.6e-3 (`SUBMISSION_A.md`, "Checks").

**MAST, member by member.** Each retrained member against the same member predicted from the shipped weights on this machine
(`logs/timing/mast_members_rebuilt_vs_shipped.txt`, no labels):

| Member | ψ R² | Largest ψ difference | q95 R² | βN R² |
|---|---|---|---|---|
| FA | 0.999898 | 0.013 | 0.990 | 0.9995 |
| FC | 0.999660 | 0.021 | 0.978 | 0.9983 |
| V7C | 0.999244 | 0.033 | 0.985 | 0.9990 |
| Z | 0.999013 | 0.048 | 0.954 | 0.9947 |
| Z15 | 0.999934 | 0.014 | 0.998 | 0.9997 |
| Z2 | 0.998752 | 0.041 | 0.951 | 0.9935 |
| Z3 | 0.999598 | 0.028 | 0.977 | 0.9977 |
| Z8 | 0.999825 | 0.010 | 0.996 | 0.9992 |
| r1 | 0.996522 | 1.96 | 0.954 | 0.975 |

r1 moves the most. Its four members each differ from their shipped versions by a similar amount
(`logs/timing/r1_members_rebuilt_vs_shipped.txt`). That is about the size of change of the earlier retrain of r1 from
rebuilt caches: that blend, with only r1 retrained, had ψ R² 0.99996 against the submitted file, which with r1's ψ weight
of 0.1224 corresponds to about 0.997 for r1 itself (`SUBMISSION_B.md`, "Checks"). Two trainings of the same model make different errors frame by frame and score alike, so a
difference between the files does not by itself mean a change in S. For MAST we therefore measured S on labelled synthetic
data, in two ways.

**r1 on Zara's internal validation set** (3,816 held-out FreeGSNKE cases, `SUBMISSION_B.md`, section "r1 (Zara Ercan's MAST
model)"), scored as in `zara/mast/r1/SPEEDUP.md`: each member, and r1 as the plain average of the four (`r1_validation/`, 1
November, one RTX PRO 6000):

| Member | Shipped networks | Retrained on this A100 | Change |
|---|---|---|---|
| `rt_ft` | 0.8356 | 0.8379 | +0.0024 |
| `rt_mix` | 0.8348 | 0.8369 | +0.0021 |
| `rt_ftw3` | 0.8372 | 0.8374 | +0.0002 |
| `rt_nonoise` | 0.8293 | 0.8295 | +0.0002 |
| **r1 (average of the four)** | **0.8404** | **0.8416** | **+0.0012** |

No frame needed r1's fallback. `rt_nonoise` again ended on a poor last epoch on DIII-D held-out frames, as in the earlier
A100 rebuild (`zara/mast/r1/checks/a2-ultragpu-1g/RT_NONOISE_SPIKE_RESULTS.md`); r1 does not use that part of the network
on MAST, and its MAST score is unchanged.

**B's whole MAST blend on 15 labelled synthetic MAST sets** (`synthetic_blend/`, 1 November, our AWS cluster). The held-out
parts of every synthetic MAST set we have, 11,360 cases, were pooled into one folder laid out as the Final MAST split, with
every label column removed and the `source` column set to `MAST` as in the Final files (`build_split.py`). The package's
`predict.sh B` (MAST part) then ran on it twice from this version of the package: with the shipped weights, and with
`--rebuilt` reading the networks retrained in this run (Chris's 15 runs and r1). Both files were scored with the competition's
scoring code against the labels, one score per set (`score.py`; MAST limiter mask from the machine geometry, as for Zara's
validation; a perfect prediction scores 1 on every set, `score_perfect.json`).

| Set | Generator | Cases | S, shipped | S, retrained | Change |
|---|---|---|---|---|---|
| All pooled | | 11,360 | 0.8117 | 0.8113 | −0.0004 |
| `zval` (Zara's internal validation) | FreeGSNKE | 3,816 | 0.8262 | 0.8265 | +0.0004 |
| `g4_20000` | FreeGSNKE | 2,560 | 0.8401 | 0.8403 | +0.0002 |
| `g4_5000` | FreeGSNKE | 616 | 0.8446 | 0.8453 | +0.0007 |
| `g4_v3` | FreeGSNKE | 640 | 0.8206 | 0.8209 | +0.0003 |
| `g4_v4` | FreeGSNKE | 640 | 0.8103 | 0.8109 | +0.0006 |
| `g4_v4b` | FreeGSNKE | 640 | 0.8334 | 0.8339 | +0.0005 |
| `g4_v2` | FreeGSNKE | 64 | 0.8492 | 0.8485 | −0.0007 |
| `g4_v6` | FreeGSNKE | 200 | 0.8091 | 0.8080 | −0.0011 |
| `g4_v7` | FreeGSNKE | 200 | 0.8180 | 0.8162 | −0.0018 |
| `gs1000` | gsolve | 359 | 0.7380 | 0.7357 | −0.0022 |
| `gst1000` | gsolve | 352 | 0.7501 | 0.7481 | −0.0020 |
| `gsc1000` | gsolve | 365 | 0.6925 | 0.6892 | −0.0033 |
| `gsc2z1000` | gsolve | 368 | 0.6976 | 0.6946 | −0.0030 |
| `gsc3_1000` | gsolve | 364 | 0.7110 | 0.7084 | −0.0026 |
| `gsx8` | gsolve | 176 | 0.7579 | 0.7562 | −0.0017 |

Every change is within 0.0034 of zero, against the 0.01 allowed. ψ, 55 % of S, changes by at most 0.0005 on any set; the
largest changes are in q95 and βN on the gsolve sets, the hardest data for these models (up to −0.0032 in S). There the
retrained blend scores lower on every set, and the drop grows as S falls: −0.0017 at S 0.758 (`gsx8`) to −0.0033 at S 0.692
(`gsc1000`). The blend
takes q95 from Chris's models only and βN mostly from FA. No set uses a MAST label. `zval`, `g4_20000`, `g4_5000` and `g4_v3`
share their equilibria (`zval` gives the networks the shipped Thomson profiles, the others the generator's own); the gsolve
sets come from a different solver. Some of these held-out sets helped choose each run's stopping epoch during development;
the retrained runs stop at the same epochs as the shipped ones, so the comparison stays like for like. Scores on
synthetic data do not predict the Final score; the change in S between two versions of the same blend is what they measure.

**Fix found by this run.** `predict.sh B --rebuilt` looked for Chris's retrained MAST checkpoints in `$FEC_WEIGHTS` (the
shipped checkpoints' folder), while `chris/mast/train.sh` writes them under
`chris/mast/multimachine/my_experiments/nn/runs/`, so the MAST part stopped at the first model
(`logs/predict_B_rebuilt.log`). The run reran the MAST part with `FEC_WEIGHTS` pointed at that folder
(`logs/predict_B_rebuilt_mast.log`). This package passes that folder with `--rebuilt`; the synthetic comparison above ran
with the fixed `predict.sh`.

## Files

- `logs/`: every step's log; `machine.txt`; `gpu0.csv`, `gpu1.csv` (GPU use and memory every 30 s); `timing/`: step tables
  (`phase1_steps.tsv`, `phase2_steps.tsv`), `sys.csv` (load, CPU, RAM, disk every 30 s), `du.tsv` (disk per folder), the
  comparisons, the label-free bounds, the projections made during training, and `notes.md` (the run's own notes, in time
  order). `timing/noise_bound_a100.py.txt` is the copy of `zara/diii-d/checks/noise_bound.py` that ran, with only its package
  path changed. The notes use the earlier location of these files (`zara/checks/repro/` is now `zara/diii-d/checks/`), and
  they name Hugging Face datasets the run uploaded its outputs to; those are private and not part of the package. The
  copy of the cluster handoff that the notes mention (`timing/HANDOFF_retrained.md`) is not included: it was instructions
  for our cluster's agent.
- `train_times/`: the DIII-D trainings' `train_times*.tsv`, each step marked CPU or GPU.
- `mast_runs/`, `r1_runs/`: the configuration files and training logs of Chris's 15 MAST runs and r1's 6 runs.
- `predictions/`: SHA256 of every file the run predicted (shipped, retrained, and B's retrained MAST member files).
- `r1_validation/`: Zara's internal validation of r1, shipped and retrained (`a100_r1_eval.sbatch`, `eval_rt_member.sh`,
  logs and `score_raw.json` per member). These ran on our cluster with Zara's evaluation code and data.
- `synthetic_blend/`: the comparison on 15 synthetic MAST sets: `build_split.py`, `prep_chris.sbatch`, `prep_r1.sbatch`,
  `predict.sbatch` (run once with `C=shipped`, once with `C=rebuilt`), `score.py`, their logs, the scores
  (`score_shipped.json`, `score_rebuilt.json`, `score_perfect.json`), the case list per set (`stems/`) and the source file of
  every case (`map.tsv`). The scripts name the folders of our cluster.
