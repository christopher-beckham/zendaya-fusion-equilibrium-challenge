# Timing notes

## 18:05 UTC, shipped-weights prediction of A, part 3 (frame functionals)
mfunc.py runs 13 passes over the 879 Final shots (the mean map, then the 12 prod_auxghc ψ members), one after another,
with --workers $NCPU = 6 here. The first two passes took 1,584 s and 1,581 s, so part 3 should take about 5.7 h
(16:56 to about 22:40 UTC). The RTX machine (16 cores) took about 1 h 22 min for all the CPU parts of predict.sh.
mfunc.py writes frames.npz only at the end, so restarting it with more workers would lose the passes done.

## 18:15 UTC, projected timeline (UTC)
The reference "training wall-clock about 26.5 h" of A is preprocessing (5.5 h) plus GPU training (20.9 h); train.sh A alone
is about 21-21.5 h. Projection: A's training about 21:30 Fri to 18:30-19:00 Sat; shipped SHA256 of A about 23:30 Fri, of B
about 02:00 Sat; B's training about 02:00 to 20:30 Sat; retrained comparisons done about 22:00 (A) and 23:30 (B) Sat.
Deadline 04:59 Sun 1 November.

## 18:32 UTC, r1's first-pass cache of the Final MAST inputs for B's rebuilt prediction
At the request of the repository's agent (relayed by Chris), tools/copy_r1_firstpass.sh waits for steps/4_shipped_weights.sh B to
finish in the shipped copy, then copies work-shipped/r1_data/cache/ch2/g4/caches/mast_private_iv_full_wdiv (1,205 files,
downloaded and SHA256-checked by step 4) into work-retrain-B, checks the count and diff -rq, and logs the time of the copy to
logs/r1_firstpass_copy.log. predict_r1.sh skips its CPU rebuild (about 2 h on 96 cores) when all 1,205 files are there.

## 18:54 UTC, timings pushed to Hugging Face (asked for by Chris)
Chris asked (18:50 UTC) for the timings to be pushed continually to his personal Hugging Face account, so they survive if the VM
is lost. tools/hf_push_timings.py uploads the logs, timing tables, train_times*.tsv, r1 run logs and MAST run text files to
the private dataset beckhamc/fec-a100-final-timings every 15 minutes (first upload 18:54:03 UTC). Text files only; each
upload is skipped if any staged file contains the token. The handoff's "do not upload anything" is set aside for this at
Chris's request.

## 00:24-01:05 UTC, shipped-weights file of A, and the label-free S bound
steps/5_compare.sh A: SHA256 cfa40b06... against the submitted 3af1ebe0...; 0 of 2,637 arrays byte-identical; largest |difference|
psi 0.0106, q95 0.0317, betaN 0.0128; pooled psi R2 0.9999999379. Validator OK. The weights matched (156/156 SHA256).
To size the change in S, zara/checks/repro/noise_bound.py (the analysis behind the RTX 98-shot test) is run on the A100 file
against the submitted file, both split into per-shot files, with PKG pointed at this machine's copy (scratch copy of the script;
no repository change). First started 00:31 with 4 workers at nice 19; stopped 01:03 after a wrong reading (the nice-19 CPU at
the time was mostly train_ch1.sh's own background step r3c_oofqc, 12 workers at nice 19, started by the recipe at 00:29:58,
not the analysis). Restarted 01:04 on one vCPU (taskset -c 23, every threading library at 1 thread, nice 19).
psi_1 (from 00:29:58) runs with GPU 0 about 79% busy (reference psi_1: 92%), alongside r3c_oofqc (12 workers), the shipped
check of B (6 workers) and, from 00:31, the analysis.

## 04:00 UTC, label-free S bound of A's shipped-weights file (noise_bound.py, all 879 shots, against the submitted file)
S_pkg_vs_ref 0.999957 (RTX 98-shot test: 0.999991). Distances (1 - R2 against the submitted file): psi 6.2e-8, q95 1.5e-7,
betaN 2.6e-7, kappa 1.6e-4, tri_bot 2.1e-4, D_lcfs 2.6e-4: 3.6-8.6 times the RTX test's. Strict bound on the change in S
4.7e-4 (Consistency term 4.3e-4), above 1e-4 (the RTX test's strict bound was 1.8e-4, also above). The RTX test's realistic
estimate (labelled-shot rescoring, scaled by the squared-distance ratio) was about 3e-6; scaled the same way here, about 3e-5.
Not separated: A100 GPU rounding against the CPU steps' worker count (mfunc.py and fchain.py ran with 6 workers here).
JSON: noise_bound_A_shipped_a100.json; the script as run: noise_bound_a100.py.txt (PKG path changed only).

## 07:51-10:00 UTC, shipped-weights files of B, and the label-free S bound of B's DIII-D file
steps/5_compare.sh B (07:51): diii_d_private_test.npz SHA256 64a069d4... (submitted 86e6a372...), 0 of 2,637 arrays identical,
largest |difference| psi 0.031, q95 0.025, betaN 0.0092, psi R2 0.99999992; mast_private_test.npz SHA256 424ee809... (submitted
3300302e...), 0 of 3,615 identical, largest psi 0.0027, q95 0.115, betaN 0.033, psi R2 0.9999999995. Both pass the validator.
noise_bound.py on B's DIII-D file (Ch1-min, all 879 shots, one vCPU): noise_bound_B_shipped_a100.json. No MAST equivalent exists.

## 15:20 UTC 31 Oct: requests relayed by Chris for the repository's agent

### 1. Realistic estimate of the change in S, shipped-weights files of A and B (DIII-D)
Method of zara/checks/README.md ("scaled up by that factor the change is at most about 3e-6"): take |dS| measured on the 101
labelled shots predicted twice (lab_noise.json: v13 3.54e-7, Ch1-min 1.59e-7) and multiply it by the ratio of squared
distances (1 - R2 against the reference, per term) between this file and the labelled test's rounding gap (noise_lab.json).
The README's RTX number is reproduced by taking the largest per-term ratio: v13 RTX/lab max 9.2 (li) x 3.54e-7 = 3.3e-6.
Applied to the A100 files (noise_bound_*_shipped_a100.json against noise_lab.json):

| | A (v13) | B (Ch1-min) |
|---|---|---|
| per-term ratio A100/lab, range | 6.2-111 | 3.7-687 |
| per-term method (largest ratio, as the README) | 111 (volume) x 3.54e-7 = 3.9e-5 | 687 (kappa) x 1.59e-7 = 1.1e-4 |
| S-weighted squared distance, A100/lab | 28.0 -> 9.9e-6 | 47.1 -> 7.5e-6 |
| strict bound (noise_bound.py) | 4.7e-4 | 5.7e-4 |

The S-weighted ratio uses 0.55 d_psi + 0.15 mean(d_q95, d_betaN) + 0.10 D_lcfs + 0.20 mean(7 consistency d), computed from
the unrounded distances (on the RTX test it gives v13 6.0 -> 2.1e-6, Ch1-min 3.8 -> 6.1e-7). B's per-term figure is set by
kappa alone: its labelled-test distance for Ch1-min is 3.0e-7, against 8.3e-6 for v13 and 1.7e-5 in the RTX 98-shot test, so
that denominator is unusually small. The earlier "about 3e-5" for A was a rough scaling; the figures above replace it.

### 3. Why the shipped-weights files differ: GPU arithmetic, not the CPU worker count (code reading)
Every CPU step of predict.sh that takes --workers handles each shot on its own and assembles the results in sorted shot order,
with one BLAS/OpenMP thread per worker:
- mfunc.py and fchain.py: ProcessPoolExecutor.map over one file per shot (results in input order); each sets OMP_NUM_THREADS,
  OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to 1 when unset, and settings.sh exports OMP_NUM_THREADS=1. No state is carried
  between shots (fchain's --retry and --anchor act within one frame of one shot).
- preprocess.py, hist_features.py, hist_features_ch1h.py, qcurve_any.py: same pattern (ProcessPoolExecutor.map, threads 1).
- hist_feats.py: Pool.imap_unordered, but results are stored by shot name and concatenated in the sorted stem order.
- preprocess_px.py: --shard/--nshards (default 1 shard) with workers inside the shard.
So the worker count cannot change any value; the byte-for-byte rerun with two worker counts was not needed and was not run.
The differences come from the GPU steps (the networks' forward passes and the q95 / betaN networks, on an A100 instead of the
RTX PRO 6000 the submitted files were made on); every later CPU step carries them forward (map, frame functionals, calibration).
This matches the RTX test, where only the GPU reductions differed between runs on the same GPU type (28 of 98 shots identical);
on a different GPU type every shot differs (0 of 879). Not examined: the MAST side (Chris's predict_model.py, r1's chain).

### 5. GitHub
Nothing is pushed to GitHub (christopher-beckham/fec-final, any branch) from here on; nothing was pushed in this run so far.
Uploads go to the Hugging Face dataset only.

### 4. Timing estimate (15:05 UTC 31 Oct; updated below when it changes)
Basis: GPU-hours projection (tools/project_gpu_hours.py) for training; for prediction, the step times of the shipped-weights
predictions on this machine (6 workers), with the CPU-bound steps halved for 12 workers (NCPU of each line from now on).
Shipped A (6 workers): input caches 9 min, networks 11 min, mean map + frame functionals 5 h 06 min (13 passes), q95/betaN 50 min,
calibration 68 min, submission 4 min, validator 21 min. Shipped B: Ch1-min DIII-D 4 h 00 min (7 passes), Chris's 8 MAST
models 74 min, r1 92 min (first-pass cache downloaded), validators 21 + 16 min.

| | UTC (EDT) |
|---|---|
| train.sh A ends (52 networks, about 20.9 GPU-hours) | about 19:00 Sat (15:00) |
| predict.sh A --rebuilt ends (about 4 h 20 min with 12 workers) | about 23:20 Sat (19:20) |
| steps/5_compare.sh A (rebuilt) ends | about 23:25 Sat (19:25) |
| train.sh B ends (Chris's runs about 21:50 Sat, then r1 3.65 h; about 17.7 GPU-hours) | about 01:30 Sun (21:30 Sat) |
| predict.sh B --rebuilt ends, as the handoff runs it after train.sh B (DIII-D about 2 h 15 min, then MAST 2 h 50 min, validators 37 min) | about 07:10 Sun (03:10) |
| steps/5_compare.sh B (rebuilt) ends | about 07:15 Sun (03:15) |
| Deadline | 04:59 Sun (00:59) |

With nothing extra on either GPU while a training runs, B's rebuilt comparison lands about 2 h 15 min after the deadline.
Running B's DIII-D and MAST parts at the same time on the two GPUs once train.sh B ends brings it to about 04:55, with no margin.
Earlier start needs an exception to that rule: B's DIII-D prediction on GPU 0 from about 19:00 (after train.sh A), and the
predictions of Chris's 8 retrained MAST models on GPU 0 from about 21:50, both at nice 19 while train.sh B trains r1 / MAST
on GPU 1; after train.sh B only r1's prediction, the blend, the MAST validator and the comparison remain (about 1 h 55 min),
so B's rebuilt comparison would land about 03:25 Sun (about 1.5 h before the deadline). Decision for Chris.

### Decision (15:40 UTC 31 Oct, Chris): B's training is not disturbed; B's rebuilt comparison at about 07:15 UTC Sun is fine
Nothing extra runs on either GPU while train.sh B runs. The lines run as the handoff lays them out: line A's predict.sh A
--rebuilt and its comparison on GPU 0 after train.sh A (about 19:00 to 23:25 UTC), alongside train.sh B on GPU 1; line B's
predict.sh B --rebuilt and its comparison on GPU 1 after train.sh B (about 01:30 to 07:15 UTC Sun).

### 15:45 UTC 31 Oct: label-free bound for A's retrained file, prepared ahead of the 00:00 UTC cut-off (asked for by Chris)
noise_bound.py on one vCPU took 2 h 52 min (A's shipped file) and 2 h 08 min (B's); started after the comparison it would land
after the cut-off. About half of it (the reference contours of the submitted file) does not depend on the retrained file, so
a scratch copy of the script (noise_bound_a100_cached.py: same code, plus saving and reloading those references) builds them
now on one vCPU at nice 19 from the submitted A file (SHA256 3af1ebe0..., checked). tools/bound_A_rebuilt.sh waits for
predict.sh A --rebuilt, then scores the retrained file against them with 4 workers at nice 19 on vCPUs 20-23 (train.sh B is
still running), into noise_bound_A_rebuilt_a100.json. Expected about 20-25 min after the prediction, so about 23:50 UTC.

### 16:45 UTC 31 Oct: B's shipped-weights MAST file for Zara's MAST S estimate (asked for by Chris)
out/B_shipped/mast_private_test.npz (SHA256 424ee809...) uploaded to a new private dataset, zenithon-zendaya/fec-a100-final-predictions,
as B_shipped/mast_private_test.npz with SHA256SUMS.txt; the SHA256 Hugging Face records equals the local one. No existing
repository was changed. The same file is also in beckhamc/fec-a100-final-timings under predictions/B_shipped/.

## 18:37 UTC 31 Oct: train.sh A finished, measured GPU-hours
train.sh A, 21:51:21 30 Oct to 18:37:43 31 Oct (20 h 46 min wall-clock), closing summary on NVIDIA A100-SXM4-80GB:
"train_ch1.sh  GPU steps 20.77 h  CPU steps in the background 2.50 h". 52 networks, against 20.83 h on the A100 of 23-25 Oct:

| group | networks | this run | reference (23-25 Oct) |
|---|---|---|---|
| psi | 12 | 7.85 h | 7.67 h |
| sequence heads | 3 | 0.42 h | 0.47 h |
| functional heads | 8 | 4.11 h | 4.27 h |
| out-of-fold psi | 5 | 2.64 h | 2.52 h |
| q95/betaN MLPs | 12 | 3.38 h | 3.45 h |
| q95/betaN sequence | 6 | 0.71 h | 0.75 h |
| variant-D | 6 | 1.67 h | 1.70 h |
| total | 52 | 20.77 h | 20.83 h |

psi and out-of-fold psi ran slower (+2%, +5%) while other CPU work ran (prepare.sh B, the shipped-weights checks and
train_ch1.sh's own background r3c_oofqc until 03:00 UTC); the later groups ran 1-11% faster. Peak GPU memory: 69.0 GB allocated,
70.0 GB reserved ([capmem]); 72,219 MiB in nvidia-smi (gpu0.csv). prepare.sh A: 5 h 52 min (15:59:27 to 21:51:21 30 Oct).
predict.sh A --rebuilt started 18:37:43; expected end about 23:00 UTC, comparison about 23:05, bound about 23:35.

### 20:11 UTC 31 Oct: B's MAST runs alongside A's rebuilt prediction
scr-F2-final-g4x5000-seed1 (19:00:40 to 20:11:23) took 70.7 min against 65.9 min on 23-25 Oct (+7%); it ran entirely
alongside predict.sh A --rebuilt (12 CPU workers, load about 13; the handoff's layout). The run before it,
scr-G1-g4x5000-noise0.3, overlapped it for 6 min only and took 66.1 min (reference 65.8). A's CPU-heavy steps run until about
21:30 UTC, so B's MAST runs up to then may run a few percent slower.

### 20:25 UTC 31 Oct: Chris's retrained MAST runs against the originals (validation metrics logged during training)
mast_rebuild_vs_original_heldout.txt: for each of the 13 runs trained so far, R2psi, R2qb and the synthetic held-out R2psi at
the last epoch trained, against the original run's metrics.jsonl at the same epoch (no MAST labels). Largest |difference| 0.0020.
R2qb is lower in every run, by 0.0003-0.0020; R2psi by 0.00003-0.00015; held-out DIII-D R2psi moves both ways (-0.0007..+0.0006).
These are not S. Through S's weights (0.55 psi, 0.15 q95/betaN) a 0.002 drop in R2qb and 0.0001 in R2psi is about 4e-4 in S
for one model, against the 0.01 allowed. The check the earlier rebuilds used (synthetic-MAST S, ensemble_synth.py --sets
g4_20000, chris/mast/checks/rebuilds/SUMMARY.md) cannot run here: it needs the held-out synthetic cases
(gen4_set20000/mast_g4s20000_heldout), which are on /fsx only; predict_synth.py itself runs on the CPU.

### 20:27 UTC 31 Oct: B's retrained MAST checkpoints for the synthetic-MAST S check on the AWS cluster (asked for by Chris)
The 12 finished runs of Chris's MAST models (last.pt, config.json, metrics.jsonl each) uploaded to the private dataset
beckhamc/fec-a100-b-retrained-mast with HANDOFF.md (for the agent on the AWS cluster: what to run and why), SHA256SUMS.txt,
reference_scores.tsv (original and RTX-rebuild synthetic-MAST S on g4_20000 per run), STATUS.md and the training-metric
comparison; the SHA256 Hugging Face records for each last.pt equals the local one. tools/hf_push_mast_ckpts.py adds the
remaining 3 runs as each finishes (about 22:10 UTC). The synthetic-MAST S itself cannot run here: it needs the held-out
synthetic cases (gen4_set20000/mast_g4s20000_heldout) and the built held-out set (synth_S_scratch/sets/g4_20000_ho*), on /fsx only.

## 20:35 UTC 31 Oct: deadline extended to 4 November 2026, 05:59 UTC (3 November 23:59 CST), organisers' email
Everything now finishes days before it; the run continues unchanged, as the handoff lays it out:
| | UTC |
|---|---|
| A: predict.sh A --rebuilt, steps/5_compare.sh A (rebuilt), label-free bound against the submitted file | about 21:35, 21:40, 22:00 Sat 31 Oct |
| B: Chris's last MAST runs (all 15 checkpoints on beckhamc/fec-a100-b-retrained-mast) | about 22:10 Sat |
| B: train.sh B ends (r1 last), B's GPU-hours | about 01:35 Sun 1 Nov |
| B: predict.sh B --rebuilt and steps/5_compare.sh B (rebuilt) | about 05:30-07:15 Sun |

## 21:54 UTC 31 Oct: line A done; retrained A against the submitted file
predict.sh A --rebuilt 18:37:43 to 21:52:52 (3 h 15 min, 12 workers; validator OK). steps/5_compare.sh A (rebuilt):
SHA256 5d872037... (submitted 3af1ebe0...); 0 of 2,637 arrays byte-identical; largest |difference| psi 0.0185, q95 0.371,
betaN 0.0586; pooled psi R2 against the submitted file 0.9999988208. The label-free bound against 0.01 (noise_bound.py on the
retrained file, references of the submitted file built at 15:45-16:46) started 21:55: tools/bound_A_rebuilt.sh's wait for the
references matched its own parent shell (whose command line held the same text) and never ended, so it was stopped and its
scoring steps run as written (tools/bound_A_rebuilt_now.sh, 4 workers at nice 19 on vCPUs 20-23).

### 22:13 UTC 31 Oct: label-free bound for A's retrained file against the submitted file (0.01 allowed)
noise_bound.py on all 879 shots (noise_bound_A_rebuilt_a100.json): S of the retrained file scored with the submitted file as
the reference 0.999726; distances (1 - R2) psi 1.2e-6, q95 1.8e-5, betaN 1.4e-5, consistency 1.3e-4 to 1.2e-3, D_lcfs 1.6e-3.
Strict bound on the change in S against the real labels (triangle inequality, label error taken at twice the Development board's):
1.54e-3 (psi 1.4e-5, q95/betaN 6.7e-5, LCFS 1.6e-4, consistency 1.3e-3), 6.5 times inside the 0.01 allowed.

## 01:32 UTC 1 Nov: train.sh B finished, measured GPU-hours
train.sh B, 07:51:33 31 Oct to 01:32:33 1 Nov, closing summary on NVIDIA A100-SXM4-80GB:
"train_ch1min.sh GPU steps 3.18 h; chris/mast/train.sh (15 runs) GPU training 11.07 h (wall-clock);
r1 train_r1.sh full (6 runs) GPU training 3.43 h (wall-clock)". Total 17.68 GPU-hours (declared 17.84: 3.23 + 10.96 + 3.65).
r1 runs (exit 0 wall): h_muon_mo 4,525 s, w3_ft 1,662 s, rt_ftw3 1,146 s, rt_ft 1,651 s, rt_mix 1,695 s, rt_nonoise 1,659 s
(5-8% under the a2-ultragpu-1g check's times). Chris's runs: within 2% of the 23-25 Oct times except the three that ran
alongside predict.sh A --rebuilt (+5 to +8%). Peak GPU memory: 55.7 GB allocated, 59.1 GB reserved ([capmem], Ch1-min);
r1 46,781 MiB (r1_runs/*.gpu.csv); 61,019 MiB in nvidia-smi over GPU 1's whole run (gpu1.csv).
prepare.sh B: 7 h 41 min (6 workers). predict.sh B --rebuilt started 01:32:33; expected about 06:00 UTC, comparison after.

## 03:16 UTC 1 Nov: predict.sh B --rebuilt FAILED at the first MAST model (package bug; line B stopped)
The DIII-D part finished (out/B_rebuilt/diii_d_private_test.npz, 03:16:08, 1 h 44 min). Then:
  [03:16:08] MAST model FA
  FileNotFoundError: [Errno 2] No such file or directory: '/root/a100-final/work-retrain-B/mast_weights/scr-F1-z15mix-s0-rebuild/last.pt'
  == 2026-11-01T03:16:12Z FAILED rc=1 env FEQ_CLEAN=1 bash predict.sh B /root/a100-final/work-retrain-B/out/B_rebuilt --rebuilt
Cause: with --rebuilt, chris/mast/predict_model.py loads WEIGHTS/RUN-rebuild/last.pt, with WEIGHTS = --weights or $FEC_WEIGHTS.
settings.sh sets FEC_WEIGHTS=$WORK/mast_weights (the shipped checkpoints' folder, which only steps/4_shipped_weights.sh fills),
while retrain.py (via chris/mast/train.sh) writes RUN-rebuild/ under chris/mast/multimachine/my_experiments/nn/runs/. predict.sh
does not pass --weights for --rebuilt, so predict.sh B --rebuilt cannot find any retrained MAST checkpoint, on any machine.
Suggested fix (for the package): in predict.sh predict_B, with --rebuilt, call predict_model.py with
--weights $HERE/chris/mast/multimachine/my_experiments/nn/runs. Nothing changed here; waiting for Chris.

### 03:51 UTC 1 Nov: label-free bound for B's retrained DIII-D file (Ch1-min) against the submitted file (0.01 allowed)
noise_bound.py on all 879 shots, 03:17-03:51 UTC, 4 workers at nice 19 (noise_bound_B_rebuilt_a100.json); submitted B DIII-D
file SHA256 86e6a372... checked. Retrained file 03:16:08 (predict.sh B --rebuilt, DIII-D part).
S of the retrained file scored with the submitted file as the reference 0.999623; largest |difference| psi 0.0265, q95 0.421, betaN 0.0964. Strict bound on the change in S 1.90e-03 (psi 1.8e-05, q95/betaN 1.0e-04, LCFS 2.1e-04, consistency 1.6e-03), 5.3 times inside 0.01.

## B's MAST part rerun (2026-11-01T10:27:37Z)
Chris approved option 1: `tools/lineB_mast.sh` reruns `predict.sh B --rebuilt` with PARTS=mast and
FEC_WEIGHTS=chris/mast/multimachine/my_experiments/nn/runs (where retrain.py wrote the checkpoints), then `5_compare.sh B`.
No code changed. The DIII-D file from the 03:16 UTC attempt is kept. Expected about 2.5 h.

## B's MAST part: result (13:43 UTC 1 Nov)
predict.sh B --rebuilt (PARTS=mast) ran 10:27:42 to 13:41:54 UTC; 5_compare.sh B done 13:43:44. LINE B DONE.
- out/B_rebuilt/mast_private_test.npz SHA256 d89a4b2c...; submitted 3300302e...
- Against the submitted file: psi R2 0.9998021, largest |diff| psi 0.238, q95 6.70, betaN 1.40 (0 of 3615 arrays identical).
- For scale, chris/mast/checks/step1_models_and_blends.result.txt (RTX, Chris's models only, no r1): retrained blends psi R2
  0.99997-0.999997, largest psi diff 0.003-0.008, q95 6.2-9.3, betaN 0.12.
- Per member, retrained against the shipped weights on this A100 (mast_members_rebuilt_vs_shipped.txt): Chris's 8 models psi R2
  0.9988-0.99993, largest psi diff 0.010-0.048; r1 psi R2 0.99652, largest psi diff 1.96, betaN R2 0.975. r1 is the outlier.
- r1's rt_nonoise ended on a bad last epoch again (work-retrain-B/r1_runs/rt_nonoise/hist.json): DIII-D val R2psi 0.9972 at
  epoch 8, -0.567 at epoch 9 (r2_bN_mag -82.9, L_ampere 38,059; other numbers normal; n_skipped 0). The same signature as the
  earlier A100 rebuild (zara/mast/r1/checks/a2-ultragpu-1g/RT_NONOISE_SPIKE.md: -3.12 at epoch 9). r1 uses the last epoch.
- On the MAST Final inputs (r1_members_rebuilt_vs_shipped.txt; per-frame RMS psi diff / psi std, retrained vs shipped):
  rt_ft/rt_mix/rt_ftw3 median 0.026, frames >0.1: about 3,000 of 68,373; rt_nonoise median 0.034, frames >0.1: 5,079. No
  non-finite frames. All four members have a tail of large frames (max 1.3-6.6), so the spike is not the only source.
- Correction (14:15 UTC): zara/mast/r1/checks/a2-ultragpu-1g/RT_NONOISE_SPIKE_RESULTS.md showed the rt_nonoise spike is one
  DIII-D frame in the edge-field correction, which r1 switches off on MAST (--no-harm); ep9 vs ep7 on the MAST public test
  differed less than two healthy epochs. So the spike is not the likely cause of r1's MAST difference here. All four retrained
  members differ from the shipped ones by a similar amount (median 0.026-0.034 per frame): ordinary retrain variation of r1,
  which the RTX full-chain retrain kept within 0.0018 of Zara's internal-validation S.

## 14:30 UTC 1 Nov: everything uploaded before shutdown (asked for by Chris)
- beckhamc/fec-a100-final-retrained-weights (new, private): every retrained network, 230 files, 23.4 GiB (A's DIII-D files, 103;
  B's DIII-D files, 30; Chris's 15 MAST runs; r1's 6 runs with every checkpoint), SHA256SUMS.txt and HANDOFF.md for the AWS agent
  (copy: logs/timing/HANDOFF_retrained.md). 230/230 SHA256 match on Hugging Face (14:25 UTC).
- beckhamc/fec-a100-final-timings predictions/: the retrained output files (A_rebuilt, B_rebuilt) and B's 9 MAST member files,
  with SHA256SUMS_rebuilt.txt; 12/12 SHA256 match (14:30 UTC).
- RESULTS.md (the handoff's report) at the top of beckhamc/fec-a100-final-timings and in logs/timing/. Not committed to GitHub.
stopped 2026-11-01T14:30:56Z: samplers, projection loop, 15-min upload loop
- 14:41 UTC 1 Nov: B's shipped-weights MAST member files (9) uploaded to predictions/B_shipped/members/ with SHA256SUMS_shipped_members.txt; 9/9 SHA256 match.
