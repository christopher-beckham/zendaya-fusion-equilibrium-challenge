# HANDOFF: DIII-D-taught conditional gsolve recipe (tune2)

Work dir `/fsx/chris.beckham/gsolve_data/tune2/`. Python `/fsx/chris.beckham/gsolve_data/tune/venv/bin/python`
(read-only use of the old venv). Always `CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=/fsx/chris.beckham/gsolve_copy`.
CPU: <= 24 cores until 05:00 UTC 16 Oct, then <= 40. Do not touch `../tune/`.

## Plan (task brief)
A. DIII-D teacher: z (machine-independent input features, `zfeat.py`) -> theta (li, K = <p>/<p_e>) fitted on DIII-D
   training labels, CV by shot. B. Conditional generator on MAST, offsets delta and spread s tuned by the frozen MAST
   critic protocol (copied from ../tune). C. DIII-D recipe check: generator on DIII-D held-out shots vs EFIT (D_label)
   for defaults / x8 / conditional. D. 1,000-shot set NAME=cond1000, cache gsolve_cond1000(_limiter), practice set
   g4_gsc1000, then COND_READY. E. REPORT.md.

## Log
- 03:51 start. 04:10 `run_tables.sh` builds `d3d/train.parquet` (700 train-split shots x16 frames) and
  `d3d/val.parquet` + `d3d/val_psi.npy` (250 val-split shots x8 frames, with EFIT flux maps) from our split
  `/fsx/chris.beckham/fusion_cache/shots.parquet`.

- 03:57 tables done: train 10,951 frames / 699 shots; val 1,932 frames / 249 shots (+ flux maps).
- 04:00 `fit_prior.py` -> `cond_model.pkl` (+ `.json`, `fit_prior.json`). Transferable features only (rr, te0, tav
  dropped: MAST is outside the DIII-D range on 24-29% of frames). li: GBM CV R2 0.75 (held-out val 0.78), resid sd
  0.13; ridge 0.57. log K: GBM CV 0.45 (val 0.50), resid sd 0.22; ridge 0.23. Applied to MAST inputs
  (`mast_z.py` -> `mast/real_z.parquet`, 65,806 frames, 1,206 shots): li* median 1.15 (gbm) / 1.21 (ridge),
  K* median 1.95.
- 04:05 `fit_emu.py mast`: gsolve li emulator on MAST (142k equilibria from ../tune/runs, joined to real z): CV R2
  0.92, resid sd 0.08. 3-shot test of `gen_param2.py --knobs '{"cond":1}'`: achieved li vs target li* corr 0.95,
  median offset 0.000.
- 04:02 DIII-D recipe check started (`run_d3dchk.sh def '{}'` then `x8 knobs_x8.json`; logs/chk_def_x8.log).
  Smoke test (75 frames): converged ~73%, kept 61/75 frames; absolute flux has a per-frame offset drifting along
  the shot (E-coil flux; same for every recipe) -> d3d_score.py also reports offset-corrected R2psi and S_off.

- 04:29 DIII-D recipe check (step C), common 1,287 frames (`compare_d3d.py def x8 cond`, d3dchk/compare_def_x8_cond.csv):
  D_label (1 - mean R2 of 8 labels) def 0.327, x8 0.350, cond 0.299; li R2 0.42/0.35/0.53; S_off 0.898/0.893/0.906;
  K median 2.51/2.45/1.93 vs EFIT 1.99. DIII-D emulator (`fit_emu.py d3d`, from def+x8 runs) CV R2 0.82.
- 04:39 `x8chk` (x8 knobs through gen_param2/tune2) reproduces x8 seed 0 exactly (loss 0.2654): the copied protocol
  is unchanged. c0 (cond base seed 0): protocol 0.293 (x8 0.265), fixed-frame 0.250 (x8 0.246); top giveaway
  rel_Rpk_ne 0.145 (x8 0.076).
- 04:59 final2.sh dry run (30 shots; names cdry30/gsolve_cdry30/g4_gscdry30) passed end to end: build exit 0,
  1,560 frames (97 held out), truth/perfect S = 1.000; cleaned with `dry_cleanup.sh` (cache, scratch set, truth pkl,
  extra_sets entry, data folder). A 10-shot dry run had no held-out case (build exit 1 at the scorer), hence 30.
  Generation speed: 30 shots in 15 min at 3x1 threads.
- 05:00 critic so far (protocol / fixed-frame loss): x8 0.268 / 0.257 (3 seeds); cond base c0 0.293, 0.281, 0.279
  (mean 0.284) / 0.250, 0.245, 0.279 (mean 0.258) -> equal to x8 on the fixed-frame scale; c_lim (d_li -0.15)
  0.279 / 0.294.
- 05:40 critic single runs (protocol / fixed): c_lip 0.271/0.253, c_lkm 0.247/0.251 (r1 0.270/0.237), c_lkp
  0.283/0.256, c_s05 0.235/0.270, c_s15 0.256/0.264, c_ridge 0.262/0.270. DIII-D (common 1,214 frames, paired
  boot diff to cond): cond_r1 -0.007+-0.010 (seed noise), cond_th0 (sel_th 0) -0.009+-0.008, cond_s05
  -0.031+-0.008 (passes, better), cond_ridge +0.055+-0.010 (fails). Queued repeats c_lkm_r2, c_lip_r1/r2,
  c_s05_r1/r2, c_th0; DIII-D checks cond_lkm, cond_lip; second runner starts after them.
  Synthetic MAST label medians (tuning panel): li 0.995 base / 1.02 x8 / 1.12 cond; K 2.60/2.59/1.92; betaN 1.20/1.22/0.90;
  kappa 1.72/1.71/1.68; R_ax 0.931/0.930/0.921 (`labels_runs.py`).
- 06:06 **DECISION: final recipe = conditional base** (`knobs_cond_base.json` = x8 knobs + cond=1, gbm teacher,
  delta 0, s 1). Fixed-frame 3-seed means: cond 0.258, x8 0.257, c_lkm 0.252, c_s05 0.255 (0.270/0.231/0.263),
  c_lip 0.266; singles c_th0 0.279, c_lkp 0.256, c_s15 0.264, c_ridge 0.270, c_lim 0.294. DIII-D gate (paired boot
  diff to cond, 1,208 common frames): cond_lkm +0.030+-0.008 FAIL, cond_lip +0.066+-0.012 FAIL, cond_ridge FAIL,
  cond_s05 -0.030 pass, cond_th0 -0.008 pass. No passing variant beats cond by more than noise on the critic
  (c_s05 by 0.003) -> rule (3): keep cond. (c_s05 is better on per-frame D_label partly by construction: a
  narrower spread is closer to a point prediction; s = 1 matches the DIII-D residual spread.)
  Search stopped (`queue.txt.stop`). final2.sh started: `GEN_THREADS=13 CONV_WORKERS=36 BUILD_WORKERS=24
  ./final2.sh knobs_cond_base.json` -> final2.log, /fsx/chris.beckham/gsolve_data/cond1000/.
- 04:30 MAST critic queue started (`runner.sh queue.txt 5`, logs/runner_a.log). Interim note sent to main.

## Decisions
- **Decision rule (coordinator, user approved, 04:40):** choose the final recipe by the DIII-D label check FIRST, then
  the MAST critic. (1) A candidate must not be worse than the conditional base ("cond" = x8 knobs + cond=1, delta 0,
  s 1) on D_label beyond noise (shot bootstrap of the paired difference, `compare_d3d.py`, plus a seed repeat
  cond_r1). (2) Among those that pass, best on the MAST critic, fixed-frame scale, 3 repeats. (3) If no MAST-critic
  variant beats cond by more than noise, keep cond. Every MAST-critic winner gets a DIII-D check before adoption
  (delta/s apply on DIII-D too); also checked: cond with sel_th 0 (sel_th hurt li/kappa/R_axis on DIII-D).
- D_label bootstrap (200 shot resamples): sd ~0.016 per recipe; paired difference to cond: def +0.028 +- 0.011,
  x8 +0.052 +- 0.012 (cond better).
- DIII-D Thomson: core vertical chord (R 1.94, Z 0.03-0.83) + tangential midplane channels (R 1.67-2.06), positions
  from thomson_chord_R/Z by channel name. Each channel is put on the outboard midplane radius of its EFIT flux
  surface: x = (R_mid - R_axis)/(R_out - R_axis), the analogue of MAST's x from the midplane Thomson profile.
- K target = <p>/<p_e>: <p> from EFIT betaN (B0 at RCENTR 1.6955 m from the bcoil current, a = LCFS half-width),
  <p_e> = volume average (cells inside the EFIT LCFS, weight R) of the Thomson p_e(psi_N) table.

## File map
| File | Purpose |
|---|---|
| `zfeat.py` | shared feature definitions z |
| `d3d_table.py` | DIII-D frame table (z, theta, coils, Thomson channels, EFIT labels, flux maps) -> d3d/ |
| `fit_prior.py` | step A: theta = (li, log K) ~ f(z), CV by shot -> cond_model.pkl/json, fit_prior.json |
| `fit_emu.py` | gsolve li emulator per machine (inverts li* -> an) -> adds `emu` to cond_model.pkl |
| `cond_prior.py` | the conditional draw used by both generators |
| `gen_param2.py` | MAST generator = ../tune/gen_param.py + z features + knobs cond, cond_model, d_li, d_lk, c_s, c_model (cond=0 reproduces gen_param.py) |
| `mast_z.py` | z for all real MAST frames (input columns only) -> mast/real_z.parquet |
| `gen_d3d.py` | DIII-D recipe check generator (oracle EFIT midplane edges stand in for MAST's Thomson edges) |
| `d3d_score.py` | D_label: per-label R2/bias/spread vs EFIT, R2psi (raw and offset-corrected), D_LCFS, S -> d3dchk/<tag>_score.json |
| `run_d3dchk.sh TAG KNOBS [COND]` | 4 shards x 5 threads + scoring |
| `tune.py`, `tune_lib.py`, `critic.py`, `fixed_frames.py`, `real/`, `panels.json`, `critic_params.json`, `noise.json` | frozen critic protocol, copied from ../tune; tune.py/tune_lib.py call gen_param2.py |
| `runner.sh queue.txt [THREADS]` | critic evaluation queue (knob overrides on knobs_x8.json), runs/<tag>/, log.csv, fixed_frames.csv |
| `runs/base_s*`, `runs/x8_m0selth3*` | copied reference runs (baseline, x8 incumbent) |
