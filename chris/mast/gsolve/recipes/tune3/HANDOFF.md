# HANDOFF: tune3 (DIII-D flux-map-supervised conditional generator)

Brief: `/fsx/chris.beckham/gsolve_data/TUNE3_BRIEF.md`. Background: `../tune2/HANDOFF.md`.
Work dir `/fsx/chris.beckham/gsolve_data/tune3/`. Python `./venv/bin/python` (-> ../tune/venv), `CUDA_VISIBLE_DEVICES=`,
`PYTHONPATH=/fsx/chris.beckham/gsolve_copy`. CPU only, <= 40 cores. Code copied from tune2 (paths sed'ed to tune3);
d3d/, d3dchk/, runs/ hold symlinks to the tune2 files (read-only), new outputs are written next to them.
`cond_model.pkl` here is an unchanged copy of tune2's (incumbent teacher); refits go to other file names.

## Log (UTC, 17 Oct)
- 00:30 setup done.
- 00:20 (clock) height fit (`height_fit.py` -> height_fit.json): EFIT Z_axis = s * z_bal on d3d/train, s = 0.764
  (R2 0.640; val 0.774, R2 0.693). Generated scorer Z_axis vs zc (cond replay): slope 0.916 -> zc factor s/0.92 = 0.83.
  Candidates: h1 zc_lo = zc_hi = 0.83 (zc_sd 0.015); h2 zc_lo 0.66, zc_hi 1.0. DIII-D replays cond_h1/cond_h2 started
  (oracle edges, 4x4 threads each; logs/d3d_h1.log, d3d_h2.log).
- 00:22 `d3d_table3.py` (= d3d_table + Thomson edges: outermost core vertical-chord channel with Te >= 5% of the
  smoothed peak -> its EFIT psi_N surface -> midplane radii R_in_th/R_out_th, pn_edge; z features recomputed with these
  edges as zth_*; te_q_th/ne_q_th) running via run_tables3.sh -> d3d/train3.parquet (+train3_psi.npy), d3d/val3.parquet.
- 00:25 `d3d_critic.py` written (flux-geometry critic; features cached in d3dchk/efit_geom.parquet, <tag>_geom.parquet;
  results appended to geom_critic.csv). MAST critic queue queue.txt (h1/h2 x 3 seeds) to start when CPUs free.
- 00:33 E-coil calibration (`run_cal.sh`, `ecoil_cal.py`): cond recipe on 479 TRAIN frames (120 shots, d3d/train_cal.parquet),
  offset = mean(EFIT) - mean(-gsolve) = k * c_ECOILA * sign + b, k = -0.00909 Wb/kA, b = -0.0042, CV R2 0.996,
  resid sd 0.014 -> ecoil_cal.json. d3d_score.py now writes r2psi_cal / S_cal (= S_D) and ssr_cal per frame;
  compare_d3d.py reports S_D, paired shot-bootstrap S_D / D_label / Z_axis-R2 differences to REF (default cond).
- 00:34 Thomson-edge tables: pn_edge median 0.975 (5-95% 0.91-1.07), valid 99.7%; R_out_th - R_out median -0.9 cm.
  `make_th.py` -> d3d/{train,val}_th.parquet (edges, z, te_q/ne_q replaced; th_ok flag). Height slope with Thomson
  edges (height_fit3.json) 0.766 (same). Teacher refit `fit_prior3.py _th cond_model_th`: li gbm CV 0.708 (val 0.719),
  logK CV 0.445 (val 0.458) (oracle-edge teacher: li 0.75/0.78). Emulators copied from cond_model.pkl.
- 00:35 geometry critic noise: cond 0.284 (seed sd 0.003), cond_r1 0.275 -> generation-seed noise ~0.009.
  cond_h1 0.248 (better by ~0.03). Oracle-edge DIII-D: cond_h1 Z_axis R2 0.687, cond_h2 0.676 (cond 0.452).
- 00:36 honest replays queued (run_th.sh, FRAMES=d3d/val_th.parquet): thc (cond), thc_h1, tht (th teacher), tht_h1.
- 00:37 Ip-history features: zfeat.HFEATS (h01/h025/h05 = Ip(t - k tau)/Ip(t), tsince, ew), zfeat.FEATS_ALL; gen_param2
  frame_z(..., traces) fills them on MAST (read_shot returns ip_trace); cond_prior indexes FEATS_ALL and feeds the
  emulator the base 14 features (EMU_FEATS) -> bit-identical output for the old teacher (checked on one shot,
  test3/). DIII-D: d3d_hist.py -> d3d/{train,val}_hist.parquet, d3d/{train,val}_thh.parquet. MAST range check
  (mast_h.py, 334 shots): MAST outside DIII-D 1-99% on 1-5% of frames per feature; MAST younger in tau units
  (tsince median 0.18 vs 0.31). Teacher `cond_model_thh` (th edges + history): li CV 0.751 (val 0.764), logK CV 0.484.
- 00:38 MAST critic h1 seed 0: protocol 0.229, fixed-frame 0.256 (cond 3-seed fixed mean 0.258).
- 00:50 BUG found+fixed: d3d_score default glob `<tag>_*.npz` also matched `cond_h1_*.npz` etc. for tag cond -> now
  `<tag>_[0-9].npz`. All tune3 scores made after the fix (cond, cond_r1, cond_h1, cond_h2 rescored with E-coil cal).
- 00:57 HEIGHT RULE PASSES (oracle-edge DIII-D, 1,370 common frames, paired shot bootstrap vs cond):
  h1: S_D 0.9117 vs 0.9033 (+0.0083 +- 0.0017, z 4.9), D_label 0.269 vs 0.302 (-0.033 +- 0.006), Z_axis R2 0.686
  (cond 0.456), li R2 0.515 (0.523), geometry critic gain 0.248 (cond 0.284 / cond_r1 0.275).
  h2: S_D +0.0076 +- 0.0016, D_label -0.032, Z_axis R2 0.678, geom 0.253. cond_r1 (seed repeat) S_D +0.0005 +- 0.0024.
  MAST critic fixed-frame: h1 0.256, 0.268 (+ r2 pending); h2 0.277, 0.299 -> h2 fails the guard (cond 0.258 + 0.02).
  -> EARLY SET started 00:57 with h1 (knobs_early_h1.json = knobs_cond_base + zc_lo = zc_hi = 0.83, zc_sd 0.015,
  old teacher tune3/cond_model.pkl == tune2's): `NAME=cond2z1000 TAG=gsolve_cond2z1000 SET=g4_gsc2z1000
  READY=tune3/EARLY_READY SKIP_EXTRA=1 GEN_THREADS=4 CONV_WORKERS=14 BUILD_WORKERS=14 ./final2.sh knobs_early_h1.json`
  -> early.log, /fsx/chris.beckham/gsolve_data/cond2z1000/. If it dies: rerun the same command (it skips finished parts).
  Critic runner restarted with 3 threads/shard (logs/runner_b.log) after h1_r2; h2_r2 dropped.
- 01:05 HONEST judge (Thomson edges, FRAMES=d3d/val_th.parquet, E-coil calibrated). Incumbent thc: S_D 0.877
  (oracle-edge 0.903; S_off 0.879 -> calibrated S_D within 0.002 of S_off), D_label 0.405, li R2 0.414, Z_axis R2
  0.421, kappa R2 -0.20 (oracle +0.32), geom gain 0.294. thc_h1: S_D +0.0081 +- 0.0015, D_label -0.037 +- 0.009,
  Z_axis R2 0.631, geom 0.250 -> height rule confirmed on the honest judge.
  tht (Thomson-edge teacher, zc default) vs thc: S_D -0.0002 +- 0.0017, D_label -0.006 +- 0.009, geom 0.288 (noise).
  Edge offset (edge_offset.py): honest generated LCFS Rmax - EFIT median -0.49 cm (oracle +0.24), Rmin +0.22 cm,
  a -0.25 cm (EFIT a 58.8 cm): small (< 1% of a) -> edge knobs low priority.
- 01:07 MAST critic h1 fixed-frame 3 seeds 0.256 / 0.268 / 0.274, mean 0.266 (cond 0.258): passes the guard.
  Queued: run_th2.sh (thh_h1), run_th3.sh (thh_h1_s075, thh_h1_s05; FRAMES=d3d/val_thh.parquet).
- 01:45 Teacher variants (honest, paired vs thc_h1, 1,370 frames): tht_h1 S_D +0.0012 +- 0.0017, D_label -0.009 +- 0.009,
  geom 0.247 (thc_h1 0.250); thh_h1 (Thomson edges + Ip history) S_D +0.0037 +- 0.0021 (z 1.8), D_label -0.023 +- 0.009,
  li R2 0.456 (thc_h1 0.404), geom 0.2495. MAST fixed-frame: t1h1 0.259, 0.282; thh1 0.291, 0.278 (mean 0.284 vs cond
  0.258: above the +0.02 guard on 2 seeds). -> no teacher change so far (z < 2 and MAST guard borderline); third seeds
  pending. NOTE: thh_h1 replay shards ran 2x slower (unexplained; same frames).
- 01:47 c_s / edge / d_li variants moved to the incumbent teacher: run_th5.sh (thc_h1_{s075,s05,esd,so3,lip,lim},
  FRAMES=d3d/val_th.parquet; esd = edge_sd 0.005, so3 = sel_out 0.03, lip/lim = d_li +-0.05). Killed the thh-based
  run_th3/run_th4 chains. MAST queue: ts075 x3, tesd, ts05 now on cond_model.pkl.
- 01:58 COORDINATOR: pilot loop (pilots/queue/<name>.json -> 150-shot pilot + fine-tune), keep total CPU < ~44 (pilots
  use 6-12). Queued pilots: thh_h1 (history teacher), h1_s075. User asks for critic-ALGORITHM work: (a) critic-guided
  variant selection, (b) geometry-critic giveaways -> next target. CPU restructured: run_th5/run_th6 killed; runner
  restarted at 2 threads/shard (logs/runner_c.log); run_sel.sh (3 threads/shard): 8-variant replays seltr (200 TRAIN
  shots x4 frames, d3d/train_sel.parquet) and selva (val_th), then thc_h1_{s05,lip,lim,esd}. gen_d3d now also saves
  err/dout/din per kept variant. sel_critic.py: feats / fit / apply LAM (re-rank by err - LAM*logit p(real)).
- 02:00 thc_h1_s075 (c_s 0.75, honest): S_D +0.0070 +- 0.0015 (z 4.6), D_label -0.033 +- 0.007, li R2 0.495 (0.413),
  Z_axis R2 0.615, geom 0.251 (thc_h1 0.250) -> DIII-D clear win; MAST critic ts075 x3 pending.
- 02:00 (b) geometry-critic giveaways (geom_imp.py thc_h1, permutation, nats): ax_R 0.54, s06 0.45, k03 0.11, kappa 0.05,
  ax_Zbal 0.05. The joint shift profile: s06/ax_R EFIT median 0.58 vs gen 0.53, frame-wise corr -0.01 (gsolve's axis
  shift relative to the psi_N 0.6 surface does not follow EFIT; core current/pressure profile shape). Addressed by (a):
  the selection critic sees these features.
- 02:08 MAST range check of geometry features (mast_geom_check.py, 667 generated cond1000 frames, limiter mask):
  inside DIII-D EFIT 1-99% for all but squareness (23% outside) and q50 (22%); shift features at the high end (ax_R
  median 0.14 vs DIII-D 0.07). The selection critic drops sq and q50.
- 02:12 selection critic fitted on seltr (TRAIN replay, 8 variants, 620 frames with kept variants, 3,421 variants):
  CV gain 0.213 (600 iters) -> sel_critic.pkl. gen_param2 knobs sel_crit (LAM, default 0 = off) / sel_crit_model:
  err -= LAM * logit p(real) over accepted variants (geometry on the MAST limiter-mask grid, d3d_critic.set_grid);
  smoke-tested on one MAST shot (p(real) median 0.36). sel_crit 0 path unchanged.
- 02:18 runner waiter bug (pgrep matched itself) cost ~15 min; runner restarted at 2 threads/shard (logs/runner_c.log).
  thh1 MAST fixed 3 seeds 0.291/0.278/0.293 (mean 0.287 vs cond 0.258) -> history teacher FAILS the MAST guard.
- 02:50 PERF BUG fixed: per-row `z["psi"][j]` on compressed npz re-decompressed the whole array each access (sel_critic,
  d3d_score, d3d_critic, edge_offset) -> now dict(np.load(...)); scoring takes seconds.
- 02:55 (a) CRITIC-GUIDED SELECTION, DIII-D honest judge (val_th, h1 recipe, same solved variants, paired vs LAM 0):
  8 variants: LAM 0.02/0.05/0.1/pure-critic: S_D +0.0057/+0.0062/+0.0074 (+-0.0024, z 3.1)/+0.0053; D_label -0.034/-0.039/
  -0.047/-0.041; kappa R2 -0.10 -> +0.07/+0.10/+0.13/+0.12; li R2 0.39 -> 0.45/0.47/0.49/0.51; geometry judge (CV on VAL
  shots, selection critic trained on TRAIN shots) 0.258 -> 0.219/0.203/0.198/0.201.
  4 variants (as the MAST generator; VMAX=4, LAM 0 reproduces thc_h1 exactly): LAM 0.1: S_D +0.0039 +- 0.0017 (z 2.4),
  D_label -0.039 +- 0.008, kappa R2 -0.22 -> -0.02, li R2 0.40 -> 0.48, Z_axis R2 0.637, geom 0.250 -> 0.206.
  LAM 0.2 same as 0.1. -> candidate sel_crit 0.1. MAST critic queue reordered: ts075_r1, tsel x3 (sel_crit 0.1),
  ts075_r2, tss x3 (sel 0.1 + c_s 0.75). run_sel2.sh: 8-variant c_s 0.75 replay (selv75) for the combined DIII-D test.
- 03:15 COMBINED DIII-D (honest, paired vs thc_h1 = selva_l0p0_v4, 1,392 frames):
  c_s 0.75 4-var: S_D +0.0070 +- 0.0015, D_label -0.033, geom 0.251.
  sel 0.1 4-var: S_D +0.0048 +- 0.0021, D_label -0.040, geom 0.206.
  c_s 0.75 + sel 0.1, 4-var: S_D 0.8963 (+0.0111 +- 0.0026, z 4.2), D_label 0.308 (-0.061 +- 0.009), li R2 0.550,
  kappa R2 +0.09, Z_axis R2 0.637, geom 0.211.
  c_s 0.75 + sel 0.1, 8-var: S_D 0.9040 (+0.0186 +- 0.0028, z 6.6), D_label 0.280, li R2 0.575, kappa R2 0.22,
  Z_axis 0.668, geom 0.206. (8 variants ~doubles generation time.)
  MAST fixed: ts075 0.231, 0.273; tesd 0.277. Queue reprioritised: tss x3 (c_s 0.75 + sel 0.1), tss8 (n_var 8),
  tsel x3, ts075_r2. Pilots queued: h1_sel, h1_sel_s075.
- 03:28 EARLY SET READY: tune3/EARLY_READY -> /fsx/chris.beckham/fusion_cache/synth/gsolve_cond2z1000_limiter/shape_targets.npz
  (data /fsx/chris.beckham/gsolve_data/cond2z1000/, practice set g4_gsc2z1000, build exit 0, truth/perfect S 1.000).
  Recipe = knobs_early_h1.json (knobs_cond_base + zc_lo = zc_hi = 0.83, zc_sd 0.015, incumbent teacher).
- 03:30 Final-set plumbing: knobs_final_a.json (= knobs_early_h1 + c_s 0.75 + sel_crit 0.1, sel_crit_model =
  sel_critic_v1.pkl, a frozen read-only copy), knobs_final_b.json (+ n_var 8). final3.sh = final2.sh with SHARDS
  processes (coordinator: gen_param2 speed scales with processes, not threads; 3 shards x 4 var took ~2 h).
  Dry run started: NAME=cdry3 (30 shots, 6 shards, knobs_final_a) -> dry3.log; clean with dry_cleanup.sh cdry3
  gsolve_cdry3 g4_gscdry3. run_sel3.sh: c_s 0.75 + d_li -0.05 / +0.05 replays (4 var) for post-hoc sel.
- 03:42 dry run cdry3 passed (gen 12 min for 30 shots on 6 x 1-thread shards with sel_crit, build exit 0, truth S 1.000); cleaned with dry_cleanup.sh.
- 04:04 c_s 0.75 + sel 0.1 + d_li -0.05 (4 var, honest): S_D +0.0004 +- 0.0013, D_label -0.002 vs combined, geom 0.225
  (combined 0.211) -> not adopted. d_li +0.05 replay killed (CPU load ~69 from pilot training; low priority).
  MAST fixed: tss 0.262, tss_r1 0.287 (mean 0.274; guard 0.278), tsel 0.273, ts075 0.231/0.273.
- 04:10 MAST critic combined tss (c_s 0.75 + sel 0.1): 0.262 / 0.287 / 0.287 = 0.279 vs guard 0.278 -> fails by 0.001
  (borderline). tss8 (n_var 8) seed 0: 0.243, but 2 more seeds + a ~70 min n_var 8 final set do not fit before 06:15
  -> n_var 8 dropped. Running tsel_r1, tsel_r2, ts075_r2 in parallel (third runner, logs/runner_e.log) to decide
  between sel 0.1 alone and c_s 0.75 alone (z rule: tsel ~2.3 + 4.9 geom, ts075 4.6 + 0).
- 04:16 tsel_r2 0.249 -> sel 0.1 MAST fixed 3 seeds 0.273 / 0.282 / 0.249 = 0.268 (guard 0.278): PASSES.
  DECISION (rule 3, z = dS_D/sd + dgain/sd among passers): h1 + sel_crit 0.1 (z ~2.3 + ~4.9) over h1 + c_s 0.75
  (4.6 + 0; ts075_r2 not run); combined c_s 0.75 + sel failed the guard by 0.001; n_var 8 does not fit the timeline.
  FINAL RECIPE = knobs_final_sel.json (knobs_early_h1 + sel_crit 0.1, sel_crit_model sel_critic_v1.pkl, 4 variants).
  Search runners stopped (queue.txt.stop). FINAL SET started 04:17: `NAME=cond3_1000 TAG=gsolve_cond3_1000
  SET=g4_gsc3_1000 READY=tune3/FINAL_READY SKIP_EXTRA=1 SHARDS=12 GEN_THREADS=3 CONV_WORKERS=24 BUILD_WORKERS=20
  ./final3.sh knobs_final_sel.json` -> final.log, /fsx/chris.beckham/gsolve_data/cond3_1000/. Resume = same command.
- 04:38 final set: variant 0 took ~1,100 s per shard (machine load 55-86 from pilots/training) -> ETA FINAL_READY ~05:45.
- 05:38 FINAL_READY written: /fsx/chris.beckham/fusion_cache/synth/gsolve_cond3_1000_limiter/shape_targets.npz
  (cond3_1000, build exit 0, truth S 1.000; selection critic p(real) median ~0.42 on MAST accepted variants).
