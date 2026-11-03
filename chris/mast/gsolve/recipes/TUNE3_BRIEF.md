# tune3 brief: DIII-D flux-map-supervised conditional generator (17 Oct 2026, 00:20 UTC)

## Goal
Make the synthetic MAST training data more realistic in the parts the MAST input critic cannot see (plasma height,
shape, internal current), using DIII-D EFIT flux maps and labels (allowed) as the teacher and judge. Deliver a frozen
recipe plus a 1,000-shot set by **~06:45 UTC**; an early set with the first safe win by **~03:30 UTC**.

## Algorithm
The generator is machine-independent: inputs → features z → hidden settings θ(z) → gsolve equilibrium. We fit θ(z) on
DIII-D, **replay the same recipe on held-out DIII-D shots from DIII-D inputs only**, and score the generated flux maps
against EFIT flux maps with the leaderboard formula. MAST gets the recipe that wins on DIII-D, provided the MAST input
critic says it does no harm.

Three judges, in this order:
1. **S_D (DIII-D replay score).** `d3d_score.py`, leaderboard formula, on held-out DIII-D shots. Two upgrades make it
   honest:
   - **E-coil calibration.** The DIII-D raw flux error is a per-frame constant set by the E-coil (gsolve's ECOILA is
     ~146× too weak; offset = k·I_ec, k fitted on TRAIN frames, CV R² 0.996). Apply k·I_ec to predicted maps (or fix
     the scale in gen_d3d) so the raw S_D is meaningful (target: within 0.02 of S_off ≈ 0.90).
   - **Realistic edges.** gen_d3d currently uses EFIT midplane edges as an oracle. Replace them with DIII-D Thomson
     edges (vertical core chord; outermost channel with Te ≥ 5% of peak, its ψN surface's midplane radii), the same
     definition used on MAST. Refit the teacher (`fit_prior.py`) on z computed from Thomson edges. Expect S_D to drop
     for every recipe; that is the honest baseline.
2. **DIII-D flux-geometry critic (`d3d_critic.py`, new).** Real (z, EFIT map) vs generated (z, gsolve map) on the same
   held-out DIII-D frames. Features from the maps in normalised coordinates (~30): LCFS ρ(θ)/a Fourier modes 1–4, κ,
   δ_up, δ_low, squareness, X-point position, R_ax, (Z_ax − 0.77·z_bal)/a, Shafranov shift and elongation of the
   ψN 0.3 and 0.6 contours, li, q at ψN 0.5/0.95; plus z as conditioning. Classifier and CV from `critic.fit_critic` /
   `heldout_eval`, GroupKFold by shot, 3 seeds; objective = gain (ln2 − log loss). This judges whole distributions,
   so it catches spreads that are too narrow (D_label alone rewards narrow spreads). Calibrate its noise on existing
   npz (cond, cond_r1, s05, lkm, lip, ridge, th0) before using it.
3. **MAST input critic** (existing frozen protocol, fixed-frame scale, 3 seeds): no-harm guard only (not worse than
   cond by > 0.02 on the 3-seed mean). Also report OUTWIN features as a tie-breaker.

## Changes to search (machine-independent knobs only)
In priority order:
1. **Height rule.** EFIT Z_axis ≈ 0.77·z_bal on DIII-D (R² 0.71); recipe draws zc ~ U(0,1)·z_bal (Z_axis R² 0.44);
   gsolve Z_ax ≈ 0.94·zc. Refit slope s on d3d/train; candidates zc_lo = zc_hi = s/0.94 (zc_sd ~0.015–0.02) and the
   spread-keeping variant zc_lo = 2s/0.94 − 1, zc_hi = 1. This is the first safe win: once DIII-D Z_axis R² ≥ 0.60,
   paired D_label/S_D not worse beyond noise, and the MAST critic no worse, **start the early 1,000-shot set** (see
   Deliverables).
2. **Spread c_s** ∈ {0.5, 0.75, 1.0}: decided by the flux-geometry critic and S_D together.
3. **li history features** in z (Ip(t−kτ)/Ip(t), k = 0.1/0.25/0.5; time since Ip > 100 kA over τ; EW Ip history),
   filled in d3d_table.py, mast_z.py, gen_param2.frame_z; check MAST inside DIII-D 1–99% range; refit fit_prior and
   fit_emu. Expected li +0.04 R², and κ follows li (κ error corr −0.65 with li error).
4. **d_li ±0.05–0.075, d_lk**, at the best of the above.
5. **Edge selection** (`sel_out` 0.03, `edge_bias` −0.01) only if the realistic-edge DIII-D replay shows a systematic
   edge offset; translate as a fraction of minor radius.

Not doing: real limiter wall in gsolve (measured ~0 effect), am/an inversion (not identifiable), a vessel-field
teacher (machine-specific), a spatial flux-map correction (residual after offset is 0.011 Wb), any leaderboard test
of the MAST solenoid scale (that would be a leaderboard probe).

## Decision rule
1. Gate: S_D (and D_label) paired shot-bootstrap difference to the current incumbent not worse by more than 1 sd;
   flux-geometry critic gain not worse beyond its measured noise.
2. MAST critic not worse than cond by > 0.02 (3-seed fixed-frame mean).
3. Rank passers by z = ΔS_D/sd + Δgain_geom/sd; adopt only clear wins (z beyond ~2), else keep the incumbent.
4. The height rule may be adopted on DIII-D Z_axis R² and S_D alone (the MAST critic cannot see it), subject to (2).

## Deliverables and timing (UTC)
- ~03:30 at the latest: **early set** `cond2z1000` = knobs_cond_base + height rule (if it passed), via the final2.sh
  pipeline with new names (data `/fsx/chris.beckham/gsolve_data/cond2z1000/`, cache `gsolve_cond2z1000(_limiter)`,
  practice set `g4_gsc2z1000`), flag file `tune3/EARLY_READY`. Use ≤ 14 generation threads so the search continues.
- 05:30 freeze; **final set** `cond3_1000` (names analogous: `gsolve_cond3_1000`, `g4_gsc3_1000`), flag
  `tune3/FINAL_READY`, ready by ~06:45. If the final recipe equals the early recipe, write FINAL_READY pointing at the
  early set instead of regenerating.
- `tune3/REPORT.md`: what changed, numbers for every judged candidate, decision trail. Keep `tune3/HANDOFF.md` current
  (timestamped log) so another agent can resume after a crash.

## Rules
- Never read MAST labels (efit_* except efit_times, efit_grid_R, efit_grid_Z); never use the 3 MAST demo shots; never
  use fusion_scoring/masks/mast_envelope.npz (run physics_suite only with FEC_MAST_MASK=limiter); never use
  /fsx/artur.toshev/synthetic_mast_v5 or check_v5.json. /fsx/zara.ercan, /fsx/artur.toshev read-only. Do not modify
  `gsolve_data/tune/` or the existing `tune2/` (copy what you need into `tune3/`). No leaderboard use of any kind.
- Fit anything on DIII-D TRAIN shots; judge on DIII-D VAL shots.
- CPU only (`CUDA_VISIBLE_DEVICES=`), ≤ 40 cores total. GPUs are reserved for training by the coordinator.
  GPU 4 belongs to Zara's job: never touch it.
- Kill processes by PID only (never pkill -f). Do not delete anything outside tune3/ scratch you created.
- Do not train models or build submissions.
