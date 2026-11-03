# tune3 report: DIII-D flux-map-supervised conditional generator (17 Oct 2026)

## Outcome
- **Early set** (03:28, `EARLY_READY`): `cond2z1000` = knobs_cond_base + height rule (zc = 0.83 z_bal, zc_sd 0.015).
  Cache `gsolve_cond2z1000_limiter`, practice set `g4_gsc2z1000`. Knobs: `knobs_early_h1.json`.
- **Final set** (`FINAL_READY`): `cond3_1000` = early recipe + critic-guided variant selection (sel_crit 0.1,
  `sel_critic_v1.pkl`, 4 variants, c_s 1, incumbent teacher). Cache `gsolve_cond3_1000_limiter`, practice set
  `g4_gsc3_1000`. Knobs: `knobs_final_sel.json`. Generated with `final3.sh` (12 shards x 3 threads).

## Judges
1. **S_D**, from `d3d_score.py`: the leaderboard formula on 249 held-out DIII-D VAL shots (1,932 frames).
   - E-coil calibration: the predicted map gets + k I_ec + b, with k = -0.00909 Wb/kA and b = -0.0042. These were fitted on
     479 TRAIN frames; shot-grouped CV R2 is 0.996 (`ecoil_cal.py`, `ecoil_cal.json`). Calibrated S_D comes within
     0.002 of the oracle-offset S_off.
   - Honest edges: Thomson edges replace the EFIT midplane edges everywhere (edge targets, z, z_bal, pass-1 tables).
     The Thomson edge is the outermost core vertical-chord channel with Te >= 5% of the smoothed peak, mapped to its
     EFIT psi_N surface (median 0.975) and then to that surface's midplane radii (`d3d_table3.py`, `make_th.py`).
   - Paired shot bootstrap (200 resamples) gives the sd of the differences (`compare_d3d.py`).
2. **Flux-geometry critic**, from `d3d_critic.py`. It compares 22 normalised map features plus 15 conditioning inputs:
   - map features: LCFS Fourier modes 1-4, kappa, upper/lower triangularity, squareness, axis shift/a,
     (Z_ax - 0.77 z_bal)/a, shift and elongation of the psi_N 0.3/0.6 surfaces, li, and q at psi_N 0.5/0.95;
   - method: 5-fold CV by shot, 3 seeds; gain = ln2 - log loss.
   - Generation-seed noise is about 0.009 (cond 0.284 vs cond_r1 0.275).
3. **MAST input critic**: the frozen protocol on the fixed-frame scale. The guard requires a 3-seed mean <= cond (0.258) + 0.02.

## Numbers
Oracle-edge DIII-D judge (1,370 common frames, paired vs cond):

| recipe | S_D | dS_D | D_label | Z_axis R2 | li R2 | kappa R2 | geom gain | MAST fixed |
|---|---|---|---|---|---|---|---|---|
| cond (incumbent) | 0.9033 | 0 | 0.302 | 0.456 | 0.523 | 0.32 | 0.284 | 0.250/0.245/0.279 = 0.258 |
| cond_r1 (seed repeat) | 0.9039 | +0.0005 +- 0.0024 | 0.296 | 0.430 | 0.575 | 0.34 | 0.275 | |
| h1 (zc = 0.83 z_bal) | 0.9117 | +0.0083 +- 0.0017 | 0.269 | 0.686 | 0.515 | 0.29 | 0.248 | 0.256/0.268/0.274 = 0.266 |
| h2 (zc ~ U(0.66,1) z_bal) | 0.9110 | +0.0076 +- 0.0016 | 0.271 | 0.678 | 0.509 | 0.29 | 0.253 | 0.277/0.299 (fails) |

Honest judge (Thomson edges; paired vs the reference named in each row):

| recipe | S_D | dS_D (ref) | D_label | Z_axis R2 | li R2 | kappa R2 | geom gain | MAST fixed |
|---|---|---|---|---|---|---|---|---|
| thc (incumbent) | 0.8771 | | 0.405 | 0.421 | 0.414 | -0.20 | 0.294 | 0.258 |
| thc_h1 = early set | 0.8852 | +0.0081 +- 0.0015 (thc) | 0.368 | 0.631 | 0.401 | -0.16 | 0.250 | 0.266 |
| tht_h1 (Thomson-edge teacher) | | +0.0012 +- 0.0017 (thc_h1) | 0.355 | 0.647 | 0.396 | -0.12 | 0.247 | 0.259/0.282/0.284 = 0.275 |
| thh_h1 (+ Ip-history teacher) | | +0.0037 +- 0.0021 (thc_h1) | 0.341 | 0.642 | 0.456 | -0.07 | 0.2495 | 0.291/0.278/0.293 = 0.287 FAIL |
| h1 + c_s 0.75 | 0.8922 | +0.0070 +- 0.0015 (thc_h1) | 0.336 | 0.615 | 0.495 | -0.07 | 0.251 | 0.231/0.273 (r2 not run) |
| h1 + edge_sd 0.005 | | (DIII-D not run) | | | | | | 0.277 (1 seed) |
| **h1 + sel 0.1 (4 var) = FINAL** | 0.8897 | +0.0048 +- 0.0021 (thc_h1) | 0.328 | 0.651 | 0.495 | +0.03 | 0.206 | 0.273/0.282/0.249 = 0.268 |
| h1 + c_s 0.75 + sel 0.1 (4 var) | 0.8963 | +0.0111 +- 0.0026 (thc_h1) | 0.308 | 0.637 | 0.550 | +0.09 | 0.211 | 0.262/0.287/0.287 = 0.279 FAIL (by 0.001) |
| ... + d_li -0.05 | 0.8999 | +0.0004 +- 0.0013 (combined) | 0.297 | | 0.592 | +0.08 | 0.225 | |
| h1 + sel 0.1, 8 var | | +0.0074 +- 0.0024 (8-var, no sel) | 0.313 | 0.649 | 0.490 | +0.13 | 0.198 | |
| h1 + c_s 0.75 + sel 0.1, 8 var | 0.9040 | +0.0186 +- 0.0028 (thc_h1) | 0.280 | 0.668 | 0.575 | +0.22 | 0.206 | 0.243 (1 seed) |

(The table rows on 1,392 common frames come from d3dchk/compare_selva_l0p0_v4_*.csv; the other honest rows come from
compare_thc_h1_*.csv.)

## What changed and why
1. **Height rule** (the first safe win). On d3d/train, EFIT Z_axis = 0.764 z_bal (R2 0.64), and the scorer's Z_axis on
   generated maps = 0.92 zc. The recipe zc ~ U(0,1) z_bal therefore becomes zc = 0.83 z_bal.
   - On the honest judge, Z_axis R2 goes from 0.42 to 0.63 and S_D rises by 0.008 (z 5.4).
   - The MAST critic is no worse.
   - On MAST, the generated Z_axis p10/p90 widens from -0.037/0.033 to -0.062/0.049 m.
2. **Critic-guided selection** (algorithm change; `sel_critic.py`, gen_param2 knobs `sel_crit`/`sel_crit_model`).
   - A real-vs-generated flux-geometry classifier is trained on a TRAIN-shot replay of 200 shots x 4 frames with 8
     variants. Its inputs are 20 geometry features plus Thomson-edge z and z_bal/a; CV gain is 0.21.
   - Squareness and q50 are excluded because MAST lies outside the DIII-D 1-99% range on more than 20% of frames.
     All other features are inside the range.
   - In the generator, accepted variants are re-ranked by err - 0.1 logit p(real).
   - Evaluation uses the same solved variants, with the judges fitted on VAL shots. With 4 variants: S_D +0.005,
     D_label -0.04, kappa R2 from -0.22 to -0.02, li R2 from 0.40 to 0.48, and geometry gain from 0.250 to 0.206.
   - With 8 variants the gains roughly double. That option was not used because of time.
   - Caveat: the geometry judge uses the same feature family the selection optimises, so its gain is partly by
     construction. The S_D and D_label gains are measured against EFIT labels and do not have this problem.
3. **Rejected changes**:
   - Thomson-edge teacher: no gain.
   - Ip-history teacher: li R2 +0.05 and S_D z 1.8 on DIII-D, but it fails the MAST guard, and its MAST li median
     drops from 1.12 to 1.00.
   - c_s 0.75: wins on DIII-D, but it fails the guard combined with selection, and as a single change it ranks below
     selection.
   - d_li -0.05: no gain.
   - Edge knobs: the honest edge offset is small (generated Rmax - EFIT = -0.5 cm, a = 59 cm), so they stayed low priority.

## Geometry-critic giveaways (idea b)
Permutation importance on thc_h1: ax_R 0.54, s06 0.45, k03 0.11, kappa 0.05, ax_Zbal 0.05.
- The main giveaway is how far gsolve shifts the magnetic axis compared with the psi_N 0.6 surface.
  s06/ax_R has median 0.58 in EFIT vs 0.53 in the generated maps, and the two do not track each other frame by frame
  (corr -0.01). This points at the shape of the core current/pressure profile.
- The selection critic sees these features, which lowers the judge gain.
- A dedicated theta target for this shift profile was not attempted.

## Notes for the trainer
- Final set = cond3_1000 (see FINAL_READY). Early set = cond2z1000.
- Both sets share the incumbent teacher and c_s 1, and both include the height rule. Only the final set uses selection.
- sel_critic_v1.pkl is frozen (read-only); refits must use another name.
- MAST-side giveaway rel_Rpk_ne (broad outboard profiles) was not addressed; only the MAST input critic can judge it.

## Generated MAST label distributions (p10 / median / p90, all kept equilibria)
| set | li | kappa | Z_axis (m) | betaN | q95 |
|---|---|---|---|---|---|
| cond1000 (previous) | 0.878 / 1.151 / 1.497 | 1.508 / 1.676 / 1.851 | -0.037 / 0.001 / 0.032 | 0.35 / 0.86 / 2.12 | 4.48 / 5.92 / 8.29 |
| cond2z1000 (early) | 0.878 / 1.146 / 1.490 | 1.509 / 1.672 / 1.843 | -0.063 / 0.002 / 0.048 | 0.36 / 0.88 / 2.15 | 4.28 / 5.82 / 8.20 |
| cond3_1000 (final) | 0.878 / 1.125 / 1.421 | 1.526 / 1.682 / 1.813 | -0.064 / 0.001 / 0.046 | 0.36 / 0.89 / 2.17 | 4.29 / 5.87 / 8.29 |

FINAL_READY written 05:38 UTC (generation 04:17-05:34 on 12 shards, conversion and build 4 min, build exit 0).
