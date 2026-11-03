# Synthetic MAST smoke-test set (500 + 64 cases), calibrated on MAST inputs only

Location: `/fsx/zara.ercan/more_data/m2/gen4_set500v2/`

| Folder / file | Contents |
|---|---|
| `mast_g4s500v2_train/` | 500 cases from training-pool shots, `mast_synth_g4_XXXXX.parquet`, 3,016 labelled frames |
| `mast_g4s500v2_heldout/` | 64 cases from held-out shots (the MASTVAL shot pool), 415 labelled frames. Do not train on these. |
| `SHA256SUMS.txt` | checksums of the 564 files |
| `raw/`, `done/`, `candidates.json`, `assembly.json` | every attempted candidate, its outcome, the sampling plan |
| `code/` | the generator code used (`gen4.py`, `gen4_set.py`, `pool.py`) |

Built to the requests in Chris's `FEEDBACK_FOR_ZARA_TRAINING.md` (version of 2026-10-08 01:13) and `FEEDBACK_FOR_ZARA.md`.
Generator: `my_experiments/m2/gen4/gen4_set.py --recipe v2` on top of `gen4.py`; checks: `check.py`, `verify_set.py`.

## What a case is

A short sequence of 3 to 9 labelled equilibria at consecutive real EFIT times (5 ms apart in most shots), centred on a
real `mast_public_test` frame with |Ip| ≥ 150 kA. On average a case has 6.0 labelled frames; 100 of the 500 training
cases have all 9. The centre frame is solved first, then a sweep forward and backward in time with warm starts, so the
equilibria evolve slowly with the real inputs.

Each file follows the challenge MAST schema in the layout of the earlier 500-case release:

- `magnetics_*`: the real input series in ±100 ms around the centre frame. Coil groups are scaled by small per-case factors
  (P4/P5 ±1 %, P2/P3/P6 ±3 %, solenoid ±5 %, Ip and TF ±1.5 %). The solver uses the solenoid at ×0.5 of the column (challenge
  note); the column keeps the challenge value.
- `thomson_core_*`, `thomson_edge_*`: a synthetic profile at every real Thomson time within 30 ms of a labelled frame.
- `efit_times`, `efit_psirz` (T, 65, 65; rows Z, columns R), `efit_q95`, `efit_beta_n`, `efit_li`, `efit_beta_p`,
  `efit_r_axis`, `efit_z_axis`, `efit_elongation`, `efit_psi_axis`, `efit_psi_boundary`, `efit_geom_axis_r/z`,
  `efit_lcfs_n/r/z` (rows padded with zeros to one length, valid count in `efit_lcfs_n`, as in the challenge files).
- `synth_meta` (JSON): origin shot and frame, stratum, per-case draws, generator version and recipe.
  `synth_frames` (JSON, one entry per frame of the 9): status, solver path, pressure ratio, profile parameters, boundary vs
  Thomson-edge diagnostics, timing.

## Plasma model (everything not listed is the real input of that frame)

- **Pressure.** p(x) = (1 + Ti/Te) p_e(x) + p_fast(x), x = normalised flux. p_e = e·ne·Te of the frame's **real** Thomson
  profile, fitted on the solution's own flux surfaces (map, solve, re-map, re-solve), with zero gradient at the boundary.
  Ti/Te (0.7–1.6) and a fast-ion share (0–50 % of the core pressure, profile (1 − x²)^nf) are not seen by Thomson; they
  are drawn per case and drift slowly between frames.
- **Pressure level.** One overall factor is moved frame by frame until the outer midplane boundary sits at the real
  Thomson outer edge (where the smoothed Te falls below a boundary temperature drawn per case in 20–60 eV). Frames between
  Thomson bursts use the nearest target within 25 ms. The total-to-electron pressure ratio stays within 1.2–5.0.
- **Current profile, independent of Thomson.** FF′ ∝ (1 − x^am)^an with am ∈ [1, 3] and an ∈ [0.5, 3.5] (log-uniform),
  drawn per case; an drifts slowly between frames. Nothing in FF′ depends on the Thomson profiles. The total current equals
  the frame's Ip.
- **Acceptance.** Converged and diverted (not limited), axis 0.45 < R < 1.3 m and |Z| < 0.15 m, frame-to-frame |ΔZ| < 3 cm,
  volume 1–20 m³, 1.5 < q95 < 40, 0.3 < li < 3, βN < 6, outer boundary within 5 cm of the Thomson edge target.

## Synthetic Thomson

- **Electron pressure only.** Thomson shows ne and Te, whose product is the electron pressure of the label. Ti/Te and the
  fast ions are hidden from it.
- **Te and ne vary independently of the current and of each other at fixed pressure.** Per case, ne is multiplied and Te
  divided by f(x) = exp(c (x² − 1/3)), c drawn in [−0.6, 0.6]. The same pressure and current then come with different Te
  and ne shapes (core ratio up to ±22 %, edge up to ±49 %).
- **Each time slice has its own noise.** For each real Thomson time, that slice's own measured ne and Te are fitted on the
  nearest labelled frame's flux surfaces and evaluated at the chord radii. Multiplicative channel noise at the scatter
  measured on that real slice (5-channel running median) is independent between slices.
- **Real gaps.** The real slice's missing channels stay missing.
- Scrape-off-layer decay outside the boundary, with decay lengths drawn per case.

## Sampling

Real frames (|Ip| ≥ 150 kA, all public-test shots, inputs only) are split into 15 strata: Ip quintile × tercile of
dIp/dt / Ip. Each stratum's quota is its share of real frames × the set size. Candidates are drawn at random inside each
stratum. A candidate with fewer than 3 labelled frames, or without a core Thomson slice of at least 20 valid channels, is
replaced by the next candidate of the same stratum, so solver failures do not change the mix. Every stratum met its quota.

## Checks (`check.py`, inputs only; real = the origin shot's Thomson at the same times)

| | train (500 cases) | held-out (64) | real MAST |
|---|---|---|---|
| labelled frames per case (mean) | 6.0 | 6.5 | |
| Thomson Te peak, eV (5 / 50 / 95 %) | 345 / 677 / 1290 | 358 / 630 / 1100 | 369 / 687 / 1270 |
| Thomson ne peak, 1e19 m⁻³ | 1.64 / 3.06 / 5.06 | 1.00 / 3.41 / 4.98 | 1.79 / 3.35 / 5.96 |
| missing channels per slice (mean) | 6.7 % | 7.3 % | 6.7 % (train windows) |
| slice-to-slice jitter of the p_e peak radius, cm (median / 90 %) | 2.6 / 7.7 | 2.6 / 6.8 | 2.6 / 14.9 |
| slice-to-slice relative jitter of the mean p_e (median / 90 %) | 3.3 % / 11.7 % | 3.0 % / 11.3 % | 3.1 % / 10.8 % |
| case-centre Ip, kA (5 / 25 / 50 / 75 / 95 %) | 412 / 546 / 641 / 754 / 890 | 413 / 537 / 634 / 753 / 791 | 401 / 536 / 632 / 751 / 862 (Chris's coverage table) |
| outer boundary − real Thomson 30 eV edge, cm (5 / 50 / 95 %) | −7.9 / −1.0 / 3.1 | −8.9 / −0.9 / 3.0 | |
| rank correlation of li with synthetic Te peaking | 0.26 | 0.23 | |

Labels (train, 5 / 50 / 95 %): q95 4.75 / 6.57 / 9.61, βN 0.58 / 1.78 / 4.11, li 0.65 / 0.90 / 1.27, κ 1.63 / 1.85 / 2.12,
R_axis 0.85 / 0.91 / 0.99 m, Z_axis −0.03 / 0.00 / 0.04 m, βp 0.13 / 0.40 / 0.84, volume 6.7 / 7.8 / 8.9 m³,
total / electron axis pressure 1.47 / 3.44 / 5.0.

`verify_set.py`: all 564 files load into rectangular arrays, with no NaN or Inf in ψ or the scalar labels.

Compute: 2,950 candidates, 44.5 CPU-hours of solver time (FreeGSNKE 3.1.1, single-threaded workers on AWS g7e8 nodes).

## Rules position (Rule 3)

Read: MAST `magnetics_*` currents, `thomson_*` profiles and positions, coil and chord geometry, `efit_times`,
`efit_grid_R/Z`. Not read: any other `efit_*` column of a MAST file, the demo shots, `mast_envelope.npz`, FAIR-MAST.
The cases are calibrated on Thomson (an input) and not fitted to any magnetic measurement. The seed library used for
initial guesses holds our own earlier synthetic solutions.

## Known limits

- FreeGSNKE has no vessel currents, and the limiter is drawn by hand from the coil positions.
- At the ×0.5 solenoid convention many frames need a high pressure ratio to put the boundary at the Thomson edge (median
  3.4, at least 5 % of frames at the cap of 5.0; an independent input-only estimate gives 2.9). Tested in `CAVEATS.md`:
  at ×1.0 the same frames need about half the pressure, βN is about half and q95 about a quarter lower, while li and κ
  barely change. The set's βN and q95 labels are therefore right only if ×0.5 is the right convention, which no label-free
  check can decide. The ratio is stored per frame in `synth_frames` and can be used to filter.
- li correlates with the synthetic Te peaking (rank 0.26) although the current-shape function ignores Thomson. Tested in
  `CAVEATS.md`: the route is the pressure-driven part of the current, R p′ (with a fixed pressure shape the correlation
  falls to 0.10, and to −0.03 after controlling for the pressure peaking). Its strength scales with the pressure in the
  labels.
- Frames near the centre column (inner gap below about 3 cm) and ramp-up frames solve less often. Their inputs are sampled
  at the real rate, but the labelled frames under-represent them.
- Synthetic ne peaks run about 9 % below real, and the tail of the pressure-peak jitter is shorter than real (90th
  percentile 7.7 against 14.9 cm).
