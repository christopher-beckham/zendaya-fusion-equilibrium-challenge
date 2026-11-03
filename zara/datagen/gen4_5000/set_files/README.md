# Synthetic MAST training set (4,874 + 616 cases), calibrated on MAST inputs only

| Folder / file | Contents |
|---|---|
| `mast_g4s5000_train/` | 4,874 cases from training-pool shots, `mast_synth_g4_XXXXX.parquet`, 29,492 labelled frames (2.2 GB) |
| `mast_g4s5000_heldout/` | 616 cases from held-out shots (the MASTVAL shot pool), 3,819 labelled frames (284 MB). Do not train on these. |
| `SHA256SUMS.txt` | checksums of the 5,490 files |
| `CAVEATS.md` | tests of two properties of the labels (li vs Te peaking; pressure ratio and βN vs the solenoid convention) |
| `raw/`, `done/`, `candidates.json`, `code/` | every attempted candidate, its outcome, the sampling plan, the generator code |

Same recipe and file layout as the 500-case smoke-test set (`/fsx/zara.ercan/more_data/m2/gen4_set500v2/README.md` describes
the method in full). Built to Chris's `FEEDBACK_FOR_ZARA_TRAINING.md` (2026-10-08) and `FEEDBACK_FOR_ZARA.md`.

## In short

- **Case.** 3 to 9 labelled equilibria at consecutive real EFIT times (6.1 per case on average; 1,039 training cases
  have all 9), centred on a real `mast_public_test` frame with |Ip| ≥ 150 kA, solved in time order with warm starts.
  Every labelled frame lies inside the stored ±100 ms of inputs.
- **File.** Challenge MAST schema: real `magnetics_*` series (coil groups scaled by small per-case factors), synthetic
  `thomson_*`, `efit_*` labels at the labelled frames (ψ rows Z, columns R; LCFS rows padded with zeros, count in
  `efit_lcfs_n`), `synth_meta` and `synth_frames` (JSON diagnostics).
- **Plasma.**
  - Pressure: electron pressure from the frame's real Thomson on the solution's own flux surfaces, plus Ti/Te and
    fast-ion parts that Thomson does not see.
  - Pressure level: set so the outer boundary meets the real Thomson edge.
  - Current: shape drawn independently of Thomson.
  - Solenoid at ×0.5 of the column.
- **Synthetic Thomson.**
  - Electron pressure only.
  - Random Te/ne split at fixed pressure.
  - Channel noise at each real slice's measured scatter, independent between slices.
  - Real missing channels kept.
- **Sampling.** 15 strata (Ip quintile × dIp/dt tercile) with quotas equal to their share of real frames. Failed
  candidates are replaced inside the stratum. Every real frame was a candidate.

## Coverage

All strata met their quota except one: Ip 474–608 kA with the plasma current ramping up (dIp/dt / Ip above
+6.7 % per 100 ms). That stratum has 330 of 456 training cases and 34 of 58 held-out cases. Only 6 % of its frames
solve: ramp-up frames with a small gap to the centre column usually end limited. Every real frame of the stratum was
tried, so this ramp-up regime is under-represented (6.8 % of training cases instead of 9.1 %).

## Checks (inputs only; real = the origin shot's Thomson at the same times)

| | train | held-out | real MAST |
|---|---|---|---|
| Thomson Te peak, eV (5 / 50 / 95 %) | 350 / 675 / 1260 | 321 / 620 / 1210 | 376 / 686 / 1260 |
| Thomson ne peak, 1e19 m⁻³ | 1.49 / 3.06 / 5.19 | 1.51 / 3.36 / 5.11 | 1.66 / 3.32 / 5.86 |
| missing channels per slice | 6.6 % | 6.7 % | 6.6 % |
| slice-to-slice jitter of the p_e peak radius, cm (median / 90 %) | 2.6 / 7.3 | 2.6 / 7.2 | 2.6 / 14.9 |
| slice-to-slice relative jitter of the mean p_e (median) | 3.2 % | 3.1 % | 3.0 % |
| case-centre Ip, kA (5 / 50 / 95 %) | 417 / 644 / 879 | 418 / 642 / 803 | 401 / 632 / 862 |
| outer boundary − real Thomson 30 eV edge, cm (5 / 50 / 95 %) | −9.0 / −1.0 / 3.1 | −7.3 / −1.0 / 2.9 | |

Labels (train, 5 / 50 / 95 %): q95 4.7 / 6.45 / 9.8, βN 0.55 / 1.72 / 4.18, li 0.66 / 0.91 / 1.29, κ 1.63 / 1.84 / 2.11,
R_axis 0.85 / 0.91 / 0.99 m, Z_axis −0.03 / 0.00 / 0.04 m, volume 6.6 / 7.8 / 8.9 m³, total / electron axis pressure
1.41 / 3.35 / 5.0. All 5,490 files load into rectangular arrays, with no NaN or Inf in ψ or the scalar labels.

Compute: 24,762 candidates, 374.5 CPU-hours of solver time (FreeGSNKE 3.1.1, single-threaded workers on AWS g7e8 and
gpu2 nodes).

## Known limits

- **βN and q95 depend on the solenoid convention.** At ×0.5 the boundary reaches the Thomson edge only with a high
  pressure ratio (median 3.4). At ×1.0 the same frames need about half the pressure, giving about half the βN and
  about 25 % lower q95. li and κ barely change. See `CAVEATS.md`.
- **li follows the pressure profile.** li correlates with the synthetic Te peaking (rank 0.24) through the
  pressure-driven current R p′, which is an equilibrium effect. The current-shape function itself ignores Thomson.
- **Machine model.** FreeGSNKE runs without vessel currents and with a hand-drawn limiter.
- **Profile detail.** Synthetic ne peaks run about 8 % below real, and the pressure-peak jitter has a shorter tail than
  real.

Rules position: only MAST inputs (magnetics currents, Thomson, coil and chord geometry, `efit_times`, `efit_grid_R/Z`) are
read. No MAST label, demo shot, envelope mask or FAIR-MAST data is used, and the cases are not fitted to magnetic
measurements.
