# Synthetic MAST training set, recipe v3 (5,000 + 640 cases), calibrated on MAST inputs only

| Folder / file | Contents |
|---|---|
| `mast_g4s5000v3_train/` | 5,000 cases from training-pool shots, `mast_synth_g4_XXXXX.parquet`, 30,144 labelled frames (2.3 GB) |
| `mast_g4s5000v3_heldout/` | 640 cases from held-out shots (the MASTVAL shot pool), 4,077 labelled frames (301 MB). Do not train on these. |
| `SHA256SUMS.txt` | checksums of the 5,640 files |
| `raw/`, `done/`, `candidates.json`, `code/` | every attempted candidate, its outcome, the sampling plan, the generator code |

Generator version `gen4.9`, recipe `v3` (stored in every file's `synth_meta`). Seeds 3,000,000 + candidate id: no case
repeats a case of the 500, 5,000 or 20,000 sets. Held-out cases come only from held-out shots, which no training set uses.

## What v3 adds to the recipe of `gen4_set5000`

Built to the requests of Chris's `FEEDBACK_FOR_ZARA_TRAINING.md` (version of 2026-10-11).

1. **Plasmas limited on the centre column.** On frames where the real Thomson is hot at its innermost channels
   (median Te > 50 eV for R < 0.26 m), an equilibrium resting on the centre column counts as a valid solution, if it
   stays clear of the outer wall. 15.7 % of the training frames (4,721) are limited on the column. They make up 45 % of
   the accepted frames that have a hot innermost channel. Other frames must still be diverted. `synth_frames` records
   `limited` and `inner_hot` per frame; the labels of a limited frame use the boundary flux surface that touches the
   column.
2. **More shape variety.** Coil currents are scaled independently for the upper and lower coil of each pair:
   P2, P3, P6 by 0.93–1.07 and P4, P5 by 0.985–1.015 (before: one factor per coil group, ±3 % and ±1 %). This gives
   up–down asymmetry and a wider range of triangularity. Z_axis now spans −5.0 to +5.0 cm (5–95 %), against ±3.5 cm.
   The factors are in `synth_meta.jitter.cols`.
3. **Thomson structure the equilibrium does not imply.** Drawn per case (in `synth_meta.thomson_var`) and applied
   on top of the realistic Thomson of the earlier sets:
   - a pedestal factor of ±30 % on Te and on ne beyond a flux position of 0.82–0.95;
   - a hollow Te core in 30 % of cases (dip of 5–30 %);
   - an in/out tilt of ±10 % across the plasma;
   - a mis-calibrated block of 3–10 adjacent channels (factor 0.75–1.25 on Te, ne or both) in half of the cases;
   - a lag of 0–6 ms between the Thomson profile and the equilibrium.

Unchanged from `gen4_set5000`:
- the plasma model (Thomson-derived pressure, pressure level set by the Thomson edge, current shape independent of
  Thomson, solenoid ×0.5);
- the realistic channel noise and missing channels;
- 3 to 9 labelled frames per case (6.0 on average; 976 training cases have all 9);
- the file layout;
- the sampling (15 strata of Ip × dIp/dt at their share of real frames). Every stratum met its quota, including the
  ramp-up stratum that fell short in `gen4_set5000`. Each real frame could be used twice with different draws.

Not in v3: an explicit outward force (vessel currents or a vertical-field offset) that would let the pressure ratio
take physical values. The total-to-electron axis pressure ratio is still 1.32 / 3.28 / 5.0 (5 / 50 / 95 %).

## Checks (inputs only; real = the origin shot's Thomson at the same times)

| | train | real MAST |
|---|---|---|
| Thomson Te peak, eV (5 / 50 / 95 %) | 315 / 628 / 1220 | 367 / 670 / 1250 |
| Thomson ne peak, 1e19 m⁻³ | 1.50 / 3.00 / 5.16 | 1.66 / 3.21 / 5.80 |
| missing channels per slice | 7.1 % | 7.1 % |
| slice-to-slice jitter of the p_e peak radius, cm (median / 90 %) | 2.6 / 7.4 | 2.6 / 14.4 |
| slice-to-slice relative jitter of the mean p_e (median) | 3.1 % | 2.9 % |
| case-centre Ip, kA (5 / 50 / 95 %) | 416 / 641 / 868 | 401 / 632 / 862 |
| outer boundary − real Thomson 30 eV edge, cm (5 / 50 / 95 %) | −10.0 / −1.0 / 2.9 | |

Labels (train, 5 / 50 / 95 %): q95 4.6 / 6.7 / 14.4, βN 0.48 / 1.61 / 3.97, li 0.64 / 0.88 / 1.27, κ 1.59 / 1.82 / 2.10,
R_axis 0.82 / 0.91 / 0.98 m, Z_axis −0.05 / 0.00 / 0.05 m, volume 6.1 / 7.7 / 8.9 m³.

The q95 tail is longer than in the earlier sets (95th percentile 14.4 against 9.8; maximum 36) because limited plasmas
and asymmetric shapes are now included. Filter or cap q95 if your pipeline is sensitive to it. Rank correlation of li
with the synthetic Te peaking: 0.20 (0.24 in `gen4_set5000`).

All 5,640 files load into rectangular arrays, with no NaN or Inf in ψ or the scalar labels, and every labelled frame
lies inside the stored ±100 ms of inputs.

Compute: 20,420 candidates, 466.9 CPU-hours of solver time (FreeGSNKE 3.1.1, single-threaded workers on AWS g7e8 nodes).

## Known limits

- **βN and pressure-dependent labels.** The pressure ratio (median 3.3) stands in for an outward force the machine
  model lacks (no vessel currents), so βN and other pressure-dependent labels lean high. See `gen4_set5000/CAVEATS.md`.
- **Machine model.** The limiter is drawn by hand from the coil positions (`mast_machine.py`, `_LIMITER_UPPER`,
  mirrored), using no EFIT data, and is unchanged since the first set.
- **Thomson structure.** The structure variations are drawn per case, not per slice. Within a case they act like a
  persistent calibration or profile feature; the channel noise still varies from slice to slice.

Rules position: only MAST inputs (magnetics currents, Thomson, coil and chord geometry, `efit_times`, `efit_grid_R/Z`) are
read. No MAST label, demo shot, envelope mask or FAIR-MAST data is used, and the cases are not fitted to magnetic
measurements.
