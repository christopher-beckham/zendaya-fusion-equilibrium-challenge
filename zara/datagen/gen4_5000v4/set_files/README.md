# Synthetic MAST training set, recipe v4 (5,000 + 640 cases), calibrated on MAST inputs only

| Folder / file | Contents |
|---|---|
| `mast_g4s5000v4_train/` | 5,000 cases from training-pool shots, 34,276 labelled frames (2.4 GB) |
| `mast_g4s5000v4_heldout/` | 640 cases from held-out shots (the MASTVAL shot pool), 4,425 labelled frames (315 MB). Do not train on these. |
| `*/frames_metadata.parquet` | one row per labelled frame, 60 columns (see "Metadata") |
| `SHA256SUMS.txt` | checksums of the 5,640 case files and of every file in `code/` |
| `code/` | `gen4.py`, `gen4_set.py`, `pool.py` (generator), `mast_machine.py`, `synth_common.py` (machine and labels), `machine_MAST_challenge_bz/` (coil set used), `check.py`, `verify_set.py`, `export_meta.py`, `environment_freegsnke.yml`, `hardware.txt` |
| `raw/`, `done/`, `candidates.json`, `assembly.json` | every attempted candidate, its outcome, the sampling plan |

Generator version `gen4.10`, recipe `v4` (in every file's `synth_meta`). Seeds 6,000,000 + candidate id: no case repeats a
case of any earlier set. Held-out cases come only from held-out shots, and blend partners of held-out cases are
held-out shots too, so no held-out shot ever enters a training case.

Built to Chris's `FEEDBACK_FOR_ZARA_2026-10-13_1200.md` and `AUDIT_V3_AGAINST_REQUESTS.md`.

## What v4 changes

### 1. More distinct coil-current patterns

- **Blends of two real shots.** 70 % of cases (3,501 of 5,000 training cases) blend the coil currents of two shots:
  `alpha × own + (1 − alpha) × partner`, with alpha 0.25–0.75. The partner is a frame of another shot of the same
  split and stratum, with Ip within 10 %; its series is aligned at the case's centre time. The blend covers P2–P6 upper
  and lower and the solenoid. Ip, TF, EFPS and Thomson stay those of the case's own shot.
- **Wider, independent coil factors on every case,** stored per coil column in `synth_meta.jitter.cols`:
  - P2, P3, P6 upper and lower, independently: 0.85–1.15;
  - P4, P5: one common factor 0.94–1.06, times 0.985–1.015 per coil;
  - solenoid: 0.85–1.15.

The real-shot pool is unchanged (the challenge provides 1,206 MAST shots, and the earlier sets already use every
training-pool shot), so new coil-current patterns have to come from blends and wider factors.

### 2. Pressure ratio from physics; the missing outward force modelled explicitly

The machine model has no vessel currents, so the plasma needs an extra outward force to sit where the real Thomson
says it does. Earlier recipes put that force into the pressure: total-to-electron ratio at the axis, median 3.3. v4 adds
a hidden, nearly uniform vertical field from a Helmholtz coil pair outside the grid (radius 4 m, at Z = ±2 m; coil
`bz` in `code/machine_MAST_challenge_bz/`). Its current, not the pressure, is moved until the outer midplane boundary
meets the real Thomson edge.

- **Pressure.** It comes from a physical prior: Ti/Te 0.7–1.4, a fast-ion share of 0–35 %, and a Ti/Te profile shape
  ±30 %. The total-to-electron ratio at the axis is **1.83 / 2.48 / 3.35** (5 / 50 / 95 %).
- **Field offset.** The field the pair adds at the plasma, per frame in `dbz_mT` (`ibz` is the pair current in A), is
  −1.6 / +8.7 / +16.9 mT (5 / 50 / 95 %; allowed range −6.7 to +18 mT). The coil-current inputs do not include this
  field, just as real inputs do not include vessel currents. The label flux maps do include it, as an EFIT map includes
  the vessel currents.
- **Labels.** βN is now 0.44 / 1.23 / 3.05 (5 / 50 / 95 %), against a median of 1.6–1.8 in earlier sets.

### 3. More variety in the internal profiles

- **Current peaking.** FF′ ∝ (1 − x^am)^an with am 0.8–4.0 and an 0.3–5.0 (log-uniform); before, am 1–3 and an 0.5–3.5.
- **Edge current.** Half of the cases add an edge-current bump, a sin² shape of height 0.1–0.5 (relative to the core)
  between flux position 0.7–0.9 and the boundary.
- **Pressure peaking not seen by Thomson.** The fast-ion profile exponent ranges 1.5–6, plus the Ti/Te profile shape.
- **Profile groups.** 80 groups of 5 candidates share their frame, blend partner and coil factors, so their coil
  currents are identical; only the profile draws differ. 165 training cases in 62 groups were accepted (48 groups with
  2–5 cases: 15 pairs, 16 triples, 12 of 4, 5 of 5). Column `group` (−1 = none).

Resulting labels (train, 5 / 50 / 95 %): li 0.57 / 0.76 / 1.08 (earlier sets: median 0.88–0.96), κ 1.69 / 1.93 / 2.26
(earlier: median 1.82–1.85).

### 4. Metadata

`frames_metadata.parquet` (one per folder) has one row per labelled frame:
- origin file, centre time, frame time, split, stratum, seed, `group`;
- blend partner (`blend_file`, `blend_t`, `blend_alpha`) and every coil factor (`jit_*`);
- profile parameters (`tite`, `ff`, `nf`, `am`, `an`, `tite_d`, `eb`, `ex1`);
- pressure ratio `cp`, `paxis`, `pe_axis`, `ibz`, `dbz_mT`;
- `limited`, `inner_hot`, and the boundary-vs-Thomson diagnostics;
- the main labels, including upper and lower triangularity computed from the stored boundary.

The same values are in each file's `synth_meta` and `synth_frames`.

### 5. Triangularity

The wider P2/P3/P6 ranges did **not** widen the triangularity spread: upper 0.26 / 0.36 / 0.46 (sd 0.061), lower
0.27 / 0.36 / 0.45 (sd 0.056); v3 had sd 0.061 / 0.057. The share of frames with |upper − lower| > 0.1 is 12 %. A
stronger lever (drawn shape targets with a constrained solve) would be a separate change.

### Kept from v3

- **Limited plasmas.** Plasmas resting on the centre column are allowed on frames whose real Thomson is hot at its
  innermost channels. They are 8.5 % of training frames; v3 had 15.7 %, because the outward field now moves many
  plasmas off the column.
- **Thomson structure.** The per-case structure the equilibrium does not imply (pedestal, hollow core, tilt,
  mis-calibrated block, lag) is unchanged. It is not needed by itself, but it does no harm.
- **Everything else.** Realistic channel noise and missing channels, at least 3 labelled frames per case (6.9 on
  average; 1,660 training cases have all 9), stratified sampling. Every stratum met its quota.

## q95 of limited frames

q95 is FreeGSNKE's flux-surface q at normalised flux 0.95, where the boundary of a limited frame is the surface that
touches the centre column. In the v3 set, limited frames with q95 > 15 have:
- broad current (li 0.69, current peaking `an` 0.86, against 0.92 and 1.33 for limited frames with q95 < 10);
- lower Ip (571 against 718 kA);
- a slightly smaller plasma.

Their X-point lies further from the boundary than for low-q95 frames (normalised flux 1.12 against 1.01), so the tail
is not caused by a surface running close to an X-point. With a broad current a large share of Ip flows outside the 0.95
surface, which raises q there. In v4, q95 is 4.6 / 6.5 / 11.9 (5 / 50 / 95 %). Use the `limited` flag to down-weight
these frames.

## Checks (inputs only; real = the origin shot's Thomson at the same times)

| | train | real MAST |
|---|---|---|
| Thomson Te peak, eV (5 / 50 / 95 %) | 318 / 676 / 1280 | 367 / 717 / 1290 |
| Thomson ne peak, 1e19 m⁻³ | 1.38 / 3.03 / 5.29 | 1.51 / 3.21 / 5.72 |
| missing channels per slice | 7.2 % | 7.2 % |
| slice-to-slice jitter of the p_e peak radius, cm (median / 90 %) | 2.2 / 6.8 | 2.6 / 13.9 |
| slice-to-slice relative jitter of the mean p_e (median) | 3.0 % | 3.0 % |
| case-centre Ip, kA (5 / 50 / 95 %) | 417 / 639 / 871 | 401 / 632 / 862 |
| outer boundary − real Thomson 30 eV edge, cm (5 / 50 / 95 %) | −9.3 / −0.8 / 2.3 | |
| inner boundary − real Thomson 30 eV inner edge, cm | −7.7 / +0.6 / +8.9 | |

Labels (train, 5 / 50 / 95 %): q95 4.6 / 6.5 / 11.9, βN 0.44 / 1.23 / 3.05, li 0.57 / 0.76 / 1.08, κ 1.69 / 1.93 / 2.26,
R_axis 0.81 / 0.91 / 0.97 m, Z_axis −0.045 / 0.000 / 0.039 m, βp 0.11 / 0.28 / 0.61, volume 6.2 / 7.8 / 8.9 m³.

All 5,640 files load into rectangular arrays, with no NaN or Inf in ψ or the scalar labels, and every labelled frame
lies inside the stored ±100 ms of inputs.

Compute: 15,541 candidates, 221.5 CPU-hours of solver time. Hardware: AWS g7e8 nodes, Intel Xeon Platinum 8559C,
96 CPUs per node, single-threaded workers (`code/hardware.txt`).

## Limits

- **The hidden field is a model of missing physics, not a measurement.** Its size (median +8.7 mT) is set by matching
  the real Thomson edge with a physical pressure. An independent label-free estimate gave about +3 mT. If real MAST
  needs less outward force, part of it should come back as pressure, and βN would rise.
- **li and κ shifted** (li lower, κ higher) because of the wider current-profile prior and the outward field. Whether
  this is closer to real MAST cannot be checked without labels.
- **Machine model.** The limiter is still drawn by hand from the coil positions (`code/mast_machine.py`), unchanged since
  the first set.

Rules position: only MAST inputs (magnetics currents, Thomson, coil and chord geometry, `efit_times`, `efit_grid_R/Z`) are
read. No MAST label, demo shot, envelope mask or FAIR-MAST data is used, and the cases are not fitted to magnetic
measurements.
