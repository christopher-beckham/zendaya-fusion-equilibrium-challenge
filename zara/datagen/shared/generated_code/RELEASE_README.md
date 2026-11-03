# Synthetic equilibrium test sets

Four sets of 500 synthetic plasma equilibria for **testing** how a DIII-D-trained model handles machines and conditions it has not seen. Use them to evaluate trained models only, not to train them: the labels come from the FreeGSNKE free-boundary solver rather than from EFIT, and the DIII-D set is built from the inputs of the 704 held-out validation shots.

| Folder | Machine | What it tests |
|---|---|---|
| `mast/` | MAST | Transfer to the challenge's test machine. Built from real `mast_public_test` input frames |
| `d3d/` | DIII-D | Calibration: how closely synthetic scores track real ones. Built from the held-out validation shots |
| `sparc/` | SPARC | Transfer to a compact, high-field conventional tokamak |
| `iter/` | ITER | Transfer to a large conventional tokamak |

Explorer: https://claude.ai/artifact/GCh8nzUPbPP183dgwWBMmz

## Files

Each folder contains:
- `parquet/<machine>_synth_XXXXX.parquet`: one equilibrium per file (`XXXXX` is its Sobol design index)
- `<machine>_synthetic_500.npz`: the same 500 equilibria in one file (`psi`, `grid_R/Z`, `lcfs_r/z`, `inputs` + `input_names`, `labels` + `label_names`, Sobol parameters; MAST, SPARC and ITER also carry the Thomson arrays)
- `design_summary.json`: design size and outcome counts
- SPARC and ITER: `<machine>_limiter_mask_65x65.npz`, the plasma-region mask for scoring
- DIII-D: `validation_vs_efit.json`, the generator check against real EFIT (below)

At the top level: `score_folder.py` (scoring) and `audit_report.json` (quality checks).

## Format

**MAST and DIII-D files use exactly the challenge's columns and types** (checked column by column against real challenge rows), so code written for the challenge data reads them unchanged, including `data_fixes.align_mast_thomson_core` and `data_fixes.fix_d3d_ip_times`. Each file holds one EFIT time with ±100 ms of input signals around it.

| | MAST | DIII-D | SPARC | ITER |
|---|---|---|---|---|
| Flux `efit_psirz` | (1, 65, 65), (nZ, nR) | same | same | same |
| Flux sign | axis at the maximum (as MAST) | axis at the minimum (as DIII-D EFIT), for both current directions | axis at the maximum | axis at the maximum |
| Grid R, Z (m) | 0.06–2.0, ±2.0 | 0.84–2.54, ±1.6 | 1.1–2.7, ±1.7 | 3.2–8.8, ±5.0 |
| Coil columns | challenge columns, kA·turn (P-coils), kA (solenoid, TF) | challenge columns, kA·turn (F-coils), kA (E-coil, B-coil) | one per circuit, kA·turn | one per circuit, kA·turn |
| Time series | real source-shot signals, ±100 ms | real source-shot signals, ±100 ms | single time point | single time point |
| Thomson | real MAST chord positions (empty channel 0 kept) | the source shot's core and tangential chords | 110 core + 16 edge midplane chords | 110 core + 16 edge midplane chords |

Details:
- **Labels:** `efit_q95`, `efit_beta_n`, `efit_li`, `efit_r_axis`, `efit_z_axis`, `efit_lcfs_*` in every set; MAST also has `efit_beta_p`, `efit_elongation`, `efit_psi_axis`, `efit_psi_boundary` and `efit_geom_axis_*` as in its schema. Extra `synth_*` columns record the generator parameters, source shot and time (MAST, DIII-D), and further labels (volume, minor radius, β_t).
- **Boundaries:** in the parquet files `efit_lcfs_r/z` hold exactly `efit_lcfs_n` points (closed, first point = last). In the `.npz` files outlines are padded with NaN; real DIII-D parquet pads with 0.
- **SPARC and ITER coils:** each `coil_*` row is one conductor element and carries `coil_polarity × column / n_elements` (polarity is −1 only for SPARC's lower VS1 coil). `magnetics_tf_current` is the total TF current in kA·turn, so R·B_tor = μ0·I_tf/(2π).
- **DIII-D Ip** is signed: 34 of the 500 frames come from reversed-current shots, as in real DIII-D.
- `magnetics_dsep` is present for schema compatibility and is NaN throughout.

## Scoring

`score_folder.py` computes the competition composite S (0.55 R²ψ + 0.15 R²{q95, βN} + 0.10 (1 − D_LCFS) + 0.20 Consistency) with the challenge's own scoring code and the right plasma-region mask. It needs the challenge starter kit at `/fsx/zara.ercan/fusion-equilibrium-challenge-starter`.

```bash
python score_folder.py --data mast/parquet --machine mast --pred my_preds.npz
python score_folder.py --data sparc/parquet --machine sparc --mode perfect   # self-check: S = 1
```

Predictions go in an `.npz` with `<stem>/psirz` (T, 65, 65), `<stem>/q95` and `<stem>/betaN` (T,) for each parquet file stem. MAST and DIII-D use the scorer's own masks; SPARC and ITER need their limiter masks, because SPARC's divertor coils sit inside the grid and would otherwise attract the scorer's axis search.

## How close to reality

**DIII-D, where real labels exist.** Synthetic DIII-D starts from real held-out DIII-D input frames. Solved without any fitting, its flux maps match the challenge's EFIT maps with a median R² of 0.96 (10th–90th percentile 0.84–0.99, 165 frames including reversed-current shots), and its plasmas fall within the real range: medians q95 4.4 vs 5.0, κ 1.73 vs 1.81, li 1.08 vs 0.97, Ip 1.0 MA in both. Shapes are somewhat more spread out than in the real data.

**MAST, inputs only.** The synthetic inputs span the real `mast_public_test` inputs; their frequencies are not matched. Share of real frames inside the synthetic range: TF current, Thomson peak Te and Ip 99.9–100%; P3–P6 coils 98.7–99.8%; Thomson peak ne 99.2%; P2 coils 98.6–98.9% (some real shots run P2 reversed below −91 kA·turn); solenoid 97.5% (real reaches +15 kA, synthetic +10.6 kA).

**Calibration recipe.** Score models trained without the 704 held-out shots on both the real held-out DIII-D shots and `d3d/`. If model variants rank the same way on both, rankings on the other synthetic sets are worth trusting. S(synthetic MAST) / S(synthetic DIII-D) estimates cross-machine transfer, since generator artifacts largely cancel in the ratio.

## Label ranges (median, 5th–95th percentile)

| | q95 | β_N | β_p | li | κ | a (m) | Ip (MA) |
|---|---|---|---|---|---|---|---|
| MAST | 6.3 (4.4–10.5) | 1.5 (0.19–4.8) | 0.36 (0.04–0.90) | 1.02 (0.74–1.45) | 1.79 (1.51–2.15) | 0.56 (0.48–0.64) | 0.61 (0.38–0.84) |
| DIII-D | 4.4 (2.6–7.1) | 0.96 (0.07–2.6) | 0.52 (0.04–1.44) | 1.08 (0.81–1.59) | 1.73 (1.45–2.08) | 0.58 (0.50–0.65) | 1.0 (0.65–1.51) |
| SPARC | 2.7 (1.9–3.8) | 1.15 (0.36–2.5) | 0.61 (0.19–1.12) | 1.14 (0.84–1.68) | 1.89 (1.62–2.20) | 0.45 (0.38–0.52) | 4.6 (3.2–7.9) |
| ITER | 2.8 (2.0–3.9) | 0.99 (0.35–2.1) | 0.40 (0.14–0.93) | 1.10 (0.82–1.49) | 1.67 (1.54–1.82) | 1.89 (1.78–1.99) | 9.6 (6.9–13.2) |

## How the sets were made

All four come from static free-boundary solves with FreeGSNKE 3.1.1, sampled with scrambled Sobol designs of 4,096 points. A point is kept only if it converges to a diverted plasma inside the machine wall with plausible labels, if re-solving it from its stored inputs reproduces its flux (R² ≥ 0.9999), and, for SPARC and ITER, if no coil circuit exceeds 100 MA·turn. Each set holds the 500 accepted points with the lowest Sobol index. Plasma boundaries come from the challenge scorer's own extraction, and every boundary-dependent label is computed from that boundary.

- **MAST:** machine built from the challenge's 812 coil conductors, with a MAST-like limiter drawn from the coil positions. Each point starts from a real `mast_public_test` input frame and scales the coil groups, Ip and TF current; the solenoid is solved at half the column value (the FAIR-MAST level), while the columns keep the challenge's convention.
- **DIII-D:** machine built from the challenge's 19 coil rectangles (E-coil with 152.8 effective turns, B-coil with 144 turns) and the scorer's DIII-D envelope as the wall. Same recipe as MAST, from the 704 held-out shots (500 frames from 327 shots).
- **SPARC:** eight double-null base equilibria with realistic coil currents, each Sobol point scaling every coil circuit by ±15% and setting Ip, field and profiles.
- **ITER:** each Sobol point sets a lower-single-null shape, Ip, field and profiles; FreeGSNKE finds the coil currents.
- **Thomson:** density follows a parametric profile (MAST and DIII-D anchored to the real Thomson level at that time; SPARC and ITER to a Greenwald fraction), and Te comes from the solved pressure.

## Limitations

- **Labels are FreeGSNKE's, not EFIT's** (median R² 0.96 against EFIT on DIII-D). Compare models with these sets rather than reading absolute scores as leaderboard predictions.
- **Thomson Te runs hotter than real plasmas.** Peak core Te reaches 8.6 keV at the 95th percentile on synthetic DIII-D (real 3.8 keV) and up to 11 keV on synthetic MAST (real below 1.5 keV), because pressure comes from force balance while density is anchored to real values. Models that use Te see some out-of-range inputs.
- **No vessel eddy currents, sensor drift or Thomson dropouts**, and one Thomson profile exactly at the EFIT time.
- **SPARC is double null only**, and SPARC and ITER have a single time point per file, so time-derivative features have nothing to work with.
- The MAST limiter is drawn from coil positions rather than a measured first wall, and many synthetic MAST plasmas sit close to its outer edge.
- ITER coils are single filaments in FreeGSNKE's description; the VS3 coils sit inside the grid.

## Quality checks

`audit_report.json` records 49 checks, all passing: schema against real challenge rows; array shapes; no NaN or infinite values in flux, labels or Thomson; flux sign convention in every frame; flux at the axis and boundary equal to the labels; every boundary closed and enclosing the axis; plausible labels; Thomson hot inside the plasma and cold outside; unique, distinct equilibria; parquet and `.npz` agreement; re-solving 25 equilibria per set from the parquet columns alone reproduces the stored flux (R² = 1.0000); and the C1 DIII-D input loader runs on the synthetic rows. The scorer's own scalar derivations agree with the labels (κ within 0.04% at the median).

## Data provenance (for the methods report)

> Synthetic test sets, used only to evaluate and compare trained models, never for training. MAST: 500 synthetic equilibria generated with FreeGSNKE 3.1.1 on a machine description built from the challenge-supplied MAST coil geometry, with a hand-drawn MAST-like limiter; coil currents, Ip and TF current were drawn from the mast_public_test input signals with Sobol-sampled scalings; no MAST EFIT reconstructions were used. DIII-D: 500 synthetic equilibria with the same recipe, built from held-out training shots, used to calibrate the synthetic scores. SPARC and ITER: 500 synthetic equilibria each, generated with FreeGSNKE 3.1.1 on its SPARC and ITER machine descriptions. An AI coding assistant (Claude Code) wrote the generation code.

The generator code and intermediate files are in `/fsx/zara.ercan/more_data/generated/`.
