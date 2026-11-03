# Synthetic MAST set, recipe v6 (1,500 + 200 cases): v4 with coil inputs that look like real MAST

| Folder / file | Contents |
|---|---|
| `mast_g4s1500v6_train/` | 1,500 training cases |
| `mast_g4s1500v6_heldout/` | 200 held-out cases from held-out shots. Do not train on these. |
| `*/frames_metadata.parquet` | one row per labelled frame, including each case's coil-pair factors, P4 factor and hidden field |
| `SHA256SUMS_mast_g4s1500v6.txt` | checksums |
| `code/` | generator and checks, as for v5 |

Built to `ZARA_NEXT_DATA_2026-10-15.md`. Generator `gen4.11`, recipe `v6` (in every file's `synth_meta`). Seeds
12,000,000 + candidate id. MAST inputs only: no MAST label, demo shot, envelope or FAIR-MAST data.

## What v6 is

The v4 recipe: short cases (9 consecutive EFIT frames, at least 3 labelled), physical pressure prior, the v4
current-profile prior, coil blends of two shots (70 %), wall-edge labels on limited frames, and the hidden field moved to
put the outer boundary at the real Thomson edge. The changes:

### 1. Coil inputs (priority 1)

| | v6 | v4 | v5 |
|---|---|---|---|
| P2 | one factor per pair, 0.85–1.15 | upper and lower independent, 0.85–1.15 | same as v4 |
| P4 | one factor per pair, 0.85–1.20, in three equal-quota bins | common P4/P5 factor 0.94–1.06 | 0.6–1.4, P5 compensating |
| P5 | one factor per pair, 0.94–1.06 | common P4/P5 factor 0.94–1.06 | compensating for P4 |
| Upper vs lower within P2, P4, P5 | × 0.998–1.002 each (at most 0.4 % apart) | P2 independent; P4/P5 ± 1.5 % | ± 3 % |
| P3, P6 | upper and lower independent, 0.85–1.15 | same | same |
| Solenoid / Ip | 0.85–1.15 / 0.985–1.015 | same | same |
| TF | 0.997–1.003 (±0.3 %) | ±1.5 % | ±1.5 % |

- **P4 coverage.** Each candidate is assigned one P4 bin: 0.85–0.97, 0.97–1.08 or 1.08–1.20. Every stratum has equal
  quotas per bin, so the accepted P4 factors cover the range evenly even where acceptance depends on the factor.
- **Strata.** Ip quintile × dIp/dt tercile × P4 bin. Quotas follow the real share of frames in each Ip × dIp/dt
  stratum, for training and held-out alike, so the held-out cases are spread over the shots as real test frames are.
- **Upper/lower spread.** A smoke test with ±0.5 % within a pair made the P4 upper − lower spread about twice the real
  one, so it is ±0.2 %.

### 2. Hidden field 0 to 20 mT (priority 2)

- **Range.** The hidden vertical field (Helmholtz pair outside the grid, as in v4) is moved per frame within 0–20 mT,
  with no 9 mT cap and no pressure raise.
- **Edge matching kept.** As in v4, the field puts the outer midplane boundary at the real Thomson edge with the
  pressure at its physical prior. That is what made βN consistent with Thomson in v4.
- **Per-case draw.** Each case starts from a field drawn uniformly over 0–20 mT (`bz0_mT`). Frames without a Thomson
  edge keep that value.
- **Not done: a fixed field per case with no edge matching.** If that is what was meant, it is a one-line change.

### 3. Thomson (priority 3, optional)

- **Low-reading channels.** Per case, 2–4 % of channels inside the plasma read low: Te 1–20 eV, density × 0.1–1.
- **Te / density offset.** The density profile is mapped from a point shifted radially by N(0, 4 cm), clipped to ±8 cm,
  per case, which offsets the Te and density peaks.
- **Not included: per-slice profile wobble.** In the smoke test it doubled the slice-to-slice jitter of the mean
  electron pressure (6.9 % against 3.4 % real). The v4 noise model already matched that, and it is already drawn
  independently at each Thomson time.

## Why v5 cases with a high P4 factor failed

The P4 factor of every v5 candidate was recomputed from its seed (89,803 candidates):

| P4 factor | accepted | start frame solved | frame failures (largest) |
|---|---|---|---|
| 0.6–0.7 | 36 % | 53 % | not converged 15 %, edge miss 15 % |
| 0.9–1.0 | 11 % | 32 % | not converged 28 %, edge miss 21 % |
| 1.1–1.2 | 2.2 % | 17 % | not converged 42 %, edge miss 20 % |
| 1.3–1.4 | 1.4 % | 13 % | not converged 49 %, limited 14 % |

- **Main cause.** The failures are mainly solver non-convergence, rising with the P4 factor, plus more plasmas resting
  on the centre column.
- **Not the cap.** The share of solved frames at the 9 mT cap stays at 15–29 % in every bin. In v5, P5 was reduced to
  hold the field at R = 0.9 m, which moved the field curvature as well as the P4/P5 ratio.

## Results

| Folder | Cases | Labelled frames | Size |
|---|---|---|---|
| `mast_g4s1500v6_train` | 1,500 | 9,824 (6.5 per case) | 718 MB |
| `mast_g4s1500v6_heldout` | 200 | 1,362 | 99 MB |

Every stratum met its quota, and all 1,700 files load into rectangular arrays with no NaN or Inf in ψ or the scalar
labels.

### Coil inputs against real MAST

Real MAST: the coil currents at every EFIT time of every `mast_public_test` shot with |Ip| ≥ 150 kA. Figures are
5 / 50 / 95 % of the labelled frames.

| | v6 train | v6 held-out | real MAST |
|---|---|---|---|
| P2 upper − lower, kA | −1.6 / +0.2 / +3.4 | −1.6 / +0.1 / +3.3 | −1.7 / 0.0 / +4.1 |
| P4 upper − lower, kA | −0.9 / −0.2 / +0.3 | −0.9 / −0.3 / +0.2 | −0.8 / −0.3 / +0.2 |
| P5 upper − lower, kA | −0.5 / +0.2 / +0.8 | −0.4 / +0.2 / +0.7 | −0.7 / +0.2 / +1.0 |
| P3 upper − lower, kA | −0.4 / +0.1 / +0.7 | −0.4 / +0.1 / +0.7 | −0.6 / +0.1 / +0.8 |
| P6 upper − lower, kA | −3.7 / −0.7 / +5.6 | −3.4 / −0.6 / +7.1 | −4.9 / −0.6 / +8.8 |
| P4 / P5 current ratio | 1.09 / 1.43 / 2.39 | 1.16 / 1.44 / 2.63 | 1.05 / 1.42 / 3.11 |
| P4 / Ip | −0.30 / −0.23 / −0.17 | −0.31 / −0.23 / −0.17 | −0.33 / −0.23 / −0.16 |
| P5 / Ip | −0.19 / −0.16 / −0.12 | −0.19 / −0.15 / −0.11 | −0.20 / −0.15 / −0.09 |
| TF share in −85.0 to −84.7 kA | 43 % | 41 % | 65 % |
| P4 pair factor | 0.87 / 1.02 / 1.18 (mean 1.025) | 0.88 / 1.02 / 1.18 | |

- **v5 for comparison:** P2 upper − lower −31 to +27 kA; P4/P5 ratio median 0.89.
- **TF.** The ±0.3 % jitter still moves some frames out of the narrow −85.0 to −84.7 kA band. A smaller jitter would
  keep more of them in it.

### Hidden field, labels and Thomson

- **Hidden field per frame:** 1.4 / 12.2 / 20 mT. 6.6 % of frames sit at 20 mT and 2.7 % at 0. The median matches the
  +10 to +13 mT estimated from the real inputs.
- **Labels (train):**
  - q95 4.7 / 6.5 / 9.8; βN 0.43 / 1.24 / 3.04; li 0.57 / 0.76 / 1.11;
  - κ 1.67 / 1.93 / 2.26 (sd 0.18); upper / lower triangularity sd 0.061 / 0.057; Z_axis −2.6 / 0.0 / +2.2 cm;
  - limited frames 9.9 %.
- **Thomson against the origin shot at the same times:**
  - Te peak 703 / 751 eV (median, synthetic / real); density peak 3.03 / 3.21 × 10¹⁹ m⁻³;
  - missing channels 7.3 % / 7.3 %;
  - slice-to-slice jitter of the mean electron pressure (median / 90th): 3.1 / 10.8 % against 3.1 / 10.8 %.
- **Low-reading channels inside the plasma.** Measured as the share of channels inside the 30 eV edges that read below
  20 eV: 2.9 % synthetic against 0.8 % real. On your definition (3.5 % real against 0.8 % synthetic), the 2–4 % of
  dropouts added here should bring synthetic close to real. On the 30 eV-edge definition it overshoots.

### Compute

- 8,153 candidates attempted, for 115 worker-hours of solver time and 136 allocated CPU-hours.
- 3 AWS g7e8 spot nodes (Intel Xeon Platinum 8559C, 96 CPUs), single-threaded FreeGSNKE workers.
- About 35 minutes of wall time.

## Regenerating this set

From `code/` (edit the repository path in the two batch scripts):

    sbatch -p <partition> --cpus-per-task=94 --mem=400G run_big.sbatch <tag> 93 <out dir> 1500 200 mast_g4s1500v6 v6 \
        --n-real 2 --rng-base 12000000        # one or more jobs; they share the candidate list
    sbatch -p <partition> assemble_v6.sbatch <out dir> 1500 200 mast_g4s1500v6

The v6 settings are in `gen4_set.py` (`--recipe v6` block) and `gen4.py` (`P2_RANGE`, `P4_RANGE_V6`, `P5_RANGE`,
`PAIR_UL`, `TF_JIT`, `BZ_START_RANGE`, `THOMSON_V6`). Label definitions are as in the v5 README.
