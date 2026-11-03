# Synthetic MAST set, recipe v7 (1,500 + 200 cases): v6 with a per-frame li taught by DIII-D

| Folder / file | Contents |
|---|---|
| `mast_g4s1500v7_train/` | 1,500 training cases |
| `mast_g4s1500v7_heldout/` | 200 held-out cases from held-out shots. Do not train on these. |
| `*/frames_metadata.parquet` | one row per labelled frame: everything v6 has, plus each frame's li target, the model's li* and log K*, the case's li residual and the current-shape exponent |
| `li_targets.npz`, `li_targets.json` | the li* table used (one row per real MAST frame) and its summary |
| `SHA256SUMS_mast_g4s1500v7.txt`, `SHA256SUMS_code.txt` | checksums of the data and of `code/` |
| `code/` | generator, li table builder and checks |

Built to `FOR_ZARA_TEAM_BLEND_2026-10-16.md`, section 3. Generator `gen4.11`, recipe `v7` (in every file's
`synth_meta`). Seeds 13,000,000 + candidate id: no case repeats a case of v6 (12,000,000 + id). MAST inputs only: no
MAST label, demo shot, envelope or FAIR-MAST data.

## What v7 is

Everything in v6 is kept: shared upper/lower factors for P2, P4 and P5, P4 factors centred on 1 in three equal-quota
bins, TF ± 0.3 %, coil blends of two shots, wall-edge labels on limited frames, the v6 Thomson model, the hidden field
moved per frame to put the outer midplane boundary at the real Thomson edge, and held-out cases spread over the
held-out shots as real test frames are. The pressure keeps the v4/v6 physical prior.

Three things change.

### 1. li follows the DIII-D-taught prediction for each frame

- **The model.** Chris's `cond_model.pkl` (`/fsx/chris.beckham/gsolve_data/tune2/`), gradient-boosted trees fitted on
  699 DIII-D training shots: li = f(z), where z are input-only features (shot phase from Ip, ramp rate over the
  current-diffusion time, q*, Greenwald fraction, electron βp, Te/ne/pe peaking and widths on a normalised radius).
  Held-out DIII-D R² 0.78, residual sd 0.13.
- **Applied to real MAST inputs.** `li_table.py` reads Chris's `mast/real_z.parquet` (z for every real MAST frame,
  computed from input columns by his `mast_z.py`) and the model, without copying or changing them, and writes li* per
  frame. The sha256 of every source file is in `li_targets.json`.
- **Smoothed in time.** li* is the median over ± 2 neighbouring EFIT frames (± 10 ms). Unsmoothed, li* jumps by
  0.11 (sd) between neighbouring frames, mostly from Thomson noise in z; smoothed, by 0.04.
- **Target per frame.** li_target = li*(frame) + e, with one residual e ~ N(0, 0.133) per case (the DIII-D fit's
  residual sd), clipped to 0.5–2.2. A case spans about 40 ms, much shorter than the current-diffusion time, so one
  residual per case keeps li smooth within a case while the spread across cases matches the model's uncertainty.
- **How it is met.** The v4 current shape h(x) = (1 − x^am)^an (+ an edge bump on half of the cases) is drawn as in
  v6, then raised to a power: h → h^pk. For the shape without a bump this is an → pk·an. Inside the frame solve, each
  round takes a secant step in log pk towards the target alongside the hidden-field step that matches the Thomson edge.
  The solve stops when the edge is within 1 cm, the Thomson tables have settled and li is within 0.03 of the target.
  Cold starts begin at pk = 1 (the drawn v6 shape); later frames of a case start from the previous frame's pk,
  corrected for the change in target.
- **Field and shape move together.** A more peaked current pushes the plasma outward. Each step in pk comes with a
  hidden-field step of −2.25 mT per expected +0.1 in li (the trade-off from the label-free force-balance check in
  `ZARA_NEXT_DATA_2026-10-15.md`). Without it, most solves above pk ≈ 1.5 did not converge in the smoke test.
- **The pressure ratio was not changed.** The model's log K* (median K* 1.95 on MAST) is stored per frame in the
  metadata (`lk_star`) but not used; the v6 pressure prior is kept, so this set changes li alone.

### 2. Hidden field −10 to +20 mT

With li near 1.1 the plasma needs less outward help to reach the Thomson edge. In the smoke test with the v6 range
(0 to 20 mT), many high-li frames sat at 0 mT with the boundary up to 12 cm outside the Thomson edge, and were rejected.
The range is therefore −10 to +20 mT, and each case's starting value (kept on frames without a Thomson edge) is drawn
from −5 to +15 mT, centred on the field the higher li leaves.

### 3. Larger solve budget

Moving the current shape costs solves. Start frames get 40 solves (v6: 20), later frames 26 (v6: 14), and up to 14
rounds (v6: 7).

## Results

| Folder | Cases | Labelled frames | Size |
|---|---|---|---|
| `mast_g4s1500v7_train` | 1,500 | 8,747 (5.8 per case) | 672 MB |
| `mast_g4s1500v7_heldout` | 200 | 1,150 | 90 MB |

Every stratum met its quota, and all 1,700 files load into rectangular arrays with no NaN or Inf in ψ or the scalar
labels.

### li against its target

5 / 25 / 50 / 75 / 95 % of the labelled frames:

| | Train | Held-out |
|---|---|---|
| li achieved (label) | 0.78 / 0.97 / 1.10 / 1.25 / 1.51 | 0.82 / 0.97 / 1.11 / 1.28 / 1.55 |
| li target | 0.81 / 1.01 / 1.14 / 1.28 / 1.53 | 0.86 / 1.01 / 1.16 / 1.32 / 1.57 |
| li* (model, no residual) | 0.94 / 1.07 / 1.15 / 1.27 / 1.48 | 0.94 / 1.08 / 1.16 / 1.29 / 1.57 |
| li − target | −0.29 / −0.02 / −0.00 / +0.01 / +0.05 | −0.29 / −0.03 / −0.00 / +0.01 / +0.04 |
| share within 0.03 / 0.05 / 0.10 of target | 71 % / 77 % / 85 % | 70 % / 76 % / 83 % |
| correlation of li with its target | 0.86 | 0.88 |
| exponent pk (5 / 50 / 95 %) | 0.86 / 2.7 / 11.2 | 0.90 / 2.7 / 11.6 |

- **Misses.** 15 % of frames end more than 0.10 from their target, nearly all below it. Their targets are high
  (median 1.22): the most peaked current shapes often fail to converge, and the frame keeps the closest state that
  did. Every frame stores `li_target` in the metadata, so frames can be filtered by `|li − li_target|` at training time
  if wanted (keeping |li − li_target| < 0.10 keeps 85 %).
- **Start and later frames.** Median |li − target| is 0.022 on start frames and 0.015 on later frames of a case.

### Labels and hidden parameters against v6

5 / 50 / 95 % of the training frames (sd in brackets):

| | v6 | v7 |
|---|---|---|
| li | 0.57 / 0.76 / 1.11 (0.17) | 0.78 / 1.10 / 1.51 (0.23) |
| κ | 1.67 / 1.93 / 2.26 (0.18) | 1.49 / 1.74 / 1.96 (0.14) |
| upper triangularity | 0.26 / 0.37 / 0.46 | 0.25 / 0.38 / 0.47 |
| lower triangularity | 0.27 / 0.36 / 0.45 | 0.26 / 0.36 / 0.45 |
| q95 | 4.66 / 6.50 / 9.81 | 4.58 / 6.13 / 8.72 |
| βN | 0.43 / 1.24 / 3.04 | 0.40 / 1.14 / 2.89 |
| βp | 0.11 / 0.29 / 0.63 | 0.09 / 0.25 / 0.55 |
| R_axis (m) | 0.813 / 0.910 / 0.969 | 0.806 / 0.898 / 0.953 |
| Z_axis (cm) | −2.6 / +0.1 / +2.2 (1.8) | −4.5 / +0.2 / +3.5 (2.8) |
| hidden field (mT) | 1.4 / 12.2 / 20 | −9.7 / 2.7 / 17.7 |
| total / electron pressure on axis | 1.81 / 2.45 / 3.38 | 1.80 / 2.38 / 3.32 |
| limited frames | 9.9 % | 17.8 % |

- **Elongation is lower.** With the same coil currents, a more peaked current gives a less elongated plasma (median
  κ 1.74 against 1.93). This follows from the solver's force balance; no setting targets it. It is the largest change
  in the shape labels and the one to watch on the board.
- **Hidden field.** 38 % of frames sit below 0 mT, 4.4 % at the −10 mT floor and 2.2 % at the 20 mT cap. The median
  (2.7 mT) is close to what the label-free estimate predicts: +10 to +13 mT at v4's li, minus about 2.3 mT per 0.1
  of extra li, so about 4 mT.
- **More limited frames** (18 % against 10 %), and a wider vertical-position spread (sd 2.8 cm against 1.8 cm).

### Coil inputs against real MAST

Unchanged from v6 by construction. Real MAST: the coil currents at every EFIT time of every `mast_public_test` shot
with |Ip| ≥ 150 kA. 5 / 50 / 95 % of the training frames:

| | v7 train | real MAST |
|---|---|---|
| P2 upper − lower, kA | −1.6 / +0.2 / +3.5 | −1.7 / 0.0 / +4.1 |
| P4 upper − lower, kA | −0.9 / −0.3 / +0.3 | −0.8 / −0.3 / +0.2 |
| P5 upper − lower, kA | −0.4 / +0.2 / +0.8 | −0.7 / +0.2 / +1.0 |
| P6 upper − lower, kA | −3.2 / −0.7 / +3.1 | −4.9 / −0.6 / +8.8 |
| P4 / P5 current ratio | 1.05 / 1.43 / 2.95 | 1.05 / 1.42 / 3.11 |
| P4 / Ip | −0.33 / −0.23 / −0.17 | −0.33 / −0.23 / −0.16 |
| TF share in −85.0 to −84.7 kA | 43 % | 65 % |

### Thomson against the origin shot at the same times

- Te peak 707 / 752 eV (median, synthetic / real); density peak 2.96 / 3.16 × 10¹⁹ m⁻³.
- Missing channels 7.4 % / 7.4 %.
- Slice-to-slice jitter of the mean electron pressure (median / 90th): 3.1 / 11.4 % against 3.1 / 11.4 %.
- Channels inside the 30 eV edges reading below 20 eV: 3.0 % synthetic against 0.8 % real (the v6 dropout model).

### Compute

- 6,996 candidates attempted, for 367 worker-hours of solver time and 402 allocated CPU-hours.
- AWS g7e8 spot nodes (Intel Xeon Platinum 8559C, 96 CPUs), single-threaded FreeGSNKE workers: three nodes for 64
  minutes and two more for the last 33 minutes.
- 27 candidates (0.4 %) stopped on a FreeGSNKE separatrix error inside the li check. As for any failed candidate,
  each was replaced by the next candidate of the same stratum, so the quotas are unaffected.
- The li* table took under a minute on 4 CPUs.

## Using the set

- Train and evaluate as with v6. File names are `mast_synth_g4v7_<candidate id>.parquet`.
- Metadata columns new in v7: `li_target`, `li_star`, `lk_star` (log K* of the model, not used), `li_e` (the case's
  residual) and `pk`. The `dbz_mT` column now goes below zero.
- The held-out folder comes from held-out shots only. Do not train on it.

## Regenerating this set

From `code/` (edit the repository path in the two batch scripts):

    python li_table.py <out dir>/li_targets.npz          # once; reads Chris's tune2 model and real_z table in place
    sbatch -p <partition> --cpus-per-task=94 --mem=400G run_big.sbatch <tag> 92 <out dir> 1500 200 mast_g4s1500v7 v7 \
        --n-real 2 --rng-base 13000000        # one or more jobs; they share the candidate list
    sbatch -p <partition> assemble_v7.sbatch <out dir> 1500 200 mast_g4s1500v7

The v7 settings are in `gen4_set.py` (`--recipe v7` block: hidden field −10 to +20 mT, starting field −5 to +15 mT,
budgets 40 / 26, 14 rounds) and `gen4.py` (`LI_TABLE`, `LI_RES_SD`, `LI_CLIP`, `LI_TOL`, `PK_RANGE`, `LI_SLOPE0`,
`DIBZ_DLI`; `li_star`, `quick_li` and the li step in `solve_frame_bz`). Label definitions are as in the v5 README.
