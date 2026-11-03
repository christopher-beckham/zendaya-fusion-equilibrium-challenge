# Tests of two properties of the synthetic MAST smoke-test set (`gen4_set500v2`)

Code: `my_experiments/m2/gen4/caveats.py` (jobs `slurm/caveats.sbatch`, `slurm/caveats_gen.sbatch`). Results:
`/fsx/zara.ercan/more_data/m2/gen4_caveats/` (`res/`, `res_b/`, one JSON per sampled frame), logs
`logs/caveats_stats.log`, `logs/caveats_gen_stats.log`. Only our synthetic data and real MAST inputs are used; no MAST label
of any kind. 320 frames sampled at random (one per case) from the 500 training cases.

## 1. Why li correlates with the peaking of the synthetic Te

In the set, li and the peaking of the synthetic Te profile (97th percentile / median of channels above 30 eV) have a rank
correlation of 0.26 over 3,016 frames, although the current-shape function FF′ does not depend on Thomson.

**Statistics over all frames:**

| Rank correlation of li with Te peaking | Value |
|---|---|
| plain | 0.26 |
| controlling for the drawn current-shape parameters (am, an) | 0.31 |
| controlling for am, an, κ and Ip | 0.27 |
| controlling for am, an and the electron-pressure peaking | −0.03 |
| frames where the pressure term carries a low / middle / high share of Ip (median 17 % / 30 % / 47 %) | 0.10 / 0.16 / 0.31 |

**Counterfactual on the sampled frames:**
- *Reproduction.* Each frame is re-solved as generated. Against the label, li differs by −0.04 to +0.07 (5–95 %, median
  0.000) and βN by −21 % to +7 % (median 0.0 %).
- *Fixed pressure shape.* The same frame is solved with the Thomson-derived pressure shape replaced by one fixed shape;
  axis pressure, current shape and coils are unchanged. 259 of 283 frames solved.
  - li vs Te peaking: 0.30 as generated, 0.10 with the fixed pressure shape.
  - li vs electron-pressure peaking: 0.26 as generated, −0.04 with the fixed pressure shape.
  - The change in li correlates with the pressure peaking at −0.75.

**Conclusion.** The correlation runs through the pressure-driven part of the toroidal current (R p′). Its shape follows
the Thomson pressure, so a more peaked pressure gives a more peaked current and a higher li. This is a term of the
Grad–Shafranov equation, not a link between Thomson and the drawn current shape. Its strength scales with the share of the
current that the pressure carries, which depends on the total pressure in the labels (section 2).

## 2. The high total-to-electron pressure ratio and βN

In the set, putting the outer boundary at the real Thomson edge needs a total / electron axis pressure ratio of median
3.4 (at least 5 % of frames at the cap of 5.0). On the generated maps, the volume-averaged ratio ⟨p⟩/⟨p_e⟩ is
1.57 / 2.73 / 3.56 / 4.60 / 5.12 (5 / 25 / 50 / 75 / 95 %). An independent input-only estimate gives 2.9.

**Test.** Each sampled frame was solved from scratch with the generator's own start path (same per-case draws and inputs,
same Thomson edge target), once with the solenoid at ×0.5 of the column (the set's convention) and once at ×1.0.

| | ×0.5 (control) | ×1.0 |
|---|---|---|
| frames accepted (of 320) | 258 | 127 (132 not converged) |
| pressure ratio, 5 / 25 / 50 / 75 / 95 % | 1.58 / 2.39 / 3.42 / 4.52 / 5.0 | 1.20 / 1.47 / 2.09 / 2.83 / 3.73 |
| same frames in the set's labels | 1.50 / 2.40 / 3.32 / 4.32 / 5.0 | 2.28 / 3.07 / 3.79 / 4.83 / 5.0 |
| βN / label βN, 5 / 50 / 95 % | 0.91 / 1.00 / 1.46 | 0.32 / 0.56 / 0.97 |
| li, median (labels) | 0.91 (0.92) | 0.87 (0.85) |
| q95, median | 6.47 | 4.87 |
| κ, median | 1.86 | 1.89 |

On the 96 frames accepted at both factors, the ratio ×1.0 / ×0.5 has a median of 0.55 for the pressure ratio and 0.52
for βN.

A first version of the test, ramping the solenoid with the profile held fixed, solved only 36 frames. It gave the same
direction: pressure ratio median 2.72, βN / label βN 0.77. With the profile unchanged, the outer boundary moved outward by
a median of 9 cm.

**Conclusion.** The high pressure ratio is mostly a consequence of the ×0.5 solenoid convention. At ×1.0, the same frames
reach the Thomson edge with about half the pressure. βN then falls to about half and q95 by about a quarter; li and κ
barely change.
- If ×1.0 described real MAST, the set's βN labels would be about 2× too high and q95 about 30 % too high.
- If ×0.5 is right, the labels stand, and the extra pressure stands in for whatever pushes real MAST plasmas outward that
  the model lacks. Candidates are vessel currents and a coil calibration offset; a separate analysis finds a steady
  outward vertical-field correction of about 3 mT.

These data cannot decide between the conventions; the leaderboard is the only label-based check. FreeGSNKE converges far
less often at ×1.0 (40 % against 81 % of frames).
