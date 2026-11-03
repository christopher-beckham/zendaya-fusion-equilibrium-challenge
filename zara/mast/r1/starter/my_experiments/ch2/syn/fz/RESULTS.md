# upegs results (2026-10-06, session zara-ercan-d9)

Scores: DIII-D = the fixed 704-shot holdout (10 %, seed 0) with the competition scorer. Synthetic sets = 500 FreeGSNKE equilibria
each (SPARC, ITER, MAST), scored with the official scorer code and an envelope mask built from the set's own LCFS polygons.
Unlabelled-machine rule for every model: learned harmonic (vessel/coil-error) correction OFF, 80 steps of label-free test-time
adaptation of the Z-even head components (Thomson flux-surface consistency, GS consistency, Ampere, pressure scale), pull-back prior
1 (p3 = prior 3). On DIII-D (labelled, learned coil gains): harmonics ON, no TTA.

| Run | Training | DIII-D S | R2psi | q95+betaN | 1-D | Cons | MAST synth | SPARC | ITER |
|---|---|---|---|---|---|---|---|---|---|
| pegs v3 | pegs, Kendall weights, edge loss | 0.9427 | 0.9967 | 0.986 | 0.969 | 0.749 | 0.742 | 0.861 | 0.732 |
| u1_phys | fixed weights, phys 0.3, 512x2, 4 steps | 0.9519 | 0.9984 | 0.987 | 0.972 | 0.788 | 0.751 | 0.807 | 0.732 |
| u1_nophys | no label-free losses | 0.9561 | 0.9983 | 0.987 | 0.973 | 0.809 | 0.716 | 0.851 | 0.751 |
| u1_harm | + direct harmonic supervision | 0.9488 | 0.9983 | 0.988 | 0.972 | 0.772 | 0.743 | 0.866 | 0.739 |
| u1_phys_ft | u1_phys + 8 ep fine-tune, func x4, band x3 | **0.9586** | 0.9982 | 0.986 | 0.973 | **0.822** | 0.742 | 0.809 | 0.735 |
| u2_multi | + JET/EAST/CMOD/KSTAR synthetic (20 % of steps) | 0.9467 | 0.9982 | 0.986 | 0.972 | 0.763 | 0.762 | 0.886 (0.943 no TTA) | 0.773 |
| u2_multi_ft | u2_multi + shape fine-tune | 0.9552 | 0.9980 | 0.985 | 0.974 | 0.806 | 0.753 | 0.883 (p3 0.898) | 0.771 |
| u3_multi | + NSTX-U 512 / NSTX 179 synthetic (25 %) | 0.9457 | 0.9981 | 0.985 | 0.971 | 0.759 | 0.767 (p3 **0.776**) | 0.883 (p3 0.897) | 0.772 |
| u3_multi_ft | u3_multi + shape fine-tune | 0.9534 | 0.9979 | 0.985 | 0.973 | 0.798 | 0.766 | 0.889 (p3 0.902) | 0.772 |
| **u4_all** | from scratch: 6 machines (25 %), func x3, band x2, 28 ep | 0.9578 | 0.9981 | 0.985 | 0.973 | 0.820 | **0.767** (p3 0.765) | 0.882 (p3 0.893, no TTA 0.946) | 0.771 (p3 0.775) |
| u6_wide | u4 recipe, trunk 768 x 3, func x4, band x3 | 0.9539 | 0.9978 | 0.981 | 0.973 | 0.803 | 0.767 | 0.895 (p3 0.910) | 0.755 |
| u5_all | u4 recipe, 32 ep, func x4, band x3, NSTX 262 | 0.9542 | 0.9979 | 0.984 | 0.974 | 0.802 | 0.760 (p3 0.762) | 0.877 (p3 0.892) | 0.766 |
| u7_fixed | u4 recipe, harmonics expanded about the machine centre (not the first-pass axis) | 0.9503 | 0.9978 | 0.968 | 0.972 | 0.796 | 0.749 | 0.873 (p3 0.889) | 0.774 |
| u8_2pass_ft | u4_all + 10 ep fine-tune with a second embedding pass from the model's own equilibrium | 0.9562 | 0.9979 | 0.982 | 0.971 | 0.815 (Z_axis 0.91, kappa 0.71) | 0.765 (p3 0.767) | 0.874 (p3 0.893) | 0.753 |

Reference bars (public leaderboard 2026-10-05): DIII-D S 0.9964 (heavy4evy); MAST S 0.9375 (dmitryzhevn); our pixel NN (DIII-D only) 0.9956.

## What moved the numbers

- R2psi 0.9967 -> 0.998 came from the wider trunk, 4 Picard steps, longer training and the fixed loss weights (the Kendall weights had
  let the GS-consistency term dominate). The edge loss (p', FF' -> 0 at the boundary) contradicts EFIT (p'(1)/p'(0) median 0.40) and was dropped.
- Consistency 0.75 -> 0.82 came from weighting the scorer-functional and LCFS-band losses (x4 / x3) in a short fine-tune.
- Zero-shot MAST 0.742 -> 0.776 came from (a) freezing the Z-odd heads during TTA (+0.08 on pegs v3), (b) physics losses (+0.035),
  (c) training on other machines' synthetic equilibria (+0.01 to +0.025, spherical tokamaks helping most), (d) TTA prior 3 (+0.01).
- Negative results: label-free fitting of the harmonics (free0 TTA) collapses the vertical position; inference-time vertical
  stabilisation is mixed; psi-averaging two runs gives nothing; direct supervision of the EFIT p'/FF' split is noise (EFIT's split
  varies 0.1-2.7x between frames); the harmonic-head aux loss did not improve DIII-D.

## Where the DIII-D gap is

Error budget on the holdout (u1_phys): psi residual rms 9.3 mWb (truth std 198); a per-frame fit of 8 vacuum harmonics still removes
~80 % of the residual sum of squares; axis error 16 mm rms with 5 % of frames above 30 mm. The remaining error is a smooth
low-order external field that the network does not predict precisely enough from the inputs; the pixel NN learns it to 0.99992 R2.

## Submission candidates (built with pegs/slurm/wrap_sub.sh, starter validator OK, bit-identical to the preds, NOT uploaded)

upegs_u1phys_rule (0.9519 / 0.751), upegs_u1physft_rule (0.9586 / 0.742), upegs_u2multi_rule (0.9467 / 0.762),
upegs_u2multift_rule (0.9552 / 0.753), upegs_u3multift_rule_p3 (0.9534 / 0.766, prior-3 TTA), upegs_u4all_rule_p3 (0.9578 / 0.765, best balance,
recommended first upload). All six validated as of 2026-10-06 13:45 UTC. Numbers = holdout S / synthetic MAST S.
