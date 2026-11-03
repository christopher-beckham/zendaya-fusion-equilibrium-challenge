# upegs: universal physics-embedded GS network (fork of pegs, 2026-10-05 23:10 UTC)

One model, one inference pipeline, for DIII-D (Challenge 1) and zero-shot on any other tokamak (Challenge 2).
Same structure as `../pegs` (exact coil flux + learned vacuum-harmonic correction + plasma flux from a profile network
through an unrolled free-boundary Grad-Shafranov layer; see `../pegs/README.md`). This copy exists so that the pegs
session and this one do not edit the same files; caches `cache/pegs/{diiid_*, mast_*, refs_d3d}` are shared read-only,
new caches go to `/fsx/zara.ercan/cache/upegs`.

## Changes relative to pegs (v3)

| Area | pegs v3 | upegs |
|---|---|---|
| `edge` loss (p', FF' -> 0 at psi_N = 1) | on, Kendall weight saturated (s = -8) | off by default. EFIT's DIII-D labels have p'(1)/p'(0) = 0.40 at the median (5-95 %: -2.2 to 2.9) and FF'(1)/FF'(0) = -0.005, so the loss contradicts the labels (xg labels_d3d, 64k frames) |
| `kin` loss | point-wise p(psi_N) = ratio x p_e | volume-average <p> = ratio <p_e> (pressure scale only; EFIT's p is not proportional to p_e). Old form available as `kinshape` |
| loss weighting | Kendall (learned), gs weight 6.7 x psi weight | `--weighting fixed` (supervised 1, label-free `--w-phys`) or Kendall with `--s-min` clamp |
| trunk | 256 x 2 | `--width --depth` (residual pre-norm blocks), `--prof-width`, `--harm-scale` (bound of the harmonic correction; v3: 0.02 mu0 Ip R0) |
| checkpoints | last only | last, best validation R2 psi, every `--save-every` epochs |
| TTA | harmonics from the net or off | `--harm-mode free0`: harmonics fitted label-free from zero (Thomson consistency + GS), same rule on every unlabelled machine |
| test sets | DIII-D holdout, synthetic MAST | + synthetic SPARC, ITER (xg/synthset_to_cache records), synthetic DIII-D (`D3DS`, its own FreeGSNKE E-coil convention) via `slurm/synth_eval.sh` |

## Error budget that motivates the work (pegs v3_nophys, 30 holdout shots, 2026-10-05)

psi residual rms 12.7 mWb/rad (psi std 198): inside the LCFS 16.7, outside 14.5; a per-frame fit of 8 vacuum harmonics
removes 92 % of the residual sum of squares. The remaining DIII-D error is therefore a smooth, low-order external
field (the 8 harmonic coefficients per frame), not fine plasma-current structure: the harmonic head is the lever on
DIII-D, while on an unlabelled machine the same head must stay off (or be fitted label-free).

## Running

```
slurm/run.sh <name> [train.py args]        # train (pinned to the running 8-GPU node) -> predict 704 holdout -> score
sbatch slurm/gpu_pin.sbatch bash slurm/synth_eval.sh runs/<name>     # SPARC / ITER / MAST synthetic zero-shot
```
