# gen4_set5000v4b: v4 with q95 on the wall edge for limited frames

`mast_g4s5000v4b_train/` (5,000 cases) and `mast_g4s5000v4b_heldout/` (640 cases) are copies of the v4 folders. The one
change is `efit_q95` on frames where the plasma rests on the centre column. Checksums: `SHA256SUMS_mast_g4s5000v4b.txt`.

## What changed

- **q95 on limited frames** is now q at normalised flux 0.95 of the plasma's own boundary: the surface touching the wall.
  - Before, FreeGSNKE's safety-factor routine normalised the flux to the first X-point whenever one existed, and for a
    limited plasma that surface passes through the centre column.
  - It is recomputed from the stored ψ map, the stored axis and boundary flux, and the frame's stored current-profile
    parameters (no new solves). Code: `code/relabel_q95.py` in the v5 set folder (`gen4_set10000v5/code/`).
- **Everything else is unchanged.** The LCFS, volume, elongation, triangularity, βN, βp and li of v4 already used the
  wall-touching boundary: on limited frames the stored LCFS reaches R = 0.200 m, the wall. ψ is untouched.
- **New metadata.** Each frame's `synth_frames` entry has `boundary` (`limiter` or `x-point`) and `q95_xpt` (the old
  value). `frames_metadata.parquet` in each folder has both columns.

## Numbers

| | limited frames | q95 before (5 / 50 / 95 %) | q95 after | share above 12.44, before → after |
|---|---|---|---|---|
| train | 2,918 (8.5 %) | 6.98 / 11.9 / 22.4 | 5.82 / 8.07 / 11.8 | 46 % → 3.1 % |
| held-out | 354 (8.0 %) | 7.36 / 12.9 / 22.0 | 6.41 / 8.71 / 12.0 | 54 % → 3.4 % |

**Check of the method.** On 3,891 diverted frames, where the X-point is the boundary and the stored value is correct,
the recomputed q95 matches the stored one within +0.1 / +0.2 / +0.45 % (5 / 50 / 95 %).
