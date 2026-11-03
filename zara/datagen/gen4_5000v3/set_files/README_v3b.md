# gen4_set5000v3b: v3 with q95 on the wall edge for limited frames

`mast_g4s5000v3b_train/` (5,000 cases) and `mast_g4s5000v3b_heldout/` (640 cases) are copies of the v3 folders. The one
change is `efit_q95` on frames where the plasma rests on the centre column. Checksums: `SHA256SUMS_mast_g4s5000v3b.txt`.

## What changed

- **q95 on limited frames** is now q at normalised flux 0.95 of the plasma's own boundary: the surface touching the wall.
  - Before, FreeGSNKE's safety-factor routine normalised the flux to the first X-point whenever one existed, and for a
    limited plasma that surface passes through the centre column.
  - It is recomputed from the stored ψ map, the stored axis and boundary flux, and the frame's stored current-profile
    parameters (no new solves). Code: `code/relabel_q95.py` in the v5 set folder (`gen4_set10000v5/code/`).
- **Everything else is unchanged.** The LCFS, volume, elongation, triangularity, βN, βp and li of v3 already used the
  wall-touching boundary: on limited frames the stored LCFS reaches R = 0.200 m. ψ is untouched.
- **New metadata.** Each frame's `synth_frames` entry has `boundary` (`limiter` or `x-point`) and `q95_xpt` (the old
  value). `frames_metadata.parquet` is new in each folder.

## Numbers

| | limited frames | q95 before (5 / 50 / 95 %) | q95 after | share above 12.44, before → after |
|---|---|---|---|---|
| train | 4,721 (15.7 %) | 6.01 / 11.3 / 22.9 | 5.62 / 7.53 / 10.3 | 41 % → 0.3 % |
| held-out | 540 (13.2 %) | 5.98 / 12.5 / 22.8 | 5.92 / 7.68 / 10.2 | 51 % → 0 % |

- **Unchanged frames.** On 6 training frames the q = 0.95 surface could not be traced; they keep the old value
  (`q95_xpt` equals `efit_q95` there).
- **Check of the method.** On diverted frames the recomputed q95 matches the stored one within 0.16 % (median).
