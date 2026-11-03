# rt_nonoise last-epoch spike: one DIII-D frame, no effect on MAST

Checked on the AWS cluster (NVIDIA RTX PRO 6000) with the checkpoints of this run that the A100 rebuild saved (all six
SHA256 OK, as listed in `RT_NONOISE_SPIKE.md`; the checkpoints themselves are not published). Zara's data mounted read-only; no labels of MAST used. `ep9` is the last epoch (the spike), `ep7`
and `ep5` healthy checkpoints of the same run.

## Cause

**A single DIII-D frame: `d3d_shot_04ac401f6f`, frame 256, near the end of the shot.** With the network's edge-field
(harmonic) correction switched on, as in the DIII-D validation during training, the ep9 weights get that frame's psi wrong
by 26,500 times the frame's own psi variance (ep7: 39 times). Its neighbours (frames 255, 257, 258) are already hard for
every checkpoint (24-45 times), so this is an awkward frame that the last epoch tips over.

| DIII-D hold-out, 704 shots, against the labels | ep9 | ep7 |
|---|---|---|
| Pooled R2psi, edge-field correction on | 0.98486 | 0.99821 |
| Pooled R2psi, edge-field correction off (`--no-harm`) | 0.99432 | 0.99429 |
| Frames where ep9's error is more than 100 times ep7's | 1 (the frame above) | |

The training-time validation (64 shots) is far more sensitive to that one frame: rerun on the ep9 weights (one epoch at
learning rate 0, then the normal validation pass) it gives R2psi 0.79971 on the RTX against -3.12 on the A100. The
blow-up is a property of these weights on this frame, and its size depends on the GPU.

**Not caused by the speed-ups.** The same validation rerun with Zara's original code and with the faster code gives
identical numbers (R2psi 0.79971, every printed value equal).

## MAST is not affected

r1 never uses the edge-field correction on MAST (trained with `--mast-harm-off`, predicted with `--no-harm`).

| Check | ep9 | ep7 | ep5 |
|---|---|---|---|
| Zara's internal validation (3,816 held-out FreeGSNKE cases with the challenge's own MAST public-test inputs and FreeGSNKE labels, raw S) | 0.8312 | 0.8318 | 0.8352 |
| Real MAST public test: frames needing the fallback | 0 | 0 | 0 |

Zara's own `rt_nonoise` scored 0.8286 on the internal validation, the RTX rebuild 0.8289.

Real MAST public test (1,206 shots, 69,075 frames, `predict.py` with r1's member flags `--sol-factor 0.5 --tag
_iv_full_wdiv --no-harm --betaN kinfill`, without the per-frame field correction), predictions compared with each other:

| Pair | psi difference per frame (RMS / frame std), median | 99th percentile | Pooled R2 psi | q95 | βN |
|---|---|---|---|---|---|
| ep9 vs ep7 | 0.0046 | 0.031 | 0.985 | 0.997 | 0.995 |
| ep5 vs ep7 (both healthy) | 0.0126 | 0.076 | 0.951 | 0.989 | 0.978 |

The spiked epoch is closer to ep7 than the two healthy checkpoints are to each other.

## Conclusion

The spike is one DIII-D frame in a part of the network that r1 does not use on MAST. An r1 built from this chain (with
the ep9 `rt_nonoise`) is fine for the MAST blend; no change to checkpoint selection is needed for r1.

## The A100 retrain of 30 October to 1 November

`train.sh B` on another A100 machine (`checks/a100-final/`) ended `rt_nonoise` on the same kind of last epoch: training-time
DIII-D validation R2psi 0.99722 at epoch 8 and -0.56723 at epoch 9, the epoch r1 uses (`checks/a100-final/r1_runs/rt_nonoise.log`).
On Zara's internal validation set that member scores S 0.8295, against 0.8293 for the shipped `rt_nonoise`, with no frame
needing r1's fallback (`checks/a100-final/r1_validation/`). The MAST predictions are again unaffected.

Scripts and outputs: `/fsx/chris.beckham/zara_speedup/out/spike/` (`d3d_compare.txt`, `d3d_harm_compare.txt`, logs) and
`/fsx/chris.beckham/zara_speedup/scripts/predict_*.sh`.
