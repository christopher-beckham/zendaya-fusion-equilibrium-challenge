"""Minimal de-intrusion: per frame, the smallest inboard (solenoid-shaped) correction b (kA column units, b <= 0) from a grid of
predictions whose scorer-style LCFS does not cross the centre column and whose inner midplane boundary is at R >= R_MIN.
Selection uses only the predictions (label-free). Writes an npz {<stem>__sol: (T,)} for predict.py --add-field npz:<file>.
    python select_b.py MAST <runs root> <grid name> <stems file> <out.npz> b0 b1 ...     (rep1.npy = phase_report --every 1)
    python select_b.py MASTVAL <runs root> <grid name> <truth.npz> <out.npz> b0 b1 ...  (intr_cost.npy; one frame per stem)"""
import sys
from pathlib import Path
import numpy as np
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

MODE, ROOT, G, SRC, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
BS = [float(b) for b in sys.argv[6:]]
BS = sorted(BS, key=lambda b: abs(b))                       # smallest correction first
R_MIN = 0.205
name = lambda b: f"{G}_b{('m' if b < 0 else '')}{abs(b):g}"
if MODE == "MAST":
    reps = {b: np.load(ROOT / name(b) / "rep1.npy") for b in BS}
    lim = np.stack([reps[b][:, 0] for b in BS], 1); rin = np.stack([reps[b][:, 1] for b in BS], 1)
    stems = sorted(SRC.read_text().split())
    T = [len(np.load(__import__("os").environ.get("MAST_SPLIT_CACHE", f"{R1_ROOT}/cache/mast/mast_public_test") + f"/{s}.npz")["t"]) for s in stems]
else:
    tr = np.load(SRC, allow_pickle=True); stems = [str(s) for s in tr["stems"]]
    have = [s for s in stems if (ROOT / name(BS[0]) / "preds" / f"{s}.npz").exists()]
    reps = {b: np.load(ROOT / name(b) / "intr_cost.npy") for b in BS}
    lim = np.stack([reps[b][:, 4] for b in BS], 1); rin = np.stack([reps[b][:, 2] for b in BS], 1)
    stems, T = have, [1] * len(have)
assert lim.shape[0] == sum(T), (lim.shape, sum(T))
good = (lim == 0) & (rin >= R_MIN)
first = np.where(good.any(1), good.argmax(1), len(BS) - 1)
b_sel = np.array(BS)[first]
b_sel = np.where(np.isfinite(lim[:, 0]), b_sel, 0.0)        # no LCFS at b = 0: leave the frame alone
out, o = {}, 0
for s, n in zip(stems, T):
    out[f"{s}__sol"] = b_sel[o:o + n].astype(np.float32); o += n
np.savez(OUT, **out)
moved = b_sel != 0
print(f"{MODE} {G}: {len(b_sel)} frames; intruding / R_in < {R_MIN} at b=0: {np.mean(~good[:, 0]):.1%}; corrected {moved.mean():.1%}; "
      f"unresolved (still crossing at the largest b) {np.mean(~good.any(1)):.2%}")
print(f"  b on corrected frames 10/50/90 %: {np.round(np.percentile(b_sel[moved], [10, 50, 90]), 1).tolist() if moved.any() else []}")
