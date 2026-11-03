"""Blend the per-model MAST prediction files of one split with fixed weights into the submitted MAST file.

    python blend/run_blend.py --weights blend/weights_chris8r1.json --members DIR --split mast_private_test --out OUT.npz

DIR holds one folder per model with that model's file, DIR/<model>/mast_<split>.npz (the layout chris/mast/predict.sh writes;
Zara's u4all / u4pp files go in the same layout). The weights file gives, per quantity (maps = psi, q95, betaN), the weight of
each model; blend.py normalises each set to sum 1 and blends frame by frame in float64. Chris's models write psi with the
DIII-D sign convention, so their psi is flipped (x -1) before blending; Zara's u4all / u4pp and r1 already use the MAST sign.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OURS = {"u4all", "u4pp", "r1"}         # MAST-sign files (Zara's); every other model is one of Chris's (DIII-D sign)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--weights", type=Path, required=True)
ap.add_argument("--members", type=Path, required=True)
ap.add_argument("--split", default="mast_private_test")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
w = json.loads(a.weights.read_text())
fmt = lambda d: ",".join(f"{k}={v}" for k, v in d.items() if v > 0)
names = sorted({k for q in ("maps", "q95", "betaN") for k, v in w[q].items() if v > 0})
files = []
for n in names:
    f = a.members / n / f"{a.split}.npz"
    assert f.exists(), f"missing {f}"
    files.append(f"{n}={f}" + ("" if n in OURS else "@-1"))
with np.load(a.members / names[0] / f"{a.split}.npz") as z:
    shots = sum(k.endswith("_psirz") for k in z.files)
cmd = [sys.executable, str(Path(__file__).with_name("blend.py")), "--out", str(a.out), "--psi", fmt(w["maps"]),
       "--q95", fmt(w["q95"]), "--betaN", fmt(w["betaN"]), "--shots", str(shots)] + files
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
