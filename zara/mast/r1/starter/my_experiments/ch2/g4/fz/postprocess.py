"""Apply predict.py's physical scalar bounds to an existing prediction folder -> a new folder (psi unchanged).

    python postprocess.py runs/v1/ho runs/v1/ho_b
"""
import shutil
import sys
from pathlib import Path

import numpy as np

from predict import bound, Q95_RANGE, BETAN_RANGE

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
(dst / "preds").mkdir(parents=True, exist_ok=True)
shutil.copy(src / "stems.txt", dst / "stems.txt")
nq = nb = 0
for f in sorted((src / "preds").glob("*.npz")):
    d = dict(np.load(f))
    d["q95"], a = bound(d["q95"], Q95_RANGE); d["betaN"], b = bound(d["betaN"], BETAN_RANGE)
    d["q95"], d["betaN"] = d["q95"].astype(np.float32), d["betaN"].astype(np.float32)
    nq += a; nb += b
    np.savez(dst / "preds" / f.name, **d)
print(f"{src} -> {dst}: replaced q95 {nq}, betaN {nb}")
