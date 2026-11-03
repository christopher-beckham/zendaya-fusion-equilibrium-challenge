"""Score a Final-format MAST file of the pooled synthetic split, one score per source set.

    python score.py ROOT PRED.npz OUT.json [--flip] [--workers 48]

PRED.npz is keyed shot_<i>_{psirz,q95,betaN} (the submission format), shot i being the i-th file of
ROOT/data/mast_private_test in sorted order. --flip: psi in the DIII-D sign (Chris's member files).
Scoring: Zara's score_folder.py (competition scoring code, MAST limiter mask) on ROOT/labels.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

SF = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4/score_folder.py"
ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path); ap.add_argument("pred", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--flip", action="store_true"); ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args()
stems = sorted(p.stem for p in (a.root / "data/mast_private_test").glob("*.parquet"))
z = np.load(a.pred)
assert sum(k.endswith("_psirz") for k in z.files) == len(stems), "shot count differs"
out, sign = {}, (-1.0 if a.flip else 1.0)
for i, s in enumerate(stems):
    k = f"shot_{i:04d}"
    T = len(pq.read_table(a.root / "labels" / f"{s}.parquet", columns=["efit_times"]).column(0)[0])
    assert z[f"{k}_psirz"].shape[0] == T, (k, s)
    out[f"{s}/psirz"] = sign * z[f"{k}_psirz"].astype(np.float32)
    out[f"{s}/q95"], out[f"{s}/betaN"] = z[f"{k}_q95"], z[f"{k}_betaN"]
tmp = a.out.with_suffix(".scorein.npz"); np.savez(tmp, **out)
subsets = ",".join(f"{p.stem}={p}" for p in sorted((a.root / "stems").glob("*.txt")))
sys.exit(subprocess.call([sys.executable, SF, "--data", str(a.root / "labels"), "--machine", "mastlim", "--pred", str(tmp),
                          "--subsets", subsets, "--workers", str(a.workers), "--ref-cache", str(a.root / "refs_mastlim.pkl"),
                          "--json", str(a.out)], cwd=Path(SF).parent))
