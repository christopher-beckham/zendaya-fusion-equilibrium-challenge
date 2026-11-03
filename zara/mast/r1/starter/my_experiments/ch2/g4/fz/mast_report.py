"""Label-free summary of a set of MAST predictions (no MAST equilibrium data of any kind is read).

    python mast_report.py runs/v1/mast [runs/v1/mast_tta ...]

Per prediction folder: distributions of the shape the scorer would extract from the predicted psi (axis, kappa,
triangularity, volume, li via the vendored scorer code with the MAST grid; envelope mask NOT used), q95 and betaN
ranges, the stored GS residual, and the share of frames whose LCFS extraction fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from derive import derive_frame          # noqa: E402

MAST_CACHE = Path(f"{R1_ROOT}/cache/mast/mast_public_test")


def summarise(run: Path, every=5, max_shots=400):
    g = np.load(MAST_CACHE / "basis.npz")
    R, Z = g["grid_R"].astype(float), g["grid_Z"].astype(float)
    files = sorted((run / "preds").glob("*.npz"))[:max_shots]
    vals = {k: [] for k in ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")}
    q, b, gsr, fail, n = [], [], [], 0, 0
    for f in files:
        d = np.load(f)
        q.append(d["q95"]); b.append(d["betaN"]); gsr.append(d["gs_res"] if "gs_res" in d.files else np.full(len(d["q95"]), np.nan))
        for k in range(0, len(d["psi"]), every):
            out = derive_frame(d["psi"][k], R, Z, "MAST")
            n += 1
            if not np.isfinite(out["kappa"]):
                fail += 1
            for key in vals:
                vals[key].append(out[key])
    pct = lambda v: [round(float(x), 3) for x in np.nanpercentile(np.asarray(v, float), [5, 25, 50, 75, 95])]
    rep = {"frames_sampled": n, "lcfs_fail_frac": fail / max(n, 1), **{k: pct(v) for k, v in vals.items()},
           "q95": pct(np.concatenate(q)), "betaN": pct(np.concatenate(b)), "gs_res": pct(np.concatenate(gsr))}
    return rep


if __name__ == "__main__":
    for r in sys.argv[1:]:
        rep = summarise(Path(r))
        print(f"== {r}  (5/25/50/75/95 percentiles)")
        for k, v in rep.items():
            print(f"   {k:15s} {v}")
        if "/ch2/syn/" in str(Path(r).resolve()):          # never write into other workstreams' folders
            (Path(r) / "mast_report.json").write_text(json.dumps(rep, indent=1))
