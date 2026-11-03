"""Write a MAST submission file that blends board-scored MAST files frame by frame.

    python blend/blend.py --out blend.npz --psi FC=0.5,r1=0.5 --q95 FC=1 --betaN FA=0.7,r1=0.3 --shots N \
        FA=<path>@-1 FC=<path>@-1 r1=<path>

Weights are normalised to sum 1 per quantity. A path ending in "@-1" has its psi sign flipped (DIII-D-convention MAST files). psi is blended in float64 and written in the first psi file's dtype.
"""
import argparse
import os

import numpy as np


def weights(s):
    w = {k: float(v) for k, v in (x.split("=") for x in s.split(",") if x)}
    t = sum(w.values())
    return {k: v / t for k, v in w.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--psi", required=True)
    ap.add_argument("--q95", required=True)
    ap.add_argument("--betaN", required=True)
    ap.add_argument("--shots", type=int, default=1206)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    paths, SIGN = {}, {}
    for x in a.files:
        n, q = x.split("=", 1)
        if q.endswith("@-1"):
            q, SIGN[n] = q[:-3], -1.0
        paths[n] = q
    W = {"psirz": weights(a.psi), "q95": weights(a.q95), "betaN": weights(a.betaN)}
    Z = {n: np.load(p) for n, p in paths.items() if any(n in w for w in W.values())}
    dt = np.float32
    out = {}
    for i in range(a.shots):
        for key, w in W.items():
            k = f"shot_{i:04d}_{key}"
            v = sum(wt * (SIGN.get(n, 1.0) if key == "psirz" else 1.0) * Z[n][k].astype(np.float64) for n, wt in w.items())
            out[k] = v.astype(dt if key == "psirz" else np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out + ".tmp.npz", **out)
    os.replace(a.out + ".tmp.npz", a.out)
    print(f"wrote {a.out}: {a.shots} shots, psi {W['psirz']}, q95 {W['q95']}, betaN {W['betaN']}, psi dtype {dt}")


if __name__ == "__main__":
    main()
