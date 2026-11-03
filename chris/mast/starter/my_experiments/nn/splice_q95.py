"""Candidate from two built submissions: flux maps and βN from MAP_SRC, q95 from Q95_SRC (both built by
build_candidate.sh / build_ensemble.sh with the same post-processing). The DIII-D file is MAP_SRC's.

Reason: shape losses (Z4) raised real-MAST Consistency but lengthened the submitted q95 tail; q95 from a model trained
without them keeps the tail of Z. Label-free: both sources are our own predictions.

    python splice_q95.py MAP_SRC Q95_SRC NEW_TAG        (submissions/<tag> folders; nothing is uploaded)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

S = Path("/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions")


def main() -> int:
    a, b, tag = sys.argv[1:4]
    za, zb = np.load(S / a / "mast_public_test.npz"), np.load(S / b / "mast_public_test.npz")
    assert za.files == zb.files
    arr = {k: np.asarray(zb[k] if k.endswith("_q95") else za[k], np.float32) for k in za.files}
    out = S / tag
    out.mkdir(exist_ok=True)
    np.savez(out / "mast_public_test.npz", **arr)
    d = out / "diii_d_public_test.npz"
    if d.exists():
        os.remove(d)
    os.link(S / a / "diii_d_public_test.npz", d)
    q = np.concatenate([arr[k] for k in arr if k.endswith("_q95")])
    print(f"{tag}: maps/βN from {a}, q95 from {b}; q95 > 12.44 on {100 * (q > 12.44).mean():.1f}% of frames; "
          f"dtypes {sorted({str(v.dtype) for v in arr.values()})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
