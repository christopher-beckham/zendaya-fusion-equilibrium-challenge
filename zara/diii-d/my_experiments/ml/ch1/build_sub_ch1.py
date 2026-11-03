"""Build a LOCAL submission directory: DIII-D from per-shot test predictions, MAST copied byte for byte from an existing
submission file (so only the DIII-D part changes), then validate both files. Never uploads, writes no manifest.

    python build_sub_ch1.py --preds <dir with d3d_shot_*.npz (psi, q95, betaN, t)> \
        --mast-npz ../../../submission/nn_prod_v23_u4pp/mast_public_test.npz --out ../../../submission/nn_prod_v5_ch1
DIII-D shots are written in sorted local parquet order (= Hugging Face stream order) by c1/nn/make_submission_nn.build_local
(imported read-only), psi float32.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "my_experiments" / "c1" / "nn"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--mast-npz", type=Path, default=None, help="MAST file to copy; omit together with --d3d-only")
    ap.add_argument("--d3d-only", action="store_true", help="build and validate the DIII-D file only (no MAST file)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-validate", action="store_true", help="dry run: skip validate_submission (code-path test only)")
    ap.add_argument("--config", default="diii_d_public_test",
                    help="(package) DIII-D config: diii_d_public_test or diii_d_private_test (file <config>.npz, shots in the sorted "
                         "parquet order of challenge_data/data/<config>)")
    a = ap.parse_args()
    if any(x.startswith("--repo") for x in sys.argv[1:]):
        raise SystemExit("never uploads")
    if not a.d3d_only and a.mast_npz is None:
        raise SystemExit("--mast-npz is required unless --d3d-only")
    import make_submission_nn as ms
    a.out.mkdir(parents=True, exist_ok=True)
    ms.build_local(a.config, "DIII-D", a.preds, a.out, np.float32)
    if not a.d3d_only:
        dst = a.out / "mast_public_test.npz"
        shutil.copyfile(a.mast_npz, dst)
        s0, s1 = sha(a.mast_npz), sha(dst)
        assert s0 == s1, "MAST copy differs"
        print(f"  mast_public_test: copied from {a.mast_npz} (sha256 {s1[:16]}..., identical)")
    # own checks: every public-test shot present (sorted local parquet order), psi (T, 65, 65) float32, all finite
    loc = ms.local_T(a.config)
    z = np.load(a.out / f"{a.config}.npz")
    n_psi = sum(1 for k in z.files if k.endswith("_psirz"))
    assert n_psi == len(loc), (n_psi, len(loc))
    n_frames = 0
    for i, (stem, T) in enumerate(loc):
        psi, q, b = z[f"shot_{i:04d}_psirz"], z[f"shot_{i:04d}_q95"], z[f"shot_{i:04d}_betaN"]
        assert psi.shape == (T, 65, 65) and psi.dtype == np.float32, (stem, psi.shape, psi.dtype)
        assert q.shape == (T,) and b.shape == (T,), stem
        assert np.isfinite(psi).all() and np.isfinite(q).all() and np.isfinite(b).all(), f"{stem}: non-finite"
        n_frames += T
    print(f"  own checks OK: {len(loc)} DIII-D public-test shots, {n_frames:,} frames, psi float32 (T, 65, 65), all finite")
    if a.no_validate:
        return 0
    from validate_submission import validate
    bad = 0
    for config in ((a.config,) if a.d3d_only else (a.config, a.config.replace("diii_d", "mast"))):
        r = validate(a.out / f"{config}.npz", config, 0)
        print(f"validate {config}: {'FAILED' if r else 'OK'}")
        bad += int(bool(r))
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
