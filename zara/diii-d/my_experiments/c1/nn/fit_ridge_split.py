"""Fit the C1 physics + Ridge stage on the training side of a given split and export it for the GPU trainer.

    python fit_ridge_split.py --seed 42 --workers 4
→ ../models/holdout10_seed42_c1_field_shaf_eddy.pkl and $FEQ_CACHE/nn/ridge_holdout_s42.npz
"""
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from holdout import split  # noqa: E402
from model import ALPHAS, Z_CLIP  # noqa: E402
from train_parallel import fit_parallel  # noqa: E402

DATA = HERE.parents[2] / "challenge_data" / "data" / "diii_d_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--data-dir", type=Path, default=DATA)
    a = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    files = [str(f) for f in sorted(a.data_dir.glob("*.parquet"))]
    train, hold = split(files, a.holdout, a.seed)
    print(f"seed {a.seed}: {len(train)} train / {len(hold)} held-out shots; {a.workers} workers", flush=True)
    t0 = time.time()
    m = fit_parallel(train, a.workers, physics=("field", "shaf", "eddy"))
    tag = f"holdout{int(round(a.holdout * 100))}_seed{a.seed}"
    pkl = HERE.parent / "models" / f"{tag}_c1_field_shaf_eddy.pkl"
    m.save(pkl)
    npz = Path(f"{FEQ_CACHE}/nn/ridge_holdout_s{a.seed}.npz")
    np.savez(npz, theta=np.asarray(m.theta), mu=np.asarray(m.mu), sd=np.asarray(m.sd), W=np.asarray(m.W), b=np.asarray(m.b),
             Ws=np.asarray(m.Ws), bs=np.asarray(m.bs), alpha_psi=m.alpha_psi, alpha_scal=m.alpha_scal, z_clip=Z_CLIP,
             names=np.asarray(list(map(str, m.names))), physics=np.asarray(list(m.physics_opts)))
    print(f"fit {time.time() - t0:.0f}s: theta={np.round(m.theta, 4)} alpha_psi={m.alpha_psi:g} -> {pkl.name}, {npz}")



if __name__ == "__main__":
    main()
