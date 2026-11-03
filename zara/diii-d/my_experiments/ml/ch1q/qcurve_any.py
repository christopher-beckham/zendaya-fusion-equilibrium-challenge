"""q profile + boundary features (ml/ch1s qcurve.py `job`, imported read-only) of any per-shot map directory, e.g. the Group H
production maps of a new test fold.

    python qcurve_any.py --preds <dir of <stem>.npz with psi> --cache $FEQ_CACHE/nn/<split> --out <file.npz>
Stems = the sorted npz files of --preds (the export / fin_test order). Output layout as ml/ch1s qcurve.py (Q, E, levels, extra, stems).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1s"))
from qcurve import EXTRA, LEVELS, job  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    stems = sorted(p.stem for p in a.preds.glob("*.npz") if p.stem not in ("basis", "meta"))
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(job, [(s, a.preds, a.cache, False) for s in stems], chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    Q = np.concatenate([r[0] for r in res]); E = np.concatenate([r[1] for r in res])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, Q=Q, E=E, levels=LEVELS, extra=np.array(EXTRA), stems=np.array(stems))
    print(f"{len(stems)} shots, {len(Q):,} frames, {time.time() - t0:.0f}s -> {a.out}; ok {np.nanmean(E[:, -1]):.4f}", flush=True)


if __name__ == "__main__":
    main()
