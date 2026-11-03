"""q profile + boundary features (ml/ch1s qcurve.py, imported read-only) of the out-of-fold psi maps of one fold, plus the fold
network's own q95 / betaN outputs.

    python qcurve_oof.py --fold oof0 [--workers N]
reads lists/<fold>_pred.txt and $FEQ_CACHE/ml/ch1q/oof/<fold>/preds -> $FEQ_CACHE/ml/ch1q/qc_<fold>.npz
(Q (N, 23), E (N, 11), S (N, 2) = fold-net q95 / betaN, levels, extra, stems)
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

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
from s_common import CACHE  # noqa: E402

OUT = Path(f"{FEQ_CACHE}/ml/ch1q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    stems = (HERE / "lists" / f"{a.fold}_pred.txt").read_text().split()
    pdir = OUT / "oof" / a.fold / "preds"
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(job, [(s, pdir, CACHE, False) for s in stems], chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    Q = np.concatenate([r[0] for r in res]); E = np.concatenate([r[1] for r in res])
    S = np.concatenate([np.column_stack([np.load(pdir / f"{s}.npz")["q95"], np.load(pdir / f"{s}.npz")["betaN"]]) for s in stems])
    assert len(S) == len(Q)
    out = OUT / f"qc_{a.fold}.npz"
    np.savez(out, Q=Q, E=E, S=S.astype(np.float32), levels=LEVELS, extra=np.array(EXTRA), stems=np.array(stems))
    print(f"{len(stems)} shots, {len(Q):,} frames, {time.time() - t0:.0f}s -> {out}; ok {np.nanmean(E[:, -1]):.4f}", flush=True)


if __name__ == "__main__":
    main()
