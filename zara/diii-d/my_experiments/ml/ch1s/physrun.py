"""Physics functionals (physfeat.py) of predicted and/or true flux maps for a list of shots.

    python physrun.py --split 0 --preds <dir with psi> --out <npz> [--true] [--workers 16]
    python physrun.py --stems train0 ...      (training shots of split 0; --stems all / test / train42 / hold0 ...)
Output npz: F (N, len(NAMES)) for the predicted maps (if --preds), Ft for the true maps (if --true), names, stems.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s_common import CACHE, TEST_CACHE, split, test_stems  # noqa: E402


def _job(a):
    from physfeat import shot_feats
    s, pdir, cache, true = a
    with np.load(cache / f"{s}.npz") as d:
        ip, bt = d["ip"].astype(np.float64), d["bt"].astype(np.float64)
        psi_t = d["psi"] if true else None
    out = {}
    if pdir is not None:
        with np.load(Path(pdir) / f"{s}.npz") as p:
            out["F"] = shot_feats((p["psi"].astype(np.float64), ip, bt))
    if true:
        out["Ft"] = shot_feats((psi_t.astype(np.float64), ip, bt))
    return out


def stems_of(spec):
    if spec == "test":
        return test_stems(), TEST_CACHE
    if spec == "all":
        return split(0, 0.0)[0], CACHE
    kind = "train" if spec.startswith("train") else "hold"
    assert spec.startswith(("train", "hold")), spec
    tr, ho = split(int(spec[len(kind):]))
    return (tr if kind == "train" else ho), CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="hold0")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--true", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    stems, cache = stems_of(a.stems)
    if a.limit:
        stems = stems[: a.limit]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(_job, [(s, a.preds, cache, a.true) for s in stems], chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(stems)} shots, {time.time() - t0:.0f}s", flush=True)
    from physfeat import NAMES
    out = {"names": np.array(NAMES), "stems": np.array(stems)}
    for k in ("F", "Ft"):
        if k in res[0]:
            out[k] = np.concatenate([r[k] for r in res])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{len(stems)} shots, {sum(len(r[next(iter(r))]) for r in res):,} frames, {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
