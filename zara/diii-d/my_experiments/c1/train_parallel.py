"""Multi-process version of train.py: same C1Model, same result up to float rounding.

    python my_experiments/c1/train_parallel.py --data-dir /path/to/data/diii_d_train --workers 16
    python my_experiments/c1/train_parallel.py --max-shots 20 --compare-serial    # equivalence check

C1Model.fit makes two passes over the shots, and each pass only accumulates sums (X^T X, X^T Y,
feature moments) that add up in any order. Here each worker takes a slice of the shots, builds
its own partial sums, and the main process adds them. Fold membership for the Ridge CV is the
shot's position in sorted file order modulo n_folds, exactly as in the serial fit.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import DEFAULT_DATA  # noqa: E402
from model import ENVELOPE, SCALAR_KEYS, C1Model, _RidgeSums, target_psi  # noqa: E402


def _read(f: str) -> dict | None:
    row = pd.read_parquet(f).iloc[0].to_dict()
    return row if row.get("source", "DIII-D") == "DIII-D" else None


def _pass1(files: list[str], physics=()):
    """Physics least-squares sums + feature moments over a slice of shots."""
    m = C1Model(physics=physics)
    ata, aty, nx, sx, sxx, keep, names = np.zeros((3, 3)), np.zeros(3), 0, 0, 0, [], None
    for f in files:
        row = _read(f)
        if row is None:
            continue
        inp = m._inputs(row)
        psi = target_psi(row)
        fixed, free = m._physics_terms(row, inp)
        A = free[:, :, ~ENVELOPE].transpose(0, 2, 1).reshape(-1, 3)
        y = (psi - fixed)[:, ~ENVELOPE].ravel()
        ata += A.T @ A
        aty += A.T @ y
        X = inp["X"]
        ok = np.isfinite(X)
        nx = nx + ok.sum(0)
        sx = sx + np.where(ok, X, 0).sum(0)
        sxx = sxx + np.where(ok, X * X, 0).sum(0)
        names = inp["names"]
        keep.append(f)
    return ata, aty, nx, sx, sxx, keep, names


def _pass2(model: C1Model, items: list[tuple[int, str]], n_folds: int):
    """Ridge sums (psi residual and scalars) over a slice of (shot index, file) pairs."""
    F, P = len(model.mu), ENVELOPE.size
    psi_sums = _RidgeSums(F, P, n_folds)
    scal_sums = _RidgeSums(F, len(SCALAR_KEYS), n_folds)
    for s, f in items:
        row = _read(f)
        inp = model._inputs(row)
        psi = target_psi(row)
        z = model._z(inp["X"])
        resid = (psi - model.physics(row, inp)).reshape(len(z), -1)
        psi_sums.add(s % n_folds, z, resid)
        S = np.column_stack([np.asarray(row[k], dtype=np.float64) for _, k in SCALAR_KEYS])
        ok = np.isfinite(S).all(1)
        scal_sums.add(s % n_folds, z[ok], S[ok])
    return psi_sums, scal_sums


def _add(total: _RidgeSums, part: _RidgeSums):
    for k in ("n", "sz", "szz", "sy", "szy", "syy"):
        setattr(total, k, getattr(total, k) + getattr(part, k))


def _chunks(seq: list, n: int) -> list[list]:
    """n contiguous slices (contiguous keeps each worker's reads sequential in file order)."""
    idx = np.array_split(np.arange(len(seq)), n)
    return [[seq[i] for i in ix] for ix in idx if len(ix)]


def fit_parallel(files: list[str], workers: int, verbose: bool = True, physics=()) -> C1Model:
    model = C1Model(physics=physics)
    n_chunks = max(1, min(len(files), workers * 4))
    # spawn, not fork: forking after pyarrow/BLAS have started their thread pools deadlocks
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        t0 = time.time()
        chunks = _chunks(files, n_chunks)
        parts = list(pool.map(_pass1, chunks, [model.physics_opts] * len(chunks)))
        ata = sum(p[0] for p in parts)
        aty = sum(p[1] for p in parts)
        nx = sum(p[2] for p in parts)
        sx = sum(p[3] for p in parts)
        sxx = sum(p[4] for p in parts)
        kept = [f for p in parts for f in p[5]]          # chunks are in order, so this is sorted
        model.names = next(p[6] for p in parts if p[6] is not None)
        model.theta = np.linalg.solve(ata, aty)
        model.mu = sx / np.maximum(nx, 1)
        model.sd = np.sqrt(np.maximum(sxx / np.maximum(nx, 1) - model.mu ** 2, 0))
        model.sd[model.sd < 1e-9] = np.inf
        if verbose:
            print(f"  pass 1 ({time.time() - t0:.0f}s): a_E={model.theta[0]:.4g} "
                  f"a_p={model.theta[1]:.4g} c={model.theta[2]:.4g}  ({len(kept)} shots)", flush=True)

        t0 = time.time()
        n_folds = min(model.n_folds, len(kept))
        F, P = len(model.mu), ENVELOPE.size
        psi_sums, scal_sums = _RidgeSums(F, P, n_folds), _RidgeSums(F, len(SCALAR_KEYS), n_folds)
        items = list(enumerate(kept))
        futs = [pool.submit(_pass2, model, ch, n_folds) for ch in _chunks(items, n_chunks)]
        for i, fut in enumerate(futs, 1):
            p_psi, p_scal = fut.result()
            _add(psi_sums, p_psi)
            _add(scal_sums, p_scal)
            if verbose and (i % max(1, len(futs) // 10) == 0 or i == len(futs)):
                print(f"  pass 2: {i}/{len(futs)} chunks, {time.time() - t0:.0f}s", flush=True)

    model.alpha_psi, _ = psi_sums.select_alpha(model.alphas)
    model.W, model.b = psi_sums.fit(model.alpha_psi)
    model.alpha_scal, _ = scal_sums.select_alpha(model.alphas)
    model.Ws, model.bs = scal_sums.fit(model.alpha_scal)
    if verbose:
        print(f"  ridge: alpha_psi={model.alpha_psi:g}  alpha_scalars={model.alpha_scal:g}  "
              f"({F} features, {n_folds} CV folds)", flush=True)
    return model


def compare(a: C1Model, b: C1Model) -> float:
    worst = 0.0
    for k in ("theta", "mu", "W", "b", "Ws", "bs"):
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        fin = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(x[fin] - y[fin]).max() / max(np.abs(x[fin]).max(), 1e-300)
        worst = max(worst, rel)
        print(f"  {k:<6s} max relative difference {rel:.2e}")
    for k in ("alpha_psi", "alpha_scal"):
        print(f"  {k:<10s} serial {getattr(a, k):g}  parallel {getattr(b, k):g}")
        if getattr(a, k) != getattr(b, k):
            worst = np.inf
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--physics", default="",
                    help="comma-separated physics_features options: field, shaf, eddy")
    ap.add_argument("--compare-serial", action="store_true",
                    help="also run the serial C1Model.fit and report the differences")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")               # inherited by the spawned workers
    files = [str(f) for f in sorted(args.data_dir.glob("*.parquet"))[: args.max_shots or None]]
    print(f"training on {len(files)} shots from {args.data_dir} with {args.workers} workers", flush=True)
    t0 = time.time()
    physics = tuple(o for o in args.physics.split(",") if o)
    model = fit_parallel(files, args.workers, physics=physics)
    print(f"parallel fit: {time.time() - t0:.0f}s", flush=True)
    if args.out:
        model.save(args.out)
        print(f"saved -> {args.out}")

    if args.compare_serial:
        def rows():
            for f in files:
                row = _read(f)
                if row is not None:
                    yield row
        t0 = time.time()
        serial = C1Model(physics=physics).fit(rows, verbose=False)
        print(f"serial fit: {time.time() - t0:.0f}s")
        worst = compare(serial, model)
        ok = worst < 1e-8
        print("EQUIVALENT" if ok else "MISMATCH", f"(worst relative difference {worst:.2e})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
