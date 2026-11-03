"""Hold-out evaluation on the full DIII-D train split, scored with the competition scorer.

    python my_experiments/c1/holdout.py --data-dir .../diii_d_train --workers 16 \
        --variants c1,c1+field,c1+shaf,c1+eddy,c1+field+shaf+eddy

A fixed random 10% of the train shots (seed 0: 704 of 7,041) is held out. Every variant is
trained on the other 90% with the multi-process trainer (train_parallel.fit_parallel, streaming,
flat memory), then predicts the held-out shots, and all variants are scored together on the same
held-out set by evaluate.Scorer (pooled R2 and the rest of S, as on the leaderboard).

Only the held-out targets (~3 GB for 704 shots) and one variant's predictions are held in
memory; shot files are re-read from disk. The public test splits carry no targets, so they
cannot be scored here.

Outputs: results/<timestamp>_holdout10_seed0.json (scores + the split, rewritten after each
variant) and models/holdout10_seed0_<variant>.pkl.
"""
from __future__ import annotations

import argparse
import json
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
from evaluate import DEFAULT_DATA, Scorer, report, summary  # noqa: E402
from model import C1Model, target_psi  # noqa: E402
from train_parallel import _chunks, fit_parallel  # noqa: E402


def split(files: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    perm = np.random.RandomState(seed).permutation(len(files))
    n_hold = int(round(frac * len(files)))
    hold = set(perm[:n_hold].tolist())
    return ([f for i, f in enumerate(files) if i not in hold],
            [f for i, f in enumerate(files) if i in hold])


def _targets(files: list[str]) -> list[dict]:
    out = []
    for f in files:
        row = pd.read_parquet(f, columns=["efit_psirz", "efit_q95", "efit_beta_n",
                                          "efit_times"]).iloc[0].to_dict()
        out.append({"_file": Path(f).stem, "efit_times": np.asarray(row["efit_times"]),
                    "efit_psirz": target_psi(row).astype(np.float32),
                    "efit_q95": np.asarray(row["efit_q95"], dtype=np.float64),
                    "efit_beta_n": np.asarray(row["efit_beta_n"], dtype=np.float64)})
    return out


def _predict(model_path: str, files: list[str]) -> list[dict]:
    m = C1Model.load(Path(model_path))
    preds = []
    for f in files:
        row = pd.read_parquet(f).iloc[0].to_dict()
        for k in [k for k in row if k.startswith("efit_") and k not in ("efit_times", "efit_grid_R",
                                                                         "efit_grid_Z")]:
            del row[k]                                  # predict from allowed inputs only
        row.pop("magnetics_dsep", None)
        row.pop("magnetics_dsep_times", None)
        preds.append(m.predict(row))
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--variants", default="c1",
                    help="comma-separated: c1, c1+<option>[+<option>] (field, shaf, eddy)")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shots", type=int, default=0, help="use only the first N files (testing)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    files = [str(f) for f in sorted(args.data_dir.glob("*.parquet"))[: args.max_shots or None]]
    train, hold = split(files, args.holdout, args.seed)
    print(f"{len(files)} shots: training on {len(train)}, holding out {len(hold)} "
          f"(seed {args.seed}), {args.workers} workers", flush=True)
    models_dir = HERE / "models"
    models_dir.mkdir(exist_ok=True)
    tag = f"holdout{int(round(args.holdout * 100))}_seed{args.seed}" + \
          (f"_first{args.max_shots}" if args.max_shots else "")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.json"
    meta = {"data_dir": str(args.data_dir), "holdout": args.holdout, "seed": args.seed,
            "n_train": len(train), "n_holdout": len(hold),
            "holdout_shots": [Path(f).stem for f in hold]}

    def save():                 # rewritten after every variant, so a requeue loses nothing scored
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=float))

    results = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        t0 = time.time()
        chunks = _chunks(hold, args.workers * 4)
        gts = [g for part in pool.map(_targets, chunks) for g in part]
        scorer = Scorer(gts, pool)
        print(f"held-out targets and references: {len(gts)} shots, "
              f"{sum(len(g['efit_times']) for g in gts):,} frames, {time.time() - t0:.0f}s", flush=True)

        for name in args.variants.split(","):
            opts = name.split("+")[1:] if name.startswith("c1+") else []
            if name != "c1" and not name.startswith("c1+"):
                raise SystemExit(f"unknown variant {name!r}")
            print(f"\n[{name}] training on {len(train)} shots", flush=True)
            t0 = time.time()
            model = fit_parallel(train, args.workers, physics=opts)
            path = models_dir / f"{tag}_{name.replace('+', '_')}.pkl"
            model.save(path)
            t_fit = time.time() - t0
            t0 = time.time()
            preds = [p for part in pool.map(_predict, [str(path)] * len(chunks), chunks) for p in part]
            t_pred = time.time() - t0
            t0 = time.time()
            results[name] = scorer(preds)
            print(f"[{name}] fit {t_fit:.0f}s, predict {t_pred:.0f}s, "
                  f"scoring {time.time() - t0:.0f}s -> {path.name}", flush=True)
            report(name, results[name], gts)
            save()
            del preds

    if len(results) > 1:
        summary(results, gts)
    save()
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
