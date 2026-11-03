"""Score a directory of held-out predictions with the competition scorer (same split/targets as holdout.py).

    python score_holdout.py --preds runs/free_v1/preds --name nn_free_v1 [--float16]

Targets come from the cache (psi, q95, betaN per held-out shot). Results JSON goes to
my_experiments/c1/results/<timestamp>_holdout10_seed0_<name>.json in holdout.py's format, so the
numbers are directly comparable with the table in HANDOFF.md section 5.
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pickle  # noqa: E402

import evaluate as ev  # noqa: E402
from evaluate import Scorer, report  # noqa: E402
import local_score as ls  # noqa: E402

REFS_CACHE = Path(f"{FEQ_CACHE}/nn/holdout_refs.pkl")


class CachedScorer(Scorer):
    """evaluate.Scorer whose reference contours / derived scalars are cached on disk (built once)."""

    def __init__(self, rows, pool, cache_path: Path | None):
        self.pool = pool
        self.gts = [{"psi": ev.target_psi(r).astype(np.float32),
                     "q95": np.asarray(r["efit_q95"], dtype=np.float64),
                     "betaN": np.asarray(r["efit_beta_n"], dtype=np.float64)} for r in rows]
        stems = [r["_file"] for r in rows]
        refs = None
        if cache_path is not None and cache_path.exists():
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if cached["stems"] == stems:
                refs = cached["refs"]
                print(f"reference targets loaded from {cache_path}", flush=True)
        if refs is None:
            refs = list(pool.map(ev._reference, [g["psi"] for g in self.gts]))
            if cache_path is not None:
                tmp = cache_path.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump({"stems": stems, "refs": refs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache_path)
                print(f"reference targets cached -> {cache_path}", flush=True)
        self.refs = refs
        ps = pss = pn = 0.0
        ssum, ssq, sn = (np.zeros(ls.N_SCALARS) for _ in range(3))
        csum, csq, cn = (np.zeros(ls.N_CONS) for _ in range(3))
        for g, ref in zip(self.gts, self.refs):
            p = g["psi"].astype(np.float64)
            p = p[np.isfinite(p)]
            ps += p.sum(); pss += (p ** 2).sum(); pn += p.size
            for j, name in enumerate(ls.SCALARS):
                v = g[name][np.isfinite(g[name])]
                ssum[j] += v.sum(); ssq[j] += (v ** 2).sum(); sn[j] += v.size
            for j in range(ls.N_CONS):
                v = ref[1][ref[2][:, j], j]
                csum[j] += v.sum(); csq[j] += (v ** 2).sum(); cn[j] += v.size
        self.ref_stats = {"psi_sum": ps, "psi_sumsq": pss, "psi_n": pn,
                          "scal_sum": ssum, "scal_sumsq": ssq, "scal_n": sn,
                          "cons_sum": csum, "cons_sumsq": csq, "cons_n": cn}
        self.means = (ps / pn, np.where(sn > 0, ssum / np.maximum(sn, 1), 0.0),
                      np.where(cn > 0, csum / np.maximum(cn, 1), 0.0))

DATA_DIR = HERE.parents[2] / "challenge_data" / "data" / "diii_d_train"


def split_stems(data_dir: Path, frac: float, seed: int, max_shots: int):
    stems = [f.stem for f in sorted(data_dir.glob("*.parquet"))[: max_shots or None]]
    perm = np.random.RandomState(seed).permutation(len(stems))
    hold = set(perm[: int(round(frac * len(stems)))].tolist())
    return [s for i, s in enumerate(stems) if i in hold]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", type=Path, default=Path(f"{FEQ_CACHE}/nn/diii_d_train"))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--stems-file", type=Path, default=None, help="score these shots instead of the holdout split")
    ap.add_argument("--refs-name", default=None, help="name for the cached references when --stems-file is used")
    ap.add_argument("--quiet-save", action="store_true", help="write the result JSON next to the preds, not in results/")
    ap.add_argument("--float16", action="store_true", help="round psi through float16 (as the skeleton submits)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    stems = (args.stems_file.read_text().split() if args.stems_file
             else split_stems(args.data_dir, args.holdout, args.seed, args.max_shots))
    t0 = time.time()
    gts, preds = [], []
    for s in stems:
        d = np.load(args.cache / f"{s}.npz")
        gts.append({"_file": s, "efit_times": d["t"], "efit_psirz": d["psi"].astype(np.float32),
                    "efit_q95": d["q95"].astype(np.float64), "efit_beta_n": d["betaN"].astype(np.float64)})
        p = np.load(args.preds / f"{s}.npz")
        psi = p["psi"].astype(np.float16).astype(np.float32) if args.float16 else p["psi"].astype(np.float32)
        preds.append({"psirz": psi, "q95": p["q95"].astype(np.float32), "betaN": p["betaN"].astype(np.float32)})
    print(f"{len(stems)} held-out shots, {sum(len(g['efit_times']) for g in gts):,} frames loaded in "
          f"{time.time() - t0:.0f}s; float16={args.float16}", flush=True)

    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        t0 = time.time()
        tag_split = f"holdout{int(round(args.holdout * 100))}_seed{args.seed}"
        refs_path = REFS_CACHE.with_name(f"holdout_refs_{args.refs_name or tag_split}.pkl")
        scorer = CachedScorer(gts, pool, refs_path if not args.max_shots else None)
        print(f"references built in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        res = scorer(preds)
        print(f"scored in {time.time() - t0:.0f}s", flush=True)
    report(args.name, res, gts)
    out_dir = HERE.parent / "results"
    tag = f"holdout{int(round(args.holdout * 100))}_seed{args.seed}" + (f"_first{args.max_shots}" if args.max_shots else "")
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_{args.name}{'_f16' if args.float16 else ''}.json"
    if args.quiet_save:
        out = args.preds.parent / f"score_{args.name}.json"
    meta = {"preds": str(args.preds), "holdout": args.holdout, "seed": args.seed, "n_holdout": len(stems),
            "float16": args.float16, "holdout_shots": stems}
    out.write_text(json.dumps({"meta": meta, "results": {args.name: res}}, indent=1, default=float))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
