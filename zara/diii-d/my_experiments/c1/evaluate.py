"""Shot-grouped cross-validation of Challenge 1 models with the competition scorer.

    python my_experiments/c1/evaluate.py --data-dir /path/to/diii_d_train --models baseline,c1
    python my_experiments/c1/evaluate.py --folds 0          # leave-one-shot-out

Every shot is predicted by a model that never saw it, then all held-out predictions are scored
together (pooled R2, like the leaderboard) with local_score.py's functions, which wrap the
vendored `fusion_scoring/` code. All models see the same folds and the same reference targets.
Results are also written to my_experiments/c1/results/<timestamp>.json.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import local_score as ls  # noqa: E402
from model import C1Model, target_psi  # noqa: E402

DEFAULT_DATA = ROOT / "my_experiments" / "local_hf" / "data" / "diii_d_train"
_M = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GRID = (_M["grid_R"], _M["grid_Z"], _M["mask_coarse"].astype(bool),
        _M["mask_coarse"].astype(np.float64))


def make_model(name: str):
    if name == "baseline":
        from baseline import BaselineModel
        return BaselineModel()
    if name == "c1":
        return C1Model()
    if name.startswith("c1+"):                      # e.g. c1+field+shaf: physics_features options
        return C1Model(physics=name.split("+")[1:])
    raise SystemExit(f"unknown model {name!r}")


def load_rows(data_dir: Path, max_shots: int) -> list[dict]:
    files = sorted(Path(data_dir).glob("*.parquet"))[: max_shots or None]
    rows = []
    for f in files:
        row = pd.read_parquet(f).iloc[0].to_dict()
        if row.get("source", "DIII-D") == "DIII-D":
            row["_file"] = f.stem
            rows.append(row)
    if len(rows) < 2:
        raise SystemExit(f"need >= 2 DIII-D shots in {data_dir}, found {len(rows)}")
    return rows


# ------------------------------------------------------------------ scoring (process pool)

def _reference(psi):
    return ls.build_reference(psi, *GRID)


def _score_one(args):
    gt, ref, pred, psi_sign, means = args
    return ls.score_shot(gt, ref, pred, *GRID, psi_sign, means)


class Scorer:
    """Builds the ground-truth references once; scores any number of prediction sets."""

    def __init__(self, rows: list[dict], pool: ProcessPoolExecutor):
        self.pool = pool
        self.gts = [{"psi": target_psi(r).astype(np.float32),
                     "q95": np.asarray(r["efit_q95"], dtype=np.float64),
                     "betaN": np.asarray(r["efit_beta_n"], dtype=np.float64)} for r in rows]
        self.refs = list(pool.map(_reference, [g["psi"] for g in self.gts]))
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

    def __call__(self, preds: list[dict]) -> dict:
        # one global flux sign for the whole set, chosen from the psi residuals (as the platform)
        totals = {sg: 0.0 for sg in ls.PSI_SIGNS}
        for g, p in zip(self.gts, preds):
            rr, _, _ = ls.psi_residuals(g["psi"], p["psirz"], self.means[0])
            for sg in ls.PSI_SIGNS:
                totals[sg] += rr[sg]
        psi_sign = min(ls.PSI_SIGNS, key=lambda sg: totals[sg])
        jobs = [(g, ref, p, psi_sign, self.means) for g, ref, p in zip(self.gts, self.refs, preds)]
        acc = ls.Accum()
        acc.psi_sign = psi_sign
        per_shot = []
        for res in self.pool.map(_score_one, jobs):
            per_shot.append({"r2_psi": res["shot_r2_psi"][psi_sign], "dlcfs": res["shot_dlcfs"],
                             "sse_q95_betaN": res["ss_res_scal"].tolist()})
            acc.add(res)
        out = ls.finalize_machine(acc, self.ref_stats)
        out["per_shot"] = per_shot
        return out


# ------------------------------------------------------------------ reporting

TERMS = [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15),
         ("1 - D_LCFS", "dlcfs", 0.10), ("Consistency", "consistency", 0.20)]


def term(res, key):
    return (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]


def report(name: str, res: dict, rows: list[dict]):
    print("\n" + "=" * 66)
    print(f"  {name}:  COMPOSITE S = {res['S']:.4f}   ({len(rows)} held-out DIII-D shots)")
    print("=" * 66)
    for label, key, w in TERMS:
        v = term(res, key)
        print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    print("  Consistency:  " + "  ".join(
        f"{k}={'n/a' if v is None else f'{v:.3f}'}" for k, v in res["r2_cons_each"].items()))
    print(f"  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames")


def summary(results: dict, rows: list[dict]):
    names = list(results)
    print("\n" + "-" * 66 + "\n  per-shot R2_psi / D_LCFS\n" + "-" * 66)
    print(f"  {'shot':<12s}{'T':>5s}  " + "".join(f"{n:>22s}" for n in names))
    for i, r in enumerate(rows):
        cells = []
        for n in names:
            s = results[n]["per_shot"][i]
            d = "  n/a " if s["dlcfs"] is None else f"{s['dlcfs']:6.3f}"
            cells.append(f"{s['r2_psi']:>12.4f} / {d}")
        print(f"  {r['_file'][9:]:<12s}{len(r['efit_times']):>5d}  " + "".join(f"{c:>22s}" for c in cells))
    print("\n" + "-" * 66 + "\n  summary\n" + "-" * 66)
    print(f"  {'term':<16s}" + "".join(f"{n:>12s}" for n in names))
    for label, key, _ in TERMS:
        print(f"  {label:<16s}" + "".join(f"{term(results[n], key):>12.4f}" for n in names))
    for k in results[names[0]]["r2_cons_each"]:
        vals = [results[n]["r2_cons_each"][k] for n in names]
        print(f"    {k:<14s}" + "".join(f"{'n/a' if v is None else f'{v:.4f}':>12s}" for v in vals))
    print(f"  {'S':<16s}" + "".join(f"{results[n]['S']:>12.4f}" for n in names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--models", default="baseline,c1",
                    help="comma-separated: baseline, c1, c1+<option>[+<option>] (field, shaf, eddy)")
    ap.add_argument("--folds", type=int, default=5, help="0 = leave-one-shot-out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = load_rows(args.data_dir, args.max_shots)
    n = len(rows)
    k = n if args.folds <= 0 else min(args.folds, n)
    groups = np.empty(n, dtype=int)
    groups[np.random.RandomState(args.seed).permutation(n)] = np.arange(n) % k
    print(f"{n} DIII-D shots, {sum(len(r['efit_times']) for r in rows)} frames, {k} folds")

    results = {}
    # spawn, not fork: forking after pyarrow/BLAS have started their thread pools deadlocks the
    # children. Workers inherit this env before importing numpy, so each stays single-threaded.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        t0 = time.time()
        scorer = Scorer(rows, pool)
        print(f"reference targets built in {time.time() - t0:.0f}s")
        for name in args.models.split(","):
            t0 = time.time()
            preds = [None] * n
            for f in range(k):
                train = [r for r, g in zip(rows, groups) if g != f]
                print(f"[{name}] fold {f + 1}/{k}: train on {len(train)} shots")
                m = make_model(name).fit(lambda: iter(train))
                for i in np.nonzero(groups == f)[0]:
                    preds[i] = m.predict(rows[i])
            t_fit = time.time() - t0
            t0 = time.time()
            results[name] = scorer(preds)
            print(f"[{name}] fit+predict {t_fit:.0f}s, scoring {time.time() - t0:.0f}s")
            report(name, results[name], rows)

    summary(results, rows)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{'-'.join(results)}.json"
    meta = {"data_dir": str(args.data_dir), "folds": k, "seed": args.seed,
            "shots": [r["_file"] for r in rows], "groups": groups.tolist()}
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=float))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
