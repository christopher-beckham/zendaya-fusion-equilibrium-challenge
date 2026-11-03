"""Best MAST blend under a training-time budget: every subset of member groups, first-order estimate + mix (c) offset.

Cost of a subset = union of the training runs behind its files (hours per run from a cost file; shared runs counted
once). Score = first-order map part (exact flux map, D model, shape terms calibrated on b2 / t2b / t3c) + exact q95 and
betaN optima, shifted so that mix (c) reproduces its board score. No MAST label is read.

usage: python budget2.py --costs costs2.json --pairs <add_pairs dir> --board board_new.json --out <json> [--workers 90]
"""
import argparse
import itertools
import json
import os
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

import fpb

ap = argparse.ArgumentParser()
ap.add_argument("--costs", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--board", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--groups", default="", help="restrict to these groups")
ap.add_argument("--blends", default="", help="name=weights.json,... further scored blends used as calibration points")
a = ap.parse_args()

C = json.load(open(a.costs))
E = fpb.Est(verbose=False, extra_pairs=[a.pairs], board_extra=a.board,
            extra_blends=dict(x.split("=", 1) for x in a.blends.split(",") if x))
E._fc = None
FC = E._first_cache()
GROUPS = {g: v for g, v in C["groups"].items() if not a.groups or g in a.groups.split(",")}


def cost(groups, key):
    runs = set(r for g in groups for r in GROUPS[g]["runs"])
    return sum(C["runs"][r][key] for r in runs)


def evaluate(files):
    allowed = {k: [n for n in files if n not in fpb.BARRED[k] and n in E.bd and (k == "maps" or E.bd[n].get(k) is not None)]
               for k in ("maps", "q95", "betaN")}
    M = [n for n in allowed["maps"] if E.bd[n].get("cons")]
    if not M or not allowed["q95"] or not allowed["betaN"]:
        return None
    pix = np.array([FC["ix"][n] for n in M])

    def f(w):
        full = np.zeros(len(FC["names"]))
        full[pix] = np.clip(w, 0, None)
        full /= full.sum()
        psi, d, c = E.fast_parts(full)
        return -(0.55 * psi + 0.10 * (1 - d) + 0.20 * c)
    best = None
    for s in range(4):
        w0 = np.random.default_rng(s).dirichlet(np.ones(len(M)))
        r = minimize(f, w0, bounds=[(0, 1)] * len(M), constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                     method="SLSQP", options=dict(ftol=1e-11, maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    wq, r2q = E.opt_exact("q95", allowed["q95"])
    wb, r2b = E.opt_exact("betaN", allowed["betaN"])
    wm = np.clip(best.x, 0, None); wm /= wm.sum()
    return dict(S=-best.fun + 0.15 * 0.5 * (r2q + r2b), q95=r2q, betaN=r2b,
                maps={n: round(float(v), 3) for n, v in zip(M, wm) if v > 0.004},
                wq={n: round(v, 3) for n, v in wq.items() if v > 0.004}, wb={n: round(v, 3) for n, v in wb.items() if v > 0.004})


def job(groups):
    files = sorted(set(f for g in groups for f in GROUPS[g]["files"]))
    try:
        r = evaluate(files)
    except Exception as e:                       # noqa: BLE001
        return dict(groups=groups, error=str(e)[:200])
    if r is None:
        return None
    r.update(groups=list(groups), cost_asrun=cost(groups, "asrun"), cost_min=cost(groups, "min"))
    return r


# mix (c) offset
wc = fpb.WBLEND["t3c"]
full = E.first_vec(wc["maps"])
psi, d, c = E.fast_parts(full)
s_c = 0.55 * psi + 0.10 * (1 - d) + 0.20 * c + 0.15 * 0.5 * (E.exact("q95", wc["q95"]) + E.exact("betaN", wc["betaN"]))
OFF = 0.815356 - s_c
print(f"mix (c) first-order {s_c:.4f} -> offset {OFF:+.4f}", flush=True)

names = list(GROUPS)
subsets = [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)]
print(f"{len(subsets)} subsets of {names}", flush=True)
with get_context("fork").Pool(a.workers) as pool:
    res = [r for r in pool.map(job, subsets, chunksize=8) if r]
for r in res:
    if "S" in r:
        r["S_board_est"] = r["S"] + OFF
json.dump(dict(offset=OFF, results=res), open(a.out, "w"), indent=1, default=float)
ok = [r for r in res if "S" in r]
for key in ("cost_min", "cost_asrun"):
    print(f"\nFrontier by {key} (best estimated board S at or below each budget):")
    for B in (4, 6, 8, 10, 12, 14, 15.6, 16, 18.5, 20, 22, 24, 30, 40):
        cand = [r for r in ok if r[key] <= B + 1e-9]
        if not cand:
            continue
        b = max(cand, key=lambda r: r["S_board_est"])
        print(f"  <= {B:4.1f} h: S {b['S_board_est']:.4f}  cost {b[key]:.2f}  groups {'+'.join(b['groups'])}")
