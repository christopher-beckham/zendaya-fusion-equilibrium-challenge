"""Search map weights for the MAST team blend with exact derived scalars (see fpb.py). No MAST label is read.

Round 0: t3c, the first-order optimum (and trade-off variants), the psi-only optimum, single-member checks, and
perturbations of the first-order optimum toward each member. Later rounds: optimise first-order S plus a linear
correction fitted to the exact estimates, evaluate the proposal and a line search, repeat.

usage: python search.py --out <dir> [--workers 32] [--rounds 3] [--pool name,name,...] [--tag x]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

import fpb

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--pool", default="", help="restrict map members (default: every allowed member)")
ap.add_argument("--qpool", default="", help="restrict q95/betaN members")
ap.add_argument("--shots", type=int, default=1206)
ap.add_argument("--no-t3c", action="store_true")
ap.add_argument("--extra-pairs", default="", help="comma list of add_pairs.py output dirs (new members)")
ap.add_argument("--board-extra", default="", help="json with board entries of the new members")
ap.add_argument("--bar", default="", help="key:name,... barred from that quantity's weights (keys maps, q95, betaN)")
ap.add_argument("--quick", action="store_true", help="evaluate only the first-order optimum (budget variants)")
ap.add_argument("--blends", default="", help="name=weights.json,... further scored blends used as calibration points")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
print("== estimator calibration", flush=True)
E = fpb.Est(extra_pairs=[d for d in a.extra_pairs.split(",") if d], board_extra=a.board_extra or None,
            extra_blends=dict(x.split("=", 1) for x in a.blends.split(",") if x))
for x in [x for x in a.bar.split(",") if x]:
    k, n = x.split(":")
    fpb.BARRED[k].add(n)

allowed = {k: [n for n in fpb.FILES if n not in fpb.BARRED[k] and n in E.bd] for k in ("maps", "q95", "betaN")}
if a.pool:
    allowed["maps"] = [n for n in a.pool.split(",") if n in allowed["maps"]]
if a.qpool:
    qp = a.qpool.split(",")
    allowed["q95"] = [n for n in allowed["q95"] if n in qp]
    allowed["betaN"] = [n for n in allowed["betaN"] if n in qp]
M = [n for n in allowed["maps"] if E.bd[n].get("cons")]
print(f"map pool {M}\nq95 pool {allowed['q95']}\nbetaN pool {allowed['betaN']}", flush=True)

wq, r2q = E.opt_exact("q95", allowed["q95"])
wb, r2b = E.opt_exact("betaN", allowed["betaN"])
print(f"q95 optimum R2 {r2q:.4f}: {json.dumps({k: round(v, 4) for k, v in wq.items()})}")
print(f"betaN optimum R2 {r2b:.4f}: {json.dumps({k: round(v, 4) for k, v in wb.items()})}", flush=True)

# sanity: the scored blends through the same estimator (their own vectors are calibration points)
for B in ("b2", "t2b"):
    w = fpb.WBLEND[B]["maps"]
    cf = E.cons_first(w)
    print(f"{B}: first-order cons {np.mean([max(0, v) for v in cf.values()]):.4f} vs board "
          f"{np.mean([max(0, v) for v in E.bd[B]['cons'].values()]):.4f}; psi exact {E.exact('psi', w):.4f} vs board {E.bd[B]['psi']:.4f}")


FC = E._first_cache()
PIX = np.array([FC["ix"][n] for n in M])


def s_first(wvec, names, cw=1.0):
    full = np.zeros(len(FC["names"]))
    full[PIX] = np.clip(wvec, 0, None)
    full /= full.sum()
    psi, d, c = E.fast_parts(full)
    return 0.55 * psi + 0.10 * (1 - d) + 0.20 * cw * c


def opt_first(names, cw=1.0, extra=None, starts=12, x0=None):
    best = None
    k = len(names)
    inits = [np.random.default_rng(s).dirichlet(np.ones(k)) for s in range(starts)]
    if x0 is not None:
        inits = [x0] + inits
    for w0 in inits:
        f = (lambda w: -(s_first(w, names, cw) + (extra(w) if extra else 0.0)))
        r = minimize(f, w0, bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                     method="SLSQP", options=dict(ftol=1e-12, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def clean(wvec, names, thr=0.004):
    w = {n: float(v) for n, v in zip(names, wvec) if v > thr}
    t = sum(w.values())
    return {n: round(v / t, 4) for n, v in w.items()}


cands, labels = [], []


def add(lbl, w):
    w = {n: v for n, v in w.items() if v > 0}
    for c in cands:
        if set(c) == set(w) and all(abs(c[n] - w[n]) < 1e-4 for n in w):
            return
    cands.append(w); labels.append(lbl)


if a.quick:
    a.rounds = 1
    a.no_t3c = True
wt3c = json.load(open(f"{fpb.TB}/wT3c.json"))["maps"]
if not a.no_t3c and set(wt3c) <= set(M):
    add("t3c", wt3c)
x1 = opt_first(M)
w1 = clean(x1, M)
add("first", w1)
for cw in (() if a.quick else (0.5, 2.0)):
    add(f"first_cw{cw}", clean(opt_first(M, cw), M))
# psi-only optimum
wpsi, r2psi = E.opt_exact("psi", M)
if not a.quick:
    add("psi_only", {n: round(v, 4) for n, v in wpsi.items() if v > 0.004})
    add("Z15_single", {"Z15": 1.0})
for m in ([] if a.quick else M):
    w = {n: 0.85 * v for n, v in w1.items()}
    w[m] = w.get(m, 0.0) + 0.15
    add(f"first+{m}", w)

rows = []
X_all, S_all, F_all = [], [], []
for rnd in range(a.rounds):
    new = [i for i in range(len(cands)) if i >= len(rows)]
    if not new:
        break
    t1 = time.time()
    sv = fpb.derive([cands[i] for i in new], a.workers, a.shots)
    print(f"\n== round {rnd}: derived {len(new)} candidates in {time.time() - t1:.0f} s", flush=True)
    for j, i in enumerate(new):
        ce = E.cons_est(sv[j])
        sc = E.score(cands[i], ce, r2q, r2b)
        wv = np.array([cands[i].get(n, 0.0) for n in M]); wv = wv / wv.sum()
        sf = s_first(wv, M) + 0.15 * 0.5 * (r2q + r2b)
        rows.append(dict(label=labels[i], w=cands[i], S_first=sf, **sc))
        X_all.append(wv); S_all.append(sc["S"]); F_all.append(sf)
        np.save(f"{a.out}/cons_{len(rows) - 1:03d}.npy", sv[j])
        print(f"  {labels[i]:22s} S_est {sc['S']:.4f} (first-order {sf:.4f}) psi {sc['psi']:.4f} D {sc['D']:.4f} "
              f"cons {sc['cons']:.4f} [" + " ".join(f"{sc['c_' + k]:.3f}" for k in fpb.CONS) + "]", flush=True)
    json.dump(dict(rows=rows, q95=dict(w=wq, r2=r2q), betaN=dict(w=wb, r2=r2b), pool=M,
                   calib=dict(css={k: float(v) for k, v in E.css.items()}, loo={k: list(map(float, v)) for k, v in E.loo.items()},
                              gamma=E.gamma)),
              open(f"{a.out}/results.json", "w"), indent=1, default=float)
    if rnd == a.rounds - 1:
        break
    # linear correction: delta(w) = S_exact - S_first, fitted on all evaluated candidates (ridge)
    X = np.array(X_all); dlt = np.array(S_all) - np.array(F_all)
    lam = 1e-3
    A = np.c_[X]
    beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ dlt)
    resid = dlt - A @ beta
    print(f"  correction fit: rms residual {np.sqrt((resid ** 2).mean()):.5f} (delta mean {dlt.mean():+.4f}, sd {dlt.std():.4f})")
    best_i = int(np.argmax(S_all))
    xb = X_all[best_i]
    xn = opt_first(M, extra=lambda w: float(w @ beta), x0=xb, starts=6)
    wn = clean(xn, M)
    add(f"r{rnd + 1}_opt", wn)
    xnv = np.array([wn.get(n, 0.0) for n in M]); xnv /= xnv.sum()
    for t in (0.5, 1.5):
        xm = np.clip(xb + t * (xnv - xb), 0, None)
        add(f"r{rnd + 1}_line{t}", clean(xm / xm.sum(), M))
    for m in sorted(M, key=lambda n: -xnv[M.index(n)])[:4]:
        xm = 0.9 * xnv; xm[M.index(m)] += 0.1
        add(f"r{rnd + 1}_opt+{m}", clean(xm / xm.sum(), M))
    for m in [n for n in M if xnv[M.index(n)] > 0.05][:4]:
        xm = xnv.copy(); xm[M.index(m)] *= 0.5
        add(f"r{rnd + 1}_opt-{m}", clean(xm / xm.sum(), M))

best = max(rows, key=lambda r: r["S"])
print(f"\nBEST {best['label']}: S_est {best['S']:.4f}  maps {json.dumps(best['w'])}")
print(f"q95 {json.dumps({k: round(v, 4) for k, v in wq.items()})}\nbetaN {json.dumps({k: round(v, 4) for k, v in wb.items()})}")
json.dump({"maps": best["w"], "q95": {k: round(v, 4) for k, v in wq.items() if v > 0.004},
           "betaN": {k: round(v, 4) for k, v in wb.items() if v > 0.004}}, open(f"{a.out}/best_weights.json", "w"), indent=1)
print(f"total {time.time() - t0:.0f} s")
