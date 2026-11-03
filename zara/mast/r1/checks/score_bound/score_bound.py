#!/usr/bin/env python3
"""Upper bound on |S(B) - S(A)| for two prediction files of the same split, without labels.

    python score_bound.py A.npz B.npz --machine mast|diii-d [--r2 ...] [--workers N] [--json out.json]

A = the submitted file, B = a rebuilt file. The composite (fusion_scoring/metrics.py) is
    S = 0.55 max(0,R2_psi) + 0.15 max(0,R2_qb) + 0.10 (1 - min(1,D_LCFS)) + 0.20 mean_j max(0,R2_cons_j)
max(0,.) and min(1,.) are 1-Lipschitz, so |dS| <= 0.55 b_psi + 0.15 b_qb + 0.10 b_D + 0.20 b_cons.

Pooled R2 terms (psi over every pixel of the fold, q95 / betaN over every frame, each shape scalar
over every frame): with d = B - A, sqrt(SSres) moves by at most ||d|| (triangle inequality), so
    |dR2| <= (2 sqrt(SSres_A) ||d|| + ||d||^2) / SST = 2 sqrt((1 - R2_A) r) + r,   r = ||d||^2 / SST.
SST of the labels is unknown. Two readings are printed:
  stand-in : SST = SST(A), the spread of A's own values on the same pooling.
  SST-safe : SST >= SST(A) / (1 + sqrt(1 - R2_A))^2. This follows from
             ||A - mean(A)|| <= ||A - mean(L)|| <= ||A - L|| + ||L - mean(L)||, so it needs no
             SST assumption; it is valid whenever the true R2_A is at least the value given.
Every bound decreases in R2_A, so the R2 values only need to be lower bounds on the truth. The
worst case over R2_A in [0, 1] is 2 sqrt(r) + r (stand-in) and 4 sqrt(r) + 4 r (SST-safe).

Shape scalars: the 7 scalars are derived from A's and B's psi with the scorer's own code
(extract_lcfs_with_sign + derive_frame, same n_iter / n_points / mask / axis sign as
local_score.score_shot). A frame where exactly one side fails is scored by the scorer with the
label mean; the stand-in for that mean is the mean of A's valid values. The LCFS search is not
continuous in psi (a 1e-6 relative change can move a contour by centimetres on a few frames), so
the changes are often sparse. For these terms the bound is the smaller of the formula above and
a range form, (||d||^2 + 2 sum_k |d_k| W_k) / SST, where W_k = max(A_k - lo, hi - A_k) and
[lo, hi] holds every scored label value (derive.PLAUSIBLE; the grid for R_axis / Z_axis).

D_LCFS: per frame the scorer uses min(1, H(X, L) / rgeo_shot), failures count as 1. Since
|H(A,L) - H(B,L)| <= H(A,B), each frame moves by at most min(1, H(A,B) / rgeo) (1 if exactly one
side fails, 0 if both fail), then the scorer's mean over frames and shots. rgeo (the shot's
time-averaged TRUE LCFS major radius) is replaced by the mean major radius of A's LCFS.

Sign: the scorer picks one global psi sign per fold, so a whole-file sign flip of B does not change
S(B); if B is closer to -A than to A it is compared as -B. The orientation used for the axis and
LCFS (AXIS_SIGN x fold sign) is read off A: the one in which A's LCFS extracts.

Never reads labels. Masks and grids come from the scorer's fusion_scoring/masks/*_envelope.npz, used only as the scorer
uses them (the masked R2 of psi and the LCFS search); the files compared are our own predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

STARTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../chris/mast/starter")  # the scorer's files
SCORING = os.path.join(STARTER, "fusion_scoring")
sys.path.insert(0, STARTER)
sys.path.insert(0, SCORING)
import local_score as ls  # noqa: E402  (constants N_ITER / N_POINTS of the scoring worker)
from common import (AXIS_SIGN, CONS_SCALARS, N_CONS, W_CONS, W_LCFS, W_PSI,  # noqa: E402
                    W_QB)
from contour import symmetric_hausdorff  # noqa: E402
from derive import PLAUSIBLE, derive_frame  # noqa: E402
from lcfs import extract_lcfs_with_sign, major_radius  # noqa: E402

MACHINES = {"mast": ("MAST", "mast_envelope.npz"), "diii-d": ("DIII-D", "d3d_envelope.npz")}
R2_NAMES = ["psi", "q95", "betaN"] + list(CONS_SCALARS)
# Development-board R2 of our entries (lower bounds are what matter: the bound decreases in R2_A).
R2_DEFAULTS = {
    "mast": dict(zip(R2_NAMES, [0.989, 0.756, 0.30, 0.68, 0.26, 0.63, 0.39, 0.45, 0.79, 0.43])),
    "diii-d": dict(zip(R2_NAMES, [0.99994, 0.9984, 0.9984] + [0.99] * N_CONS)),
}
VALS = ("q95", "betaN")


# ----------------------------------------------------------------------------- small helpers

class Moments:
    """Pooled count / mean / M2 (sum of squared deviations), merged with Chan's formula."""

    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n, self.mean, self.m2 = float(n), float(mean), float(m2)

    @classmethod
    def of(cls, x):
        x = np.asarray(x, np.float64).ravel()
        if x.size == 0:
            return cls()
        mu = float(x.mean())
        return cls(x.size, mu, float(np.sum((x - mu) ** 2)))

    def add(self, o):
        if o.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self.m2 = o.n, o.mean, o.m2
            return
        n = self.n + o.n
        d = o.mean - self.mean
        self.m2 += o.m2 + d * d * self.n * o.n / n
        self.mean += d * o.n / n
        self.n = n


def diff_stats(a, b):
    """Squared-difference pieces for arrays a, b (same shape), deferring the non-finite fill.

    The scorer replaces a non-finite prediction by the label mean m. Pixels where both sides are
    finite give sum (b - a)^2 directly; pixels where exactly one side is finite give
    sum (v - m)^2 = ss - 2 m s + n m^2 once m is known (stand-in: A's pooled mean).
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = b[both] - a[both]
    v = np.concatenate([a[fa & ~fb], b[fb & ~fa]])
    return {"dd": float(np.dot(d, d)), "n1": int(v.size), "s1": float(v.sum()),
            "ss1": float(np.dot(v, v)), "nonfin_a": int((~fa).sum()), "nonfin_b": int((~fb).sum()),
            "mom_a": Moments.of(a[fa])}


def r2_bound(dd, sst, r2a):
    """(stand-in, SST-safe, worst stand-in, worst SST-safe, r) bounds on |dR2| (each capped at 1)."""
    if dd <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if not (sst > 0):
        return 1.0, 1.0, 1.0, 1.0, math.inf
    r = dd / sst
    q = 1.0 - min(1.0, max(0.0, r2a))
    k = (1.0 + math.sqrt(q)) ** 2
    standin = 2.0 * math.sqrt(q * r) + r
    safe = 2.0 * math.sqrt(q * k * r) + k * r
    return (min(1.0, standin), min(1.0, safe), min(1.0, 2.0 * math.sqrt(r) + r),
            min(1.0, 4.0 * math.sqrt(r) + 4.0 * r), r)


def shot_ids(z):
    ids = sorted(int(m.group(1)) for k in z.files for m in [re.fullmatch(r"shot_(\d+)_psirz", k)] if m)
    return ids


def key(i, name):
    return f"shot_{i:04d}_{name}"


# ----------------------------------------------------------------------------- worker state

_W = {}


def _init(pa, pb, machine, mask_file):
    _W["A"] = np.load(pa, allow_pickle=False)
    _W["B"] = np.load(pb, allow_pickle=False)
    m = np.load(os.path.join(SCORING, "masks", mask_file))
    _W["R"], _W["Z"] = m["grid_R"], m["grid_Z"]
    _W["mc"] = m["mask_coarse"].astype(bool)
    _W["mf"] = _W["mc"].astype(np.float64)
    _W["machine"] = machine


def _load(i):
    A, B = _W["A"], _W["B"]
    out = {}
    for name in ("psirz",) + VALS:
        a, b = A[key(i, name)], B[key(i, name)]
        out[name] = (a, b)
    return out


def _phase1(args):
    """Byte identity, shapes, and the pooled-R2 pieces for psi, q95, betaN (one shot)."""
    i, flip = args
    arr = _load(i)
    res = {"i": i, "identical": True, "shape_ok": True, "keys": {}}
    for name, (a, b) in arr.items():
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        res["identical"] &= bool(same)
        if a.shape != b.shape:
            res["shape_ok"] = False
            res["keys"][name] = None
            continue
        bb = -b.astype(np.float64) if (flip and name == "psirz") else b
        res["keys"][name] = diff_stats(a, bb)
    a, b = arr["psirz"]
    if a.shape == b.shape:
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        res["dd_minus"] = float(np.sum((a64 - b64) ** 2))
        res["dd_plus"] = float(np.sum((a64 + b64) ** 2))
    return res


def _frame(psi, axis_sign):
    R, Z, mc, mf, machine = _W["R"], _W["Z"], _W["mc"], _W["mf"], _W["machine"]
    c, _ = extract_lcfs_with_sign(psi, R, Z, machine, mc, mf, n_iter=ls.N_ITER,
                                  n_points=ls.N_POINTS, axis_sign=axis_sign)
    vals = derive_frame(psi, R, Z, machine, mc, mf, contour=c, with_li=True, axis_sign=axis_sign)
    return c, np.array([vals[n] for n in CONS_SCALARS], np.float64)


def _phase2(args):
    """Derived scalars and LCFS of A and B on every frame of one shot."""
    i, flip, axis_sign, full = args
    A, B = _W["A"], _W["B"]
    pa = np.asarray(A[key(i, "psirz")], np.float64)
    pb = np.asarray(B[key(i, "psirz")], np.float64)
    if flip:
        pb = -pb
    T = pa.shape[0]
    fa = np.full((T, N_CONS), np.nan)
    fb = np.full((T, N_CONS), np.nan)
    okA = np.zeros(T, bool)
    okB = np.zeros(T, bool)
    hab = np.full(T, np.nan)
    rA = np.full(T, np.nan)
    rB = np.full(T, np.nan)
    n_reused = 0
    for k in range(T):
        cA, fa[k] = _frame(pa[k], axis_sign)
        if not full and np.array_equal(pa[k], pb[k]):
            cB, fb[k] = cA, fa[k]
            n_reused += 1
        else:
            cB, fb[k] = _frame(pb[k], axis_sign)
        okA[k], okB[k] = cA is not None, cB is not None
        if okA[k]:
            rA[k] = major_radius(cA)
        if okB[k]:
            rB[k] = major_radius(cB)
        if okA[k] and okB[k]:
            hab[k] = 0.0 if cB is cA else symmetric_hausdorff(cA, cB)
    return {"i": i, "fa": fa, "fb": fb, "okA": okA, "okB": okB, "hab": hab, "rA": rA, "rB": rB,
            "reused": n_reused}


def _orient_probe(args):
    """LCFS extraction successes of A's frames under both orientations (axis = max of s * psi)."""
    i, = args
    p = np.asarray(_W["A"][key(i, "psirz")], np.float64)
    T = p.shape[0]
    ks = sorted(set([0, T // 4, T // 2, (3 * T) // 4, T - 1]))
    out = {}
    for s in (1.0, -1.0):
        out[s] = sum(_frame(p[k], s)[0] is not None for k in ks)
    return out


# ----------------------------------------------------------------------------- reduction

def finish_r2(stats, r2a):
    """Combine per-shot diff_stats of one quantity into the R2 bound."""
    mom = Moments()
    dd = n1 = s1 = ss1 = 0.0
    nfa = nfb = 0
    for s in stats:
        mom.add(s["mom_a"])
        dd += s["dd"]
        n1 += s["n1"]; s1 += s["s1"]; ss1 += s["ss1"]
        nfa += s["nonfin_a"]; nfb += s["nonfin_b"]
    m = mom.mean
    dd += max(0.0, ss1 - 2.0 * m * s1 + n1 * m * m)
    b = r2_bound(dd, mom.m2, r2a)
    return {"dd": dd, "sst_A": mom.m2, "n": int(mom.n), "r": b[4], "r2_A": r2a, "standin": b[0],
            "safe": b[1], "worst_standin": b[2], "worst_safe": b[3], "nonfinite_A": nfa,
            "nonfinite_B": nfb, "one_side_nonfinite": int(n1)}


def cons_ranges(R, Z):
    """Interval every SCORED label value of each shape scalar lies in.

    kappa / tri_top / tri_bot / volume / li: derive._clamp_to_nan turns values outside
    derive.PLAUSIBLE into nan, and nan reference frames are not scored. R_axis / Z_axis: the
    O-point is a grid cell refined by at most one cell (o_point._subpixel clips to [-1, 1]).
    """
    dR, dZ = float(R[1] - R[0]), float(Z[1] - Z[0])
    rng = {"R_axis": (float(R.min()) - abs(dR), float(R.max()) + abs(dR)),
           "Z_axis": (float(Z.min()) - abs(dZ), float(Z.max()) + abs(dZ))}
    rng.update({k: (float(lo), float(hi)) for k, (lo, hi) in PLAUSIBLE.items()})
    return rng


def finish_cons(p2, r2d, ranges):
    """Per shape scalar: min of the Cauchy-Schwarz bound and the range bound.

    With A', B' the values the scorer uses (a failed frame takes the label mean m; stand-in: mean
    of A's valid values) and d = B' - A':  SSres_B - SSres_A = ||d||^2 + 2 sum_k d_k (A'_k - L_k).
      Cauchy-Schwarz: |sum d_k (A'_k - L_k)| <= ||d|| sqrt(SSres_A)        (the r2_bound formula)
      range:          |sum d_k (A'_k - L_k)| <= sum |d_k| W_k,  W_k = max(A'_k - lo, hi - A'_k)
    since L_k is in [lo, hi] on every scored frame. The range form is much tighter when only a
    few frames change (LCFS jumps), the Cauchy-Schwarz form when every frame moves a little.
    """
    fa = np.concatenate([p["fa"] for p in p2])
    fb = np.concatenate([p["fb"] for p in p2])
    out = {}
    for j, name in enumerate(CONS_SCALARS):
        a, b = fa[:, j], fb[:, j]
        okA, okB = np.isfinite(a), np.isfinite(b)
        mom = Moments.of(a[okA])
        m = mom.mean
        a2 = np.where(okA, a, m)
        b2 = np.where(okB, b, m)
        d = b2 - a2
        dd = float(np.dot(d, d))
        lo, hi = ranges[name]
        w = np.maximum(a2 - lo, hi - a2)
        l1w = float(np.sum(np.abs(d) * w))
        r2a = r2d[name]
        cs = r2_bound(dd, mom.m2, r2a)
        q = 1.0 - min(1.0, max(0.0, r2a))
        k = (1.0 + math.sqrt(q)) ** 2
        if dd <= 0:
            rb = (0.0, 0.0, 0.0, 0.0)
        elif not mom.m2 > 0:
            rb = (1.0, 1.0, 1.0, 1.0)
        else:
            base = (2.0 * l1w + dd) / mom.m2
            rb = tuple(min(1.0, x) for x in (base, k * base, base, 4.0 * base))
        cols = ("standin", "safe", "worst_standin", "worst_safe")
        res = {"dd": dd, "sst_A": mom.m2, "n": int(mom.n), "r": cs[4], "r2_A": r2a,
               "range": [lo, hi], "n_changed": int(np.count_nonzero(d)),
               "fail_A": int((~okA).sum()), "fail_B": int((~okB).sum())}
        for c, x_cs, x_rb in zip(cols, cs[:4], rb):
            res[c] = min(x_cs, x_rb)
            res[c + "_cs"] = x_cs
            res[c + "_range"] = x_rb
        out[name] = res
    return out


def finish_lcfs(p2):
    shot_b, n_one, n_both_fail, n_frames, hmax = [], 0, 0, 0, 0.0
    for p in p2:
        okA, okB, hab = p["okA"], p["okB"], p["hab"]
        T = okA.size
        if T == 0:
            continue
        rho = np.nanmean(p["rA"]) if okA.any() else (np.nanmean(p["rB"]) if okB.any() else np.nan)
        fb = np.zeros(T)
        both = okA & okB
        one = okA ^ okB
        if both.any():
            fb[both] = np.minimum(1.0, hab[both] / rho) if rho > 0 else 1.0
            hmax = max(hmax, float(np.max(hab[both])))
        fb[one] = 1.0
        n_one += int(one.sum())
        n_both_fail += int((~okA & ~okB).sum())
        n_frames += T
        shot_b.append(float(fb.mean()))
    b = float(np.mean(shot_b)) if shot_b else 0.0
    return {"bound": b, "frames": n_frames, "one_side_fail": n_one, "both_fail": n_both_fail,
            "max_hausdorff_m": hmax}


# ----------------------------------------------------------------------------- main

def parse_r2(spec, machine):
    r2 = dict(R2_DEFAULTS[machine])
    if not spec:
        return r2
    items = [x.strip() for x in spec.split(",") if x.strip()]
    if all("=" in x for x in items):
        for x in items:
            k, v = x.split("=", 1)
            k = k.strip()
            if k == "all":
                r2 = {n: float(v) for n in R2_NAMES}
            elif k == "cons":
                r2.update({n: float(v) for n in CONS_SCALARS})
            elif k in r2:
                r2[k] = float(v)
            else:
                raise SystemExit(f"--r2: unknown name {k!r}; use {R2_NAMES}, all or cons")
    elif len(items) == len(R2_NAMES):
        r2 = dict(zip(R2_NAMES, map(float, items)))
    else:
        raise SystemExit(f"--r2 takes {len(R2_NAMES)} comma-separated values in the order {R2_NAMES}, "
                         "or name=value pairs (names, 'all', 'cons')")
    return r2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("A", help="submitted prediction file (.npz)")
    ap.add_argument("B", help="rebuilt prediction file for the same split (.npz)")
    ap.add_argument("--machine", required=True, choices=list(MACHINES))
    ap.add_argument("--r2", default="", help="R2 of A (lower bounds): 10 values in the order "
                    + ",".join(R2_NAMES) + ", or name=value pairs (also all=, cons=). Default: Development-board values")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--max-shots", type=int, default=0, help="only the first N shots (testing)")
    ap.add_argument("--full", action="store_true",
                    help="no shortcuts: score even byte-identical files/frames from scratch")
    ap.add_argument("--psi-sign", choices=["auto", "+1", "-1"], default="auto",
                    help="orientation for axis/LCFS: axis = max of AXIS_SIGN*sign*psi (auto: read off A)")
    ap.add_argument("--json", help="write all numbers here")
    a = ap.parse_args()

    t0 = time.time()
    scorer_machine, mask_file = MACHINES[a.machine]
    r2d = parse_r2(a.r2, a.machine)
    za, zb = np.load(a.A, allow_pickle=False), np.load(a.B, allow_pickle=False)
    ids_a, ids_b = shot_ids(za), shot_ids(zb)
    if ids_a != ids_b:
        print(f"ERROR: shot sets differ (A {len(ids_a)}, B {len(ids_b)}): files are not for the same split")
        return 2
    for i in ids_a:
        for n in VALS:
            if key(i, n) not in za.files or key(i, n) not in zb.files:
                print(f"ERROR: {key(i, n)} missing from one file")
                return 2
    za.close(); zb.close()
    ids = ids_a[: a.max_shots] if a.max_shots else ids_a
    print(f"score_bound: {a.machine} ({scorer_machine} scorer conventions, mask {mask_file}), {len(ids)} shots")
    print(f"  A = {a.A}\n  B = {a.B}")

    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_init, initargs=(a.A, a.B, scorer_machine, mask_file))

    # ---- phase 1: identity + global sign of B relative to A + psi / q95 / betaN pieces
    p1 = pool.map(_phase1, [(i, False) for i in ids], chunksize=4)
    bad = [p["i"] for p in p1 if not p["shape_ok"]]
    if bad:
        print(f"ERROR: array shapes differ on {len(bad)} shots (first {bad[:5]}): not comparable")
        return 2
    identical = all(p["identical"] for p in p1)
    dd_minus = sum(p["dd_minus"] for p in p1)
    dd_plus = sum(p["dd_plus"] for p in p1)
    flip = dd_plus < dd_minus
    if flip:
        print("  note: B's psi is closer to -A than to A; the scorer's fold sign makes S(-B) = S(B), "
              "so B is compared as -B")
        p1 = pool.map(_phase1, [(i, True) for i in ids], chunksize=4)
    n_ident_shots = sum(p["identical"] for p in p1)
    print(f"  byte-identical shots: {n_ident_shots}/{len(ids)}   ({time.time() - t0:.0f} s)")

    res = {"A": a.A, "B": a.B, "machine": a.machine, "n_shots": len(ids), "r2_A": r2d,
           "flip_B": bool(flip), "identical_shots": int(n_ident_shots)}
    if identical and not a.full:
        print("\n  byte-identical: every psirz / q95 / betaN array of B equals A's bit for bit")
        print("  TOTAL |S(B) - S(A)| <= 0  (exactly; all terms 0)")
        res.update({"byte_identical": True, "total": {k: 0.0 for k in
                    ("standin", "safe", "worst_standin", "worst_safe")}})
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        pool.close()
        return 0
    res["byte_identical"] = bool(identical)

    psi = finish_r2([p["keys"]["psirz"] for p in p1], r2d["psi"])
    vals = {n: finish_r2([p["keys"][n] for p in p1], r2d[n]) for n in VALS}

    # ---- orientation for the axis / LCFS (the scorer's AXIS_SIGN x fold sign), read off A
    if a.psi_sign == "auto":
        probe_ids = ids[:: max(1, len(ids) // 12)][:12]
        votes = {1.0: 0, -1.0: 0}
        for o in pool.map(_orient_probe, [(i,) for i in probe_ids]):
            for s in votes:
                votes[s] += o[s]
        axis_sign = 1.0 if votes[1.0] >= votes[-1.0] else -1.0
        print(f"  orientation probe on A (LCFS found, axis = max of s*psi): s=+1 {votes[1.0]}, s=-1 {votes[-1.0]}"
              f" -> axis sign {axis_sign:+.0f} (fold psi sign {axis_sign * AXIS_SIGN[scorer_machine]:+.0f})")
    else:
        axis_sign = AXIS_SIGN[scorer_machine] * float(a.psi_sign)
    res["axis_sign"] = axis_sign

    # ---- phase 2: derived scalars + LCFS for every frame
    t1 = time.time()
    p2 = pool.map(_phase2, [(i, flip, axis_sign, a.full) for i in ids], chunksize=2)
    pool.close()
    reused = sum(p["reused"] for p in p2)
    nfr = sum(p["okA"].size for p in p2)
    print(f"  derived scalars + LCFS: {nfr} frames ({reused} identical psi frames reused)   ({time.time() - t1:.0f} s)")
    m = np.load(os.path.join(SCORING, "masks", mask_file))
    cons = finish_cons(p2, r2d, cons_ranges(m["grid_R"], m["grid_Z"]))
    lcfs = finish_lcfs(p2)

    # ---- combine
    cols = ("standin", "safe", "worst_standin", "worst_safe")
    qb = {c: 0.5 * (vals["q95"][c] + vals["betaN"][c]) for c in cols}
    cb = {c: float(np.mean([cons[n][c] for n in CONS_SCALARS])) for c in cols}
    total = {c: W_PSI * psi[c] + W_QB * qb[c] + W_LCFS * lcfs["bound"] + W_CONS * cb[c] for c in cols}

    def row(label, w, d, extra=""):
        r = d.get("r", float("nan"))
        print(f"  {label:<14s}{w:>6s} {d.get('r2_A', float('nan')):>8.5g} {r:>11.3e}"
              + "".join(f" {d[c]:>11.3e}" for c in cols) + extra)

    print("\n  per-term bounds on |d term| (unweighted); r = ||B-A||^2 / SST(A)")
    print(f"  {'term':<14s}{'w':>6s} {'R2_A':>8s} {'r':>11s} {'stand-in':>11s} {'SST-safe':>11s}"
          f" {'worst/in':>11s} {'worst/safe':>11s}")
    row("R2_psi", "0.55", psi, f"   nonfinite A/B px {psi['nonfinite_A']}/{psi['nonfinite_B']}")
    for n in VALS:
        row(f"  {n}", "", vals[n], f"   nonfinite A/B {vals[n]['nonfinite_A']}/{vals[n]['nonfinite_B']}")
    row("R2_qb (mean)", "0.15", {**qb, "r2_A": float("nan"), "r": float("nan")})
    for n in CONS_SCALARS:
        c_ = cons[n]
        row(f"  {n}", "", c_, f"   frames changed {c_['n_changed']}, fails A/B {c_['fail_A']}/{c_['fail_B']},"
            f" via {'range' if c_['standin_range'] < c_['standin_cs'] else 'CS'}")
    row("Consistency", "0.20", {**cb, "r2_A": float("nan"), "r": float("nan")})
    dl = lcfs["bound"]
    row("D_LCFS", "0.10", {c: dl for c in cols} | {"r2_A": float("nan"), "r": float("nan")},
        f"   max H(A,B) {lcfs['max_hausdorff_m'] * 1e3:.3g} mm; one-side LCFS fails {lcfs['one_side_fail']},"
        f" both {lcfs['both_fail']} of {lcfs['frames']}")
    print("\n  weighted contributions (stand-in / SST-safe):"
          f"  psi {W_PSI * psi['standin']:.3e}/{W_PSI * psi['safe']:.3e}"
          f"  qb {W_QB * qb['standin']:.3e}/{W_QB * qb['safe']:.3e}"
          f"  D {W_LCFS * dl:.3e}"
          f"  cons {W_CONS * cb['standin']:.3e}/{W_CONS * cb['safe']:.3e}")
    print("\n  TOTAL |S(B) - S(A)| <=")
    labels = {"standin": "R2_A as given, SST = SST(A)            [stand-in]",
              "safe": "R2_A as given, SST lower bound          [no SST assumption]",
              "worst_standin": "any R2_A in [0,1], SST = SST(A)",
              "worst_safe": "any R2_A in [0,1], SST lower bound"}
    for c in cols:
        print(f"    {total[c]:.6g}   {labels[c]}")
    for tol, what in ((1e-4, "shipped-weights rule 1e-4"), (1e-2, "retrain rule 0.01")):
        verdict = ("within" if total["safe"] <= tol else
                   "within on the stand-in only" if total["standin"] <= tol else "NOT shown within")
        print(f"    {what}: {verdict}")
    print(f"  ({time.time() - t0:.0f} s)")

    res.update({"psi": psi, "vals": vals, "r2_qb": qb, "cons": cons, "consistency": cb, "lcfs": lcfs,
                "total": total, "frames_reused": int(reused)})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
