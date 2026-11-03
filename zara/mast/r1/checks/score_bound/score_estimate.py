#!/usr/bin/env python3
"""Expected change in S from a score_bound.py result, if the differences between the two files are unrelated to the errors
against the labels (for example the last-digit rounding of another GPU type).

    python score_estimate.py BOUND.json

Per pooled R2 term, with r = ||B - A||^2 / SST(A) and the Development-board R2 of A as stand-in (both in BOUND.json): a
change d unrelated to the error e adds ||d||^2 to the residual on average, so the term moves by r on average; the cross
term 2 e.d adds a spread of standard deviation 2 sqrt((1 - R2) r / n), with n the number of shots, each counted as one
independent unit. Printed: the average (r per term), and the average plus one and two standard deviations, with the terms'
standard deviations added (an upper end: added in quadrature they would be smaller). The LCFS term enters with its strict
bound in every line. Weights of S as in the scorer: 0.55 psi, 0.15 mean(q95, betaN), 0.20 mean of the shape scalars,
0.10 LCFS.
"""
import json
import math
import sys

d = json.load(open(sys.argv[1]))
n = d["n_shots"]


def term(t, k):  # average (k = 0) or average + k standard deviations of |dR2|; one standard deviation = 2 sqrt((1 - R2) r / n)
    return t["r"] + k * 2 * math.sqrt(max(0.0, 1 - t["r2_A"]) * t["r"] / n)


for name, k in (("average", 0), ("average + 1 sd", 1), ("average + 2 sd", 2)):
    psi = 0.55 * term(d["psi"], k)
    qb = 0.15 * (term(d["vals"]["q95"], k) + term(d["vals"]["betaN"], k)) / 2
    cons = 0.20 * sum(term(v, k) for v in d["cons"].values()) / len(d["cons"])
    lcfs = 0.10 * d["lcfs"]["bound"]
    print(f"{name}: |dS| {psi + qb + cons + lcfs:.2e}  (psi {psi:.2e}, q95/betaN {qb:.2e}, consistency {cons:.2e}, "
          f"LCFS strict bound {lcfs:.2e}; {n} shots)")
