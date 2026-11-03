"""Pin an unscored member's board R2 from the board result of a scored blend that contains it (no MAST label read).

For a pooled-R2 quantity (flux map, q95, betaN) with blend weights w (sum 1), error energies E_i = (1 - R2_i) SS_y and
pairwise distances D_ij = |x_i - x_j|^2, the blend error energy is
    E_B = A0 + w_u * sum_{i != u} w_i (E_i - D_iu) + w_u * E_u,      A0 = sum_{i,j != u} w_i w_j G_ij,
so E_u follows exactly from the board's R2_B. Shape scalars use the same relation on their derived values (first order,
the blend's own scalars are not a linear blend), D_LCFS uses the D model (weighted D minus gamma * 1/2 w'Hw).

usage: python pin.py --member FC --blend s1fc --weights wS1_FC.json --result '<json of the blend board numbers>' --out board_pin_FC.json
"""
import argparse
import json

import numpy as np

import fpb

ap = argparse.ArgumentParser()
ap.add_argument("--member", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--result", required=True, help="json: {psi, q95, betaN, D, cons: {7 scalars}} of the scored blend")
ap.add_argument("--board", default="board_new.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

E = fpb.Est(verbose=False, extra_pairs=["/fsx/zara.ercan/cache/ch2/g4/final_push/pairs_new"], board_extra=a.board)
W = json.load(open(a.weights))
R = json.loads(a.result) if a.result.strip().startswith("{") else json.load(open(a.result))
u = a.member
out = {}


def solve(key, w, r2B, ssv):
    names = [n for n, v in w.items() if v > 0]
    wv = np.array([w[n] for n in names], float); wv /= wv.sum()
    D = E.dmat(key, names)
    iu = names.index(u)
    Ek = np.array([0.0 if n == u else (1 - E.r2(n, key)) * ssv for n in names])
    G = fpb.gram(Ek, D)
    mask = np.ones(len(names), bool); mask[iu] = False
    A0 = wv[mask] @ G[np.ix_(mask, mask)] @ wv[mask]
    lin = wv[iu] * float(wv[mask] @ (Ek[mask] - D[iu, mask]))
    EB = (1 - r2B) * ssv
    Eu = (EB - A0 - lin) / wv[iu]
    return 1 - Eu / ssv, wv[iu]


for key, wk in (("psi", "maps"), ("q95", "q95"), ("betaN", "betaN")):
    if W[wk].get(u, 0) > 0:
        out[key], wu = solve(key, W[wk], R[key], E.ss[key])
        print(f"{u} {key}: pinned R2 {out[key]:.6f} (member weight {wu:.3f}; blend board R2 {R[key]:.6f})")
cons = {}
if W["maps"].get(u, 0) > 0:
    for k in fpb.CONS:
        cons[k], _ = solve(k, W["maps"], R["cons"][k], E.css[k])
    out["cons"] = {k: float(v) for k, v in cons.items()}
    print(f"{u} shape terms (first order): " + " ".join(f"{k} {v:+.3f}" for k, v in cons.items()))
    # D_LCFS from the D model
    names = [n for n, v in W["maps"].items() if v > 0]
    wv = np.array([W["maps"][n] for n in names], float); wv /= wv.sum()
    H = np.array([[0 if p == q else E.haus[(p, q)] for q in names] for p in names])
    iu = names.index(u)
    rest = sum(wv[i] * E.bd[n]["D"] for i, n in enumerate(names) if n != u)
    out["D"] = float((R["D"] - rest + E.gamma * 0.5 * wv @ H @ wv) / wv[iu])
    print(f"{u} D_LCFS (model): {out['D']:.4f}")
S = 0.55 * out.get("psi", np.nan) + 0.15 * 0.5 * (out.get("q95", np.nan) + out.get("betaN", np.nan)) + 0.10 * (1 - out.get("D", np.nan)) \
    + 0.20 * np.mean([max(0, v) for v in out.get("cons", {}).values()]) if "cons" in out else float("nan")
print(f"{u} implied stand-alone S (if its betaN were known): {S:.4f}")
json.dump({u: dict(out, src=f"pinned from scored blend ({a.weights}) board result")}, open(a.out, "w"), indent=1, default=float)
