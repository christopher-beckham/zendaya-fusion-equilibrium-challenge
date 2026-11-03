"""Per-frame calibration targets (R_axis, Z_axis, kappa, li, ...) from the mean map, the member maps and the head networks,
written as per-shot files for kapcal.py / axcal.py / lical.py (--targets). Label-free: works on holdout and public test.

    python targets.py --frames FRAMES.npz --cache $FEQ_CACHE/nn/diii_d_train --heads DIR [DIR ...] \
        --rule rule.json --out DIR

FRAMES.npz = mfunc.py output (or the holdout equivalent, frames_holdout.py) whose first entry is the mean map and the rest
the member maps. rule.json: {term: spec}, term in R_axis, Z_axis, kappa, li (others ignored by the transforms). spec:
  {"wh": a, "wm": b, "b": c}               target = x0 + a (head mean - x0) + b (member mean - x0) + c
  + "gate": "agree"                        only where heads and members move x0 in the same direction (else no target)
  + "alpha_bins": {"feat": name, "edges": [...], "wh": [...], "wm": [...]}  per-bin weights (feat: sh, sm, dis, dhm, aip,
                                           kT, regime = 1 for |Ip| < 0.6 MA or the last 10 % of a shot)
  + "gbm": "model.pkl"                     target = x0 + shrink * model(F) (+ the linear part if wh / wm given: model on top)
  + "maxd": d                              |target - x0| capped at d
  + "mingap": g                            no target where |target - x0| < g
  + "alt_w": w                             head mean = (1 - w) main set + w alternative set (--heads-alt)
A term with no spec gets no target (NaN). li targets are capped at the li guard 4.9.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import kfeat

CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
GEO = ["Zmax", "Zmin", "Rmax", "Rmin", "R_Zmax", "R_Zmin"]
KEYS = {"R_axis": "R", "Z_axis": "Z", "kappa": "kappa", "tri_top": "tri_top", "tri_bot": "tri_bot", "volume": "volume", "li": "li"}
KEYS.update({g: g for g in GEO})


def bin_feat(X, j, name):
    if name == "sh":
        return X["sh"][:, j]
    if name == "sm":
        return X["sm"][:, j]
    if name == "dis":
        return np.abs(X["xh"][:, j] - X["x0"][:, j])
    if name == "dhm":
        return np.abs(X["xh"][:, j] - X["xm"][:, j])
    if name == "aip":
        return X["aip"]
    if name == "kT":
        return X["kT"]
    if name == "regime":                    # 1 = low current (|Ip| < 0.6 MA) or last 10 % of the shot, 0 = other frames
        aip = X["aip"]
        lo = aip < (0.6e6 if np.nanmax(aip) > 1e3 else 0.6)
        return (lo | (X["kT"] >= 0.9)).astype(np.float64)
    raise ValueError(name)


def apply_rule(X, rule, X_alt=None):
    """-> (N, 13) absolute targets (7 consistency terms, then the 6 LCFS extremes; NaN = no target)."""
    N = len(X["x0"])
    out = np.full((N, 13), np.nan)
    for j, n in enumerate(CONS + GEO):
        sp = rule.get(n)
        if not sp:
            continue
        aw = float(sp.get("alt_w", 0.0)) if X_alt is not None else 0.0      # weight of the alternative head set
        if j < 7:
            xh = (1 - aw) * X["xh"][:, j] + (aw * X_alt["xh"][:, j] if aw else 0.0)
            x0, dh, dm = X["x0"][:, j], xh - X["x0"][:, j], X["xm"][:, j] - X["x0"][:, j]
        else:
            g = j - 7
            xh = (1 - aw) * X["hgm"][:, g] + (aw * X_alt["hgm"][:, g] if aw else 0.0)
            x0, dh, dm = X["g0"][:, g], xh - X["g0"][:, g], X["gm"][:, g] - X["g0"][:, g]
        wh = np.full(N, float(sp.get("wh", 0.0)))
        wm = np.full(N, float(sp.get("wm", 0.0)))
        if "alpha_bins" in sp:
            ab = sp["alpha_bins"]
            f = bin_feat(X, j, ab["feat"])
            b = np.digitize(f, ab["edges"])
            wh = np.asarray(ab["wh"], np.float64)[b]
            wm = np.asarray(ab.get("wm", [0.0] * (len(ab["edges"]) + 1)), np.float64)[b]
        t = x0 + wh * dh + wm * dm + float(sp.get("b", 0.0))
        if "gbm" in sp:
            with open(sp["gbm"], "rb") as fh:
                g = pickle.load(fh)
            t = t + float(sp.get("shrink", 1.0)) * g.predict(np.nan_to_num(X["F"], nan=0.0) if sp.get("nan0") else X["F"])
        if "maxd" in sp:
            t = x0 + np.clip(t - x0, -sp["maxd"], sp["maxd"])
        if "mingap" in sp:                      # only frames whose correction is large (realisation noise is per warped frame)
            t = np.where(np.abs(t - x0) >= sp["mingap"], t, np.nan)
        if sp.get("gate") == "agree":
            t = np.where(np.sign(dh) == np.sign(dm), t, np.nan)
        if n == "li":
            t = np.minimum(t, kfeat.LI_GUARD)
        out[:, j] = t
    return out


def frames_features(fr, cache, heads):
    stems, shot, k = fr["stems"], fr["shot"], fr["k"]
    off = np.r_[0, np.cumsum(np.bincount(shot, minlength=len(stems)))]
    N = len(shot)
    hc = np.full((len(heads), N, 7), np.nan)
    hg = np.full((len(heads), N, 6), np.nan)
    t, ip, bt, T = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i, s in enumerate(stems):
        sl = slice(off[i], off[i + 1])
        c = np.load(Path(cache) / f"{s}.npz")
        t[sl], ip[sl], bt[sl], T[sl] = c["t"], c["ip"], c["bt"], off[i + 1] - off[i]
        for h, d in enumerate(heads):
            z = np.load(Path(d) / f"{s}.npz")
            hc[h, sl] = z["cons"]
            if "geo" in z.files:
                hg[h, sl] = z["geo"]
    cons, geo = fr["cons"], fr["geo"]
    return kfeat.build(cons[0], geo[0], cons[1:], geo[1:], hc, hg, shot, k, T, t, ip, bt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--heads", type=Path, nargs="+", required=True)
    ap.add_argument("--heads-alt", type=Path, nargs="*", default=[],
                    help="alternative head set; a term with 'alt_w': w uses (1 - w) main head mean + w alternative head mean")
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr = dict(np.load(a.frames))
    X = frames_features(fr, a.cache, a.heads)
    X_alt = frames_features(fr, a.cache, a.heads_alt) if a.heads_alt else None
    rule = json.loads(a.rule.read_text())
    tg = apply_rule(X, rule, X_alt)
    a.out.mkdir(parents=True, exist_ok=True)
    stems, shot = fr["stems"], fr["shot"]
    for i, s in enumerate(stems):
        sel = shot == i
        np.savez(a.out / f"{s}.npz", **{KEYS[n]: tg[sel, j] for j, n in enumerate(CONS + GEO)})
    nt = np.isfinite(tg).sum(0)
    print(f"{len(stems)} shots, {len(shot):,} frames; targets per term " + ", ".join(f"{n} {v}" for n, v in zip(CONS + GEO, nt)))
    (a.out / "rule.json").write_text(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
