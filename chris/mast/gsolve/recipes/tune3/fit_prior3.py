#!/usr/bin/env python3
"""Step A: fit theta ~ f(z) on DIII-D training frames, cross-validated by shot; check on held-out DIII-D shots.

theta: li (EFIT), log K (K = <p>/<p_e>). z: zfeat.FEATS. Models: ridge on clipped, standardised features (the
transferable model: smooth, extrapolates linearly, clipped to the DIII-D range) and gradient boosting (reference).
Writes cond_model.json (ridge coefficients, clip ranges, residual covariance) and fit_prior.json (report numbers).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import zfeat

HERE = Path(__file__).resolve().parent


def prep(df):
    df = df.copy()
    df["logK"] = np.log(df.K)
    ok = np.isfinite(df[[f"z_{k}" for k in zfeat.FEATS_ALL if f"z_{k}" in df]]).all(1) & np.isfinite(df.li) & (df.K > 0.7) & (df.K < 12) \
        & (df.pe_av > 0) & (df.betaN > 0.02)
    return df[ok].reset_index(drop=True)


def design(df, feats, lo, hi, mu, sd):
    X = np.clip(df[[f"z_{k}" for k in feats]].values, lo, hi)
    return (X - mu) / sd


def main():
    import sys
    sfx, out = sys.argv[1], sys.argv[2]          # e.g. _th cond_model_th  (input d3d/train_th.parquet, outputs out.pkl/json)
    tr = pd.read_parquet(HERE / "d3d" / f"train{sfx}.parquet"); va = pd.read_parquet(HERE / "d3d" / f"val{sfx}.parquet")
    if "th_ok" in tr:
        tr, va = tr[tr.th_ok], va[va.th_ok]
    tr, va = prep(tr), prep(va)
    # transferable (dimensionless) features only: raw ramp rate (1/s) and absolute Te are machine-scale dependent
    # (MAST is outside the DIII-D range on 24-29% of frames); rr_tau replaces rr
    feats = [k for k in zfeat.FEATS if k not in ("rr", "te0", "tav")]
    if all(f"z_{k}" in tr for k in zfeat.HFEATS):
        feats += zfeat.HFEATS
    rep = dict(n_train=len(tr), shots_train=int(tr.shot.nunique()), n_val=len(va), shots_val=int(va.shot.nunique()))
    # univariate Spearman correlations
    rep["spearman"] = {t: {k: float(spearmanr(tr[f"z_{k}"], tr[t]).statistic) for k in zfeat.FEATS} for t in ("li", "logK")}
    Xr = tr[[f"z_{k}" for k in feats]].values
    lo, hi = np.percentile(Xr, 1, axis=0), np.percentile(Xr, 99, axis=0)
    Xc = np.clip(Xr, lo, hi)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    X = (Xc - mu) / sd
    g = tr.shot.values
    cv = GroupKFold(5)
    model = {"feats": feats, "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "targets": {}}
    resid, resid_g, models = {}, {}, {}
    rep["cv"] = {}
    for t in ("li", "logK"):
        y = tr[t].values
        pr_l, pr_g = np.zeros(len(y)), np.zeros(len(y))
        for a_, b_ in cv.split(X, y, g):
            pr_l[b_] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[a_], y[a_]).predict(X[b_])
            pr_g[b_] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                     min_samples_leaf=40).fit(X[a_], y[a_]).predict(X[b_])
        r2 = lambda p: float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))  # noqa: E731
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X, y)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40).fit(X, y)
        Xv = design(va, feats, lo, hi, mu, sd)
        yv = va[t].values
        r2v = lambda p: float(1 - np.sum((yv - p) ** 2) / np.sum((yv - yv.mean()) ** 2))  # noqa: E731
        rep["cv"][t] = dict(ridge_cv_r2=r2(pr_l), gbm_cv_r2=r2(pr_g), ridge_val_r2=r2v(m.predict(Xv)),
                            gbm_val_r2=r2v(gb.predict(Xv)), sd_y=float(y.std()), sd_resid_ridge=float(np.std(y - pr_l)),
                            alpha=float(m.alpha_),
                            coef={k: float(c) for k, c in sorted(zip(feats, m.coef_), key=lambda kc: -abs(kc[1]))})
        model["targets"][t] = dict(coef=m.coef_.tolist(), intercept=float(m.intercept_))
        models[t] = dict(ridge=m, gbm=gb)
        rep["cv"][t]["gbm_resid_sd"] = float(np.std(y - pr_g))
        resid_g[t] = y - pr_g
        resid[t] = y - pr_l
    Rm = np.stack([resid["li"], resid["logK"]], 1)
    model["cov"] = np.cov(Rm.T).tolist()
    model["cov_gbm"] = np.cov(np.stack([resid_g["li"], resid_g["logK"]], 1).T).tolist()
    import pickle
    old = pickle.loads((HERE / "cond_model.pkl").read_bytes())   # gsolve li emulators do not depend on the teacher
    (HERE / f"{out}.pkl").write_bytes(pickle.dumps(dict(model=model, models=models, emu=old["emu"], emu_rep=old.get("emu_rep"))))
    rep["resid_corr"] = float(np.corrcoef(Rm.T)[0, 1])
    rep["theta_quantiles"] = {t: np.percentile(tr[t], [5, 50, 95]).round(3).tolist() for t in ("li", "logK", "K")}
    (HERE / f"{out}.json").write_text(json.dumps(model, indent=1))
    (HERE / f"fit_prior_{out}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "spearman"}, indent=1))
    print(pd.DataFrame(rep["spearman"]).round(2).sort_values("li").to_string())


if __name__ == "__main__":
    main()
