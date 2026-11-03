"""Critic features and the real-vs-synthetic classifier.

Feature blocks (all from inputs only; no efit label column of any real file is read):
  coil      compact coil/Ip/TF model inputs (features_v5 scalars + flux harmonics). In gsolve data these are the
            real frame's own inputs, so they are IDENTICAL for a synthetic frame and its real twin; they only
            condition the classifier (joint Thomson-given-coils behaviour).
  model_th  the 13 Thomson-derived features_v5 inputs (betaN_e, R_ax_th, p/Te/ne profile ratios), computed exactly as
            features_v2._thomson_rows does, over channels inside the model window (R - R0)/a in [-0.02, 0.60].
  rel       extra joint Thomson features on the same window channels, from a lightly smoothed profile: peak
            positions of Te / ne / p, the Te-ne peak offset, fall-off points, log-slopes, Te0/Ip, ne0 / Greenwald density.
  hist      changes over -20 / +20 ms of model_th and rel (coil changes are identical by construction).
  noise     roughness / cold-channel fraction / channel count in the window: trivial artefacts, reported, NOT in the objective.
  outwin    profile shape outside the model window (inner half, outer edge, in/out asymmetry): reported, NOT in the objective.
"""
from __future__ import annotations

import os
import sys

os.environ["FEC_MAST_MASK"] = "limiter"
import numpy as np  # noqa: E402

sys.path.insert(0, "/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn")
import features_v2 as fv  # noqa: E402

R0, A = 0.84, 0.56                       # features_v2.GEOMETRY["mast"]
WIN = (-0.02, 0.60)                      # features_v5.THOMSON_WINDOW
MU0 = 4e-7 * np.pi
TF_TURNS = 24
E = 1.602176634e-19

MODEL_TH = ["betaN_e", "R_ax_th", "p_rho0.1", "p_rho0.2", "p_rho0.3", "Te_rho0.1", "Te_rho0.2", "Te_rho0.3", "Te_rho0.4",
            "ne_rho0.1", "ne_rho0.2", "ne_rho0.3", "ne_rho0.4"]
COIL = ["Bz_over_Bp_a", "Br_over_Bp_a", "hexa_c3", "q_ITER", "dIp_dt_norm", "V_loop_norm"] + \
    [f"{c}{k}_rho{r}" for r in ("0.5", "1.0") for c, ks in (("c", (1, 2, 3, 5)), ("s", (1, 2, 3, 4, 5))) for k in ks]
REL = ["rf_Rpk_te", "rf_Rpk_ne", "rf_Rpk_p", "rf_pk_off", "rf_te70", "rf_ne70", "rf_p70",
       "rf_slope_te", "rf_slope_ne", "rf_te0_ip", "rf_ne0_gw", "rf_curv_te"]
NOISE = ["nz_rough_te", "nz_rough_ne", "nz_cold_frac", "nz_nvalid"]
OUTWIN = ["ow_rin", "ow_rout", "ow_asym_te", "ow_asym_ne", "ow_Rpk_te_full", "ow_in_out_te05"]
SLICE_COLS = ["pe_pa"] + MODEL_TH[1:] + REL + NOISE + OUTWIN   # per-slice columns (betaN_e built from pe_pa later)


def _smooth(v):
    """5-point running median then 3-point mean (edges kept)."""
    n = len(v)
    if n < 5:
        return v.copy()
    pad = np.pad(v, 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    p2 = np.pad(med, 1, mode="edge")
    return (p2[:-2] + p2[1:-1] + p2[2:]) / 3


def _fall(x, y, ypk, ipk, frac):
    """First x outboard of the peak where y < frac * ypk (linear), NaN if not reached."""
    below = np.nonzero(y[ipk:] < frac * ypk)[0]
    if not len(below):
        return np.nan
    k = ipk + below[0]
    if k == ipk:
        return x[k]
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    return x0 + (frac * ypk - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1


def slice_features(R, te, ne, ip_kA, out=None):
    """One Thomson slice (core chord radii R (m), Te (eV), ne (m^-3)) -> dict of SLICE_COLS (model window rules)."""
    o = {c: np.nan for c in SLICE_COLS} if out is None else out
    R, te, ne = np.asarray(R, float), np.asarray(te, float), np.asarray(ne, float)
    u = (R - R0) / A
    inw = (u >= WIN[0]) & (u <= WIN[1])
    # --- model features (features_v2._thomson_rows on the window channels)
    rows = fv._thomson_rows(R[inw], te[None, inw], ne[None, inw], R0, A)[0]
    o["pe_pa"] = rows[1]
    o["R_ax_th"] = (rows[2] - R0) / A
    for c, v in zip(MODEL_TH[2:], rows[3:]):
        o[c] = v
    # --- window channels, valid
    ok = inw & np.isfinite(te) & np.isfinite(ne) & np.isfinite(R) & (te > 0) & (ne > 0)
    o["nz_nvalid"] = float(ok.sum())
    if ok.sum() >= 8:
        s = np.argsort(R[ok]); x, T, N = u[ok][s], te[ok][s] / 1e3, ne[ok][s] / 1e19
        Ts, Ns = _smooth(T), _smooth(N)
        Ps = Ts * Ns
        core = np.abs(x) < fv.CORE_HALF
        if core.any():
            it = int(np.argmax(np.where(core, Ts, -np.inf)))
            ine = int(np.argmax(np.where(core, Ns, -np.inf)))
            ip_ = int(np.argmax(np.where(core, Ps, -np.inf)))
            o["rf_Rpk_te"], o["rf_Rpk_ne"], o["rf_Rpk_p"] = x[it], x[ine], x[ip_]
            o["rf_pk_off"] = x[ine] - x[it]
            o["rf_te70"] = _fall(x, Ts, Ts[it], it, 0.7) - x[it]
            o["rf_ne70"] = _fall(x, Ns, Ns[ine], ine, 0.7) - x[ine]
            o["rf_p70"] = _fall(x, Ps, Ps[ip_], ip_, 0.7) - x[ip_]
            ob = x >= x[ip_]
            if ob.sum() >= 4 and (Ts[ob] > 0).all() and (Ns[ob] > 0).all():
                o["rf_slope_te"] = np.polyfit(x[ob], np.log(Ts[ob]), 1)[0]
                o["rf_slope_ne"] = np.polyfit(x[ob], np.log(Ns[ob]), 1)[0]
            ipm = max(abs(ip_kA), 100.0) / 1e3
            o["rf_te0_ip"] = Ts[it] / ipm
            o["rf_ne0_gw"] = (Ns[ine] / 10.0) / (ipm / (np.pi * A ** 2))
            nb = np.abs(x - x[it]) < 0.15
            if nb.sum() >= 5:
                c2 = np.polyfit(x[nb] - x[it], Ts[nb] / Ts[it], 2)[0]
                o["rf_curv_te"] = c2
        o["nz_rough_te"] = float(np.median(np.abs(T - Ts) / np.maximum(Ts, 0.02)))
        o["nz_rough_ne"] = float(np.median(np.abs(N - Ns) / np.maximum(Ns, 0.05)))
        o["nz_cold_frac"] = float(np.mean(T < 0.02))
    # --- full chord (outside the model window): report only
    okf = np.isfinite(te) & np.isfinite(ne) & np.isfinite(R) & (te > 0) & (ne > 0)
    if okf.sum() >= 20:
        s = np.argsort(R[okf]); rr, T, N = R[okf][s], _smooth(te[okf][s] / 1e3), _smooth(ne[okf][s] / 1e19)
        if T.max() > 0.05 and (T >= 0.05 * T.max()).sum() >= 8:
            hs = rr[T >= 0.05 * T.max()]
            rin, rout = hs.min(), hs.max()
            rpk = rr[np.argmax(T)]
            o["ow_rin"], o["ow_rout"], o["ow_Rpk_te_full"] = rin, rout, rpk
            if rin < rpk < rout:
                o["ow_asym_te"] = (rout - rpk) / (rpk - rin)
                rpn = rr[np.argmax(N)]
                if rin < rpn < rout:
                    o["ow_asym_ne"] = (rout - rpn) / (rpn - rin)
                # Te at half-way in / out (same normalised distance from the peak)
                ti = np.interp(rpk - 0.5 * (rpk - rin), rr, T)
                to = np.interp(rpk + 0.5 * (rout - rpk), rr, T)
                o["ow_in_out_te05"] = ti / max(to, 1e-3)
    return o


def betaN_e(pe_pa, ip_kA, tf_kA):
    """features_v2.shot_features: 100 * 2 mu0 pe a / (Bt Ip[MA])."""
    bt = np.abs(MU0 * TF_TURNS * tf_kA * 1000.0 / (2.0 * np.pi * R0))
    ip_a = np.maximum(np.abs(ip_kA), fv.IP_FLOOR_KA) * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * 2.0 * MU0 * pe_pa * A / (bt * ip_a / 1e6)


def frame_thomson_cols():
    """Columns produced per frame (time-interpolated slices on real, the remapped profile on synthetic)."""
    return MODEL_TH + REL + NOISE + OUTWIN


OBJ_NOW = COIL + MODEL_TH + REL
OBJ_HIST = MODEL_TH + REL


def objective_columns():
    return OBJ_NOW + [f"dprev:{c}" for c in OBJ_HIST] + [f"dnext:{c}" for c in OBJ_HIST]


def group_of(c):
    pre = ""
    if c.startswith("dprev:") or c.startswith("dnext:"):
        pre, c = "d:", c.split(":", 1)[1]
    if c in COIL:
        return pre + "coil"
    if c in MODEL_TH:
        return pre + ("model_" + c.split("_rho")[0] if "_rho" in c else "model_" + c)
    if c in REL:
        return pre + "rel_" + c[3:]
    return pre + c


# ---------------------------------------------------------------- classifier

def cv_auc(X, y, groups, folds=5, seed=0, max_iter=200, perm_feats=None, n_jobs_threads=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    imp = None
    gkf = GroupKFold(folds, shuffle=True, random_state=seed) if "shuffle" in GroupKFold.__init__.__code__.co_varnames else GroupKFold(folds)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=30,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if perm_feats is not None and k == 0:
            imp = group_perm(m, X[te], y[te], perm_feats)
    return roc_auc_score(y, oof), oof, imp


def group_perm(m, X, y, feats, n_rep=2):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    base = roc_auc_score(y, m.predict_proba(X)[:, 1])
    gs = {}
    for j, f in enumerate(feats):
        gs.setdefault(group_of(f), []).append(j)
    out = {}
    for g, js in gs.items():
        d = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, js] = X[rng.permutation(len(y))][:, js]
            d.append(base - roc_auc_score(y, m.predict_proba(Xp)[:, 1]))
        out[g] = float(np.mean(d))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def single_auc(X, y, feats):
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, f in enumerate(feats):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() > 50 and len(np.unique(y[ok])) == 2:
            out[f] = float(roc_auc_score(y[ok], x[ok]))
    return out


def energy_distance(Xa, Xb, n_max=1500, seed=0):
    """Energy distance between two samples of standardised features (NaN -> column median)."""
    rng = np.random.default_rng(seed)
    if len(Xa) > n_max:
        Xa = Xa[rng.choice(len(Xa), n_max, replace=False)]
    if len(Xb) > n_max:
        Xb = Xb[rng.choice(len(Xb), n_max, replace=False)]
    Z = np.concatenate([Xa, Xb])
    med = np.nanmedian(Z, 0)
    Z = np.where(np.isfinite(Z), Z, med)
    q1, q3 = np.percentile(Z, [25, 75], 0)
    sc = np.where(q3 - q1 > 0, (q3 - q1) / 1.349, 1.0)
    Z = (Z - med) / sc
    Z = np.clip(Z, -10, 10)
    a, b = Z[:len(Xa)], Z[len(Xa):]

    def md(P, Q):
        s = 0.0
        for i in range(0, len(P), 500):
            s += np.sqrt(((P[i:i + 500, None, :] - Q[None]) ** 2).sum(-1)).sum()
        return s / (len(P) * len(Q))
    return 2 * md(a, b) - md(a, a) - md(b, b)


# ---------------------------------------------------------------- fixed-capacity critic, held-out evaluation

DEFAULT_PARAMS = dict(max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, max_iter=600)


def _hgb(params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)


def fit_critic(X, y, g, params, seed=0, es_frac=0.2):
    """Fit on (X, y), early-stopping on a shot-grouped es_frac split of it. Returns (model, best n_iter)."""
    from sklearn.metrics import log_loss
    shots = np.unique(g)
    rng = np.random.default_rng(seed + 17)
    es_shots = set(rng.choice(shots, max(2, int(round(es_frac * len(shots)))), replace=False).tolist())
    es = np.array([s in es_shots for s in g])
    m = _hgb(params, seed).fit(X[~es], y[~es])
    best, best_k = np.inf, 1
    for k, p in enumerate(m.staged_predict_proba(X[es]), 1):
        ll = log_loss(y[es], np.clip(p[:, 1], 1e-4, 1 - 1e-4), labels=[0, 1])
        if ll < best - 1e-6:
            best, best_k = ll, k
    return m, best_k


def predict_at(m, k, X):
    p = None
    for i, q in enumerate(m.staged_predict_proba(X), 1):
        p = q[:, 1]
        if i == k:
            break
    return p


def heldout_eval(Xtr, ytr, gtr, Xva, yva, gva, params, seed=0, perm_feats=None, n_boot=200):
    """Held-out log loss / AUC of a fixed-capacity critic, with a shot-bootstrap 95% interval for the AUC."""
    from sklearn.metrics import log_loss, roc_auc_score
    m, k = fit_critic(Xtr, ytr, gtr, params, seed)
    p = np.clip(predict_at(m, k, Xva), 1e-4, 1 - 1e-4)
    out = dict(ll=float(log_loss(yva, p, labels=[0, 1])), auc=float(roc_auc_score(yva, p)), n_iter=int(k))
    rng = np.random.default_rng(seed + 5)
    shots = np.unique(gva)
    pos = {s: np.nonzero(gva == s)[0] for s in shots}
    bs = []
    for _ in range(n_boot):
        ii = np.concatenate([pos[s] for s in rng.choice(shots, len(shots))])
        if len(np.unique(yva[ii])) == 2:
            bs.append(roc_auc_score(yva[ii], p[ii]))
    out["auc_lo"], out["auc_hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    if perm_feats is not None:
        imp = {}
        gs = {}
        for j, f in enumerate(perm_feats):
            gs.setdefault(group_of(f), []).append(j)
        r2 = np.random.default_rng(0)
        for gname, js in gs.items():
            d = []
            for _ in range(2):
                Xp = Xva.copy()
                Xp[:, js] = Xva[r2.permutation(len(yva))][:, js]
                pp = np.clip(predict_at(m, k, Xp), 1e-4, 1 - 1e-4)
                d.append(log_loss(yva, pp, labels=[0, 1]) - out["ll"])
            imp[gname] = float(np.mean(d))
        out["imp"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    return out
