"""Conditional prior over the hidden physics, taught by DIII-D (fit_prior.py) and gsolve's own li response (fit_emu.py).

Per frame with input features z (zfeat.FEATS order) and per variant:
    (li*, log K*) ~ N( f(z) + (d_li, d_lk),  c_s^2 * Sigma )     f, Sigma: DIII-D fit (c_model 'gbm' or 'ridge')
    K   = exp(log K*)  clipped to [1.0, 6.0]
    am  ~ U(am_lo, am_hi)  (as before)
    an  = the value on a log grid 0.1..8 whose emulated gsolve li (given am, K, z) is closest to li*
Frames without z (no Thomson table) get NaN: the caller keeps the old draws there.
Draws use the generator's Draws object (common random numbers, shot correlation).
"""
import pickle

import numpy as np

import zfeat

_CACHE = {}
AN_GRID = np.exp(np.linspace(np.log(0.1), np.log(8.0), 41))


def load(path):
    if path not in _CACHE:
        _CACHE[path] = pickle.loads(open(path, "rb").read())
    return _CACHE[path]


def _design(M, Zfull):
    idx = [zfeat.FEATS.index(k) for k in M["feats"]]
    X = np.clip(Zfull[:, idx], M["lo"], M["hi"])
    return (X - np.array(M["mu"])) / np.array(M["sd"]), Zfull[:, idx]


def mean_theta(cp, Zfull, kind="gbm"):
    M = cp["model"]
    X, _ = _design(M, Zfull)
    ok = np.isfinite(X).all(1)
    mu = np.full((len(X), 2), np.nan)
    if ok.any():
        mu[ok, 0] = cp["models"]["li"][kind].predict(X[ok])
        mu[ok, 1] = cp["models"]["logK"][kind].predict(X[ok])
    return mu, ok


def invert_an(cp, machine, li_t, am, K, Zraw, ok):
    """an on AN_GRID whose emulated li is closest to li_t (rows with ok)."""
    emu = cp["emu"][machine]
    an = np.full(len(li_t), np.nan)
    i = np.nonzero(ok)[0]
    if not len(i):
        return an
    G = len(AN_GRID)
    X = np.c_[np.repeat(np.log(AN_GRID)[None], len(i), 0).ravel(), np.repeat(np.log(am[i]), G),
              np.repeat(np.log(K[i]), G), np.repeat(Zraw[i], G, axis=0)]
    L = emu.predict(X).reshape(len(i), G)
    an[i] = AN_GRID[np.argmin(np.abs(L - li_t[i, None]), axis=1)]
    return an


def draw(cp, Zfull, kn, dr, machine="mast"):
    kind = kn.get("c_model", "gbm")
    mu, ok = mean_theta(cp, Zfull, kind)
    cov = np.array(cp["model"]["cov_gbm" if kind == "gbm" else "cov"]) * float(kn.get("c_s", 1.0)) ** 2
    Lc = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
    e = np.stack([dr.normal(), dr.normal()], 1) @ Lc.T
    li_t = np.clip(mu[:, 0] + float(kn.get("d_li", 0.0)) + e[:, 0], 0.5, 2.2)
    K = np.clip(np.exp(mu[:, 1] + float(kn.get("d_lk", 0.0)) + e[:, 1]), 1.0, 6.0)
    am = dr.uniform(kn["am_lo"], max(kn["am_hi"], kn["am_lo"]))
    _, Zraw = _design(cp["model"], Zfull)
    an = invert_an(cp, machine, li_t, am, np.where(ok, K, 2.0), Zraw, ok)
    li_t = np.where(ok, li_t, np.nan)
    return np.where(ok, K, np.nan), am, an, li_t
