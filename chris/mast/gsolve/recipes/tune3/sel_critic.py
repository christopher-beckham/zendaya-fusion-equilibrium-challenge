#!/usr/bin/env python3
"""Critic-guided variant selection (density-ratio rejection), DIII-D test.

1. geometry features (d3d_critic.geom, zbal = 0 so ax_Zbal = Z_ax/a; z_bal/a enters as conditioning) for every accepted
   variant of an 8-variant replay (gen_d3d --variants 8 --keep 8) and for the EFIT map of the same frames.
2. selection critic: HGB classifier real (z, EFIT geom) vs generated (z, gsolve geom, each frame's variants weighted to
   total 1) fitted on the TRAIN-subset replay (seltr), conditioning = Thomson-edge z + z_bal/a (inputs MAST also has).
3. on the VAL replay (selva) re-rank the accepted variants by  err - lam * logit p(real)  and write
   d3dchk/selva_l<lam>_<k>.npz (rank replaced) for d3d_score / compare_d3d / d3d_critic.
    python sel_critic.py feats seltr train | feats selva val | fit | apply LAM [LAM ...]
"""
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import d3d_critic as dc

HERE = Path(__file__).resolve().parent
ZC = [f"z_{k}" for k in dc.ZFE]
FEATS = [k for k in dc.GEOM if k not in ("sq", "q50")]   # ax_Zbal here = Z_ax / a (zbal 0); sq, q50: MAST outside DIII-D 1-99% on > 20%


def frames(split):
    return pd.read_parquet(HERE / "d3d" / ("train_sel.parquet" if split == "train" else "val_th.parquet"))


def zb_th(split, df):
    if split == "train":
        return np.load(HERE / "d3d" / "train3_zbal.npy")[df.i3.values]
    return np.load(HERE / "d3d" / "val3_zbal.npy")


def feats(tag, split, workers=8):
    df = frames(split)
    key = {(s, f): i for i, (s, f) in enumerate(zip(df.shot.values, df.frame.values))}
    rows, args, meta = [], [], []
    for fn in sorted(glob.glob(str(HERE / "d3dchk" / f"{tag}_[0-9].npz"))):
        z0 = np.load(fn, allow_pickle=True)
        z = {k: z0[k] for k in ("rank", "shot", "frame", "psi", "err")}
        for j in range(len(z["rank"])):
            i = key[(z["shot"][j], int(z["frame"][j]))]
            args.append((-np.asarray(z["psi"][j], np.float64), 0.0, dc.tf_F(df.tf.values[i])))
            meta.append(dict(file=Path(fn).name, j=j, i=i, rank=int(z["rank"][j]), err=float(z["err"][j])))
    with ProcessPoolExecutor(workers) as ex:
        G = pd.DataFrame(list(ex.map(dc.geom, args, chunksize=8)))
    G = pd.concat([pd.DataFrame(meta), G], axis=1)
    G.to_parquet(HERE / "d3dchk" / f"{tag}_vgeom.parquet")
    # EFIT side
    if split == "train":
        P = np.load(HERE / "d3d" / "train3_psi.npy", mmap_mode="r")
        ea = [(np.asarray(P[k], np.float64), 0.0, dc.tf_F(t)) for k, t in zip(df.i3.values, df.tf.values)]
    else:
        P = np.load(HERE / "d3d" / "val_psi.npy", mmap_mode="r")
        ea = [(np.asarray(P[k], np.float64), 0.0, dc.tf_F(t)) for k, t in enumerate(df.tf.values)]
    with ProcessPoolExecutor(workers) as ex:
        E = pd.DataFrame(list(ex.map(dc.geom, ea, chunksize=8)))
    E["i"] = np.arange(len(df))
    E.to_parquet(HERE / "d3dchk" / f"efit_{split}_vgeom.parquet")
    print(tag, len(G), "variant rows;", len(E), "EFIT rows", flush=True)


def cond(split, df, idx):
    zb = zb_th(split, df)[idx]
    a = np.clip(0.5 * (df.R_out.values - df.R_in.values)[idx], 0.3, 1.0)
    return np.c_[df[ZC].values[idx], zb / a]


def X_of(split, df, T, idx):
    return np.c_[cond(split, df, idx), T[FEATS].values].astype(np.float32)


def fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    df = frames("train")
    G = pd.read_parquet(HERE / "d3dchk" / "seltr_vgeom.parquet")
    E = pd.read_parquet(HERE / "d3dchk" / "efit_train_vgeom.parquet")
    E = E[E.i.isin(G.i.unique())]
    nv = G.groupby("i").i.transform("size").values
    X = np.r_[X_of("train", df, E, E.i.values), X_of("train", df, G, G.i.values)]
    y = np.r_[np.ones(len(E)), np.zeros(len(G))].astype(int)
    w = np.r_[np.ones(len(E)), 1.0 / nv]
    g = np.r_[df.shot.values[E.i.values], df.shot.values[G.i.values]]
    us = np.unique(g); fold = dict(zip(us, np.random.default_rng(0).permutation(len(us)) % 5)); fo = np.array([fold[s] for s in g])
    P = dict(max_leaf_nodes=7, learning_rate=0.03, min_samples_leaf=40, l2_regularization=1.0)
    # pick the number of iterations by shot-grouped CV (weighted log loss), then refit on all TRAIN rows
    curves = []
    for k in range(5):
        m = HistGradientBoostingClassifier(early_stopping=False, max_iter=600, random_state=k, **P).fit(X[fo != k], y[fo != k], sample_weight=w[fo != k])
        te = fo == k
        curves.append([log_loss(y[te], np.clip(p[:, 1], 1e-4, 1 - 1e-4), sample_weight=w[te]) for p in m.staged_predict_proba(X[te])])
    cm = np.mean(curves, 0); it = int(np.argmin(cm)) + 1
    m = HistGradientBoostingClassifier(early_stopping=False, max_iter=it, random_state=0, **P).fit(X, y, sample_weight=w)
    (HERE / "sel_critic.pkl").write_bytes(pickle.dumps(dict(model=m, feats=FEATS, n_iter=it)))
    rep = dict(n_iter=it, cv_gain=float(np.log(2) - cm[it - 1]), n_real=int(len(E)), n_gen=int(len(G)))
    print(json.dumps(rep)); (HERE / "sel_critic.json").write_text(json.dumps(rep))


SRC = __import__("os").environ.get("SRC", "selva")


def apply(lams):
    df = frames("val")
    M = pickle.loads((HERE / "sel_critic.pkl").read_bytes())
    G = pd.read_parquet(HERE / "d3dchk" / f"{SRC}_vgeom.parquet")
    p = np.clip(M["model"].predict_proba(X_of("val", df, G, G.i.values))[:, 1], 1e-3, 1 - 1e-3)
    G["logit"] = np.log(p / (1 - p))
    import os
    vmax = int(os.environ.get("VMAX", "99"))
    if vmax < 99:      # emulate fewer solved variants (MAST generator: 4): drop variants >= vmax, re-rank the rest
        G["variant"] = np.concatenate([np.load(HERE / "d3dchk" / fn)["variant"][g.j.values] for fn, g in G.groupby("file", sort=False)])
        G = G[G.variant < vmax].copy()
    for lam in lams:
        tag = f"{SRC}_l{str(lam).replace('.', 'p')}" + (f"_v{vmax}" if vmax < 99 else "")
        sc = G.err.values - lam * G.logit.values if lam < 1e6 else -G.logit.values
        G["sc"] = sc
        G["nr"] = G.groupby("i").sc.rank(method="first").astype(int) - 1
        for fn, g in G.groupby("file"):
            z = dict(np.load(HERE / "d3dchk" / fn, allow_pickle=True))
            r = np.full(len(z["rank"]), 99); r[g.j.values] = g.nr.values
            z["rank"] = r
            k = fn.split("_")[-1]
            np.savez(HERE / "d3dchk" / f"{tag}_{k}", **z)
        print(tag, "changed rank-0 on", float((G[G.nr == 0]["rank"] != 0).mean()), "of frames", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "feats":
        feats(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "fit":
        fit()
    else:
        apply([float(v) for v in sys.argv[2:]])
