#!/usr/bin/env python3
"""Per-feature giveaways of the DIII-D flux-geometry critic: 5-fold (by shot) out-of-fold permutation importance
(increase of log loss, nats) for each geometry feature.   geom_imp.py TAG [frames parquet]"""
import sys
import numpy as np, pandas as pd, json
from sklearn.metrics import log_loss
import d3d_critic as dc
tag = sys.argv[1]; fr = sys.argv[2] if len(sys.argv) > 2 else "val_th.parquet"
df = pd.read_parquet(dc.HERE / "d3d" / fr)
Rg = pd.read_parquet(dc.HERE / "d3dchk" / "efit_geom.parquet").set_index("row"); Gg = pd.read_parquet(dc.HERE / "d3dchk" / f"{tag}_geom.parquet").set_index("row")
c = np.intersect1d(Rg.index, Gg.index)
zb = np.load(dc.HERE / "d3d" / "val_zbal.npy")
zz = np.c_[df[[f"z_{k}" for k in dc.ZFE]].values[c], zb[c] / np.clip(0.5 * (df.R_out.values - df.R_in.values)[c], 0.3, 1)]
X = np.r_[np.c_[zz, Rg.loc[c, dc.GEOM].values], np.c_[zz, Gg.loc[c, dc.GEOM].values]].astype(np.float32)
y = np.r_[np.ones(len(c)), np.zeros(len(c))].astype(int); g = np.r_[df.shot.values[c], df.shot.values[c]]
params = json.loads((dc.HERE / "critic_params.json").read_text())
us = np.unique(g); fold = dict(zip(us, np.random.default_rng(100).permutation(len(us)) % 5)); fo = np.array([fold[s] for s in g])
names = [f"z:{k}" for k in dc.ZFE] + ["z:zbal/a"] + dc.GEOM
imp = np.zeros(len(names)); base = 0; rng = np.random.default_rng(0)
p = np.zeros(len(y)); ps = {j: np.zeros(len(y)) for j in range(len(names))}
for k in range(5):
    tr, te = fo != k, fo == k
    m, it = dc.fit_critic(X[tr], y[tr], g[tr], params, seed=0)
    p[te] = dc.predict_at(m, it, X[te])
    for j in range(len(names)):
        Xp = X[te].copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        ps[j][te] = dc.predict_at(m, it, Xp)
ll = log_loss(y, np.clip(p, 1e-4, 1 - 1e-4))
out = {n: float(log_loss(y, np.clip(ps[j], 1e-4, 1 - 1e-4)) - ll) for j, n in enumerate(names)}
print(tag, "gain", round(np.log(2) - ll, 4)); print({k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])[:12]})
