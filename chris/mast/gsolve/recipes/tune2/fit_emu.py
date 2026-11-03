#!/usr/bin/env python3
"""gsolve's own li response (no labels involved): li_gsolve ~ g(log an, log am, log K, z) per machine, fitted on
generator outputs (MAST: earlier critic-tuning runs in ../tune/runs, read-only; DIII-D: the recipe-check runs).
Used to invert li* -> an. Monotone increasing in an. Adds 'emu' to cond_model.pkl."""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
C = pickle.loads((HERE / "cond_model.pkl").read_bytes())
FE = C["model"]["feats"]


def fit(df, tag):
    X = np.c_[np.log(df.gen_an), np.log(df.gen_am), np.log(df.gen_K), df[[f"z_{k}" for k in FE]].values]
    y = df.li.values
    mono = [1, 0, 0] + [0] * len(FE)
    mk = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30, monotonic_cst=mono)  # noqa: E731
    pr = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, df.shot.values):
        pr[b] = mk().fit(X[a], y[a]).predict(X[b])
    r2 = 1 - np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{tag}: {len(y)} equilibria, CV R2 {r2:.3f}, resid sd {np.std(y - pr):.3f}, li 5/50/95 {np.percentile(y, [5, 50, 95]).round(2)}")
    return mk().fit(X, y), dict(n=len(y), cv_r2=float(r2), resid_sd=float(np.std(y - pr)))


def main():
    emu, rep = C.get("emu", {}), C.get("emu_rep", {})
    which = sys.argv[1:] or ["mast", "d3d"]
    if "mast" in which:
        rz = pd.read_parquet(HERE / "mast" / "real_z.parquet")
        rows = []
        for d in sorted(glob.glob(str(HERE.parent / "tune" / "runs" / "*"))):
            try:
                k = json.load(open(d + "/knobs.json"))
            except Exception:  # noqa: BLE001
                continue
            if k.get("w_te", 0) != 0 or k.get("kp_hi", 0) != 0:
                continue
            for f in sorted(glob.glob(d + "/gen_*.npz")):
                z = np.load(f)
                rows.append(pd.DataFrame(dict(shot=z["shot"], frame=z["frame"], li=z["li"], gen_an=z["gen_an"], gen_am=z["gen_am"], gen_K=z["gen_K"])))
            if len(rows) > 200:
                break
        D = pd.concat(rows, ignore_index=True).merge(rz, on=["shot", "frame"])
        D = D[np.isfinite(D.li) & np.isfinite(D[[f"z_{k}" for k in FE]]).all(1)]
        D = D.sample(min(len(D), 150000), random_state=0)
        emu["mast"], rep["mast"] = fit(D, "mast")
    if "d3d" in which:
        va = pd.read_parquet(HERE / "d3d" / "val.parquet")
        rows = []
        for f in sorted(glob.glob(str(HERE / "d3dchk" / "def_*.npz")) + sorted(glob.glob(str(HERE / "d3dchk" / "x8_*.npz")))):
            z = np.load(f, allow_pickle=True)
            rows.append(pd.DataFrame(dict(shot=z["shot"], frame=z["frame"], li=z["gs_li"], gen_an=z["gen_an"], gen_am=z["gen_am"], gen_K=z["gen_K"])))
        D = pd.concat(rows, ignore_index=True).merge(va[["shot", "frame"] + [f"z_{k}" for k in FE]], on=["shot", "frame"])
        D = D[np.isfinite(D.li)]
        emu["d3d"], rep["d3d"] = fit(D, "d3d")
    C["emu"], C["emu_rep"] = emu, rep
    (HERE / "cond_model.pkl").write_bytes(pickle.dumps(C))


if __name__ == "__main__":
    main()
