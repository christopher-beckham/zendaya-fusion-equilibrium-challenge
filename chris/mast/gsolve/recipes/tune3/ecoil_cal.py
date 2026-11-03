#!/usr/bin/env python3
"""E-coil calibration of the DIII-D raw flux offset (TRAIN frames only).
  ecoil_cal.py subset   -> d3d/train_cal.parquet (120 train shots x <= 4 frames) + d3d/train_cal_idx.npy
  ecoil_cal.py fit TAG  -> ecoil_cal.json: offset = mean(EFIT) - mean(-gsolve) per frame ~ k * I_ec + b,
                           I_ec = c_ECOILA * sign (kA); shot-grouped 5-fold CV R2."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
if sys.argv[1] == "subset":
    tr = pd.read_parquet(HERE / "d3d" / "train3.parquet"); tr["i3"] = np.arange(len(tr))
    sh = np.random.default_rng(7).choice(sorted(tr.shot.unique()), 120, replace=False)
    sub = pd.concat([g.iloc[np.unique(np.linspace(0, len(g) - 1, 4).round().astype(int))] for s, g in tr[tr.shot.isin(sh)].groupby("shot")])
    sub = sub.reset_index(drop=True); sub.to_parquet(HERE / "d3d" / "train_cal.parquet"); print(len(sub), "frames")
else:
    tag = sys.argv[2]
    sub = pd.read_parquet(HERE / "d3d" / "train_cal.parquet")
    P = np.load(HERE / "d3d" / "train3_psi.npy", mmap_mode="r")
    key = {(s, f): i for i, (s, f) in enumerate(zip(sub.shot.values, sub.frame.values))}
    off, iec, shots = [], [], []
    for fn in sorted(glob.glob(str(HERE / "d3dchk" / f"{tag}_[0-9].npz"))):
        z = np.load(fn, allow_pickle=True)
        for j in np.nonzero(z["rank"] == 0)[0]:
            i = key[(z["shot"][j], int(z["frame"][j]))]
            gt = np.asarray(P[sub.i3.values[i]], np.float64)
            off.append(gt.mean() - (-np.asarray(z["psi"][j], np.float64)).mean())
            iec.append(sub.c_ECOILA.values[i] * sub.sign.values[i]); shots.append(sub.shot.values[i])
    off, iec, shots = map(np.asarray, (off, iec, shots))
    A = np.c_[iec, np.ones(len(iec))]
    k, b = np.linalg.lstsq(A, off, rcond=None)[0]
    us = np.unique(shots); fold = dict(zip(us, np.random.default_rng(0).permutation(len(us)) % 5)); fo = np.array([fold[s] for s in shots])
    pr = np.zeros(len(off))
    for f in range(5):
        c = np.linalg.lstsq(A[fo != f], off[fo != f], rcond=None)[0]; pr[fo == f] = A[fo == f] @ c
    r2 = 1 - np.sum((off - pr) ** 2) / np.sum((off - off.mean()) ** 2)
    rep = dict(k=float(k), b=float(b), cv_r2=float(r2), resid_sd=float(np.std(off - pr)), n=int(len(off)), shots=int(len(us)),
               off_sd=float(off.std()), iec_range=[float(iec.min()), float(iec.max())])
    print(json.dumps(rep, indent=1)); (HERE / "ecoil_cal.json").write_text(json.dumps(rep, indent=1))
