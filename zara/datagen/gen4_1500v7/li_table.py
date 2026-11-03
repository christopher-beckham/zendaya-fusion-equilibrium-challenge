"""Recipe v7: per-frame li target from Chris's DIII-D-taught model, applied to real MAST inputs.

    python li_table.py <out.npz>

Reads, without changing them, Chris's files in /fsx/chris.beckham/gsolve_data/tune2/:
  * mast/real_z.parquet   input features z of every real MAST frame (zfeat.py, from magnetics_* / thomson_* input columns
                          and efit_times only; mast_z.py, gen_param2.read_shot)
  * cond_model.pkl        li = f(z) and log K = g(z), gradient-boosted trees fitted on 699 DIII-D training shots
                          (DIII-D labels are training data; fit_prior.py)
and writes, per real frame (shot file, index into efit_times): the predicted li* and log K*, the li* smoothed over
+-2 neighbouring frames (median), and the residual covariance of the fit. No MAST label is read.
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

SRC = "/fsx/chris.beckham/gsolve_data/tune2"
sys.path.insert(0, SRC)
import zfeat  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(out):
    cp = pickle.loads(open(os.path.join(SRC, "cond_model.pkl"), "rb").read())
    M = cp["model"]
    Z = pd.read_parquet(os.path.join(SRC, "mast/real_z.parquet"))
    Zfull = Z[[f"z_{k}" for k in zfeat.FEATS]].to_numpy(np.float64)
    idx = [zfeat.FEATS.index(k) for k in M["feats"]]
    X = (np.clip(Zfull[:, idx], M["lo"], M["hi"]) - np.array(M["mu"])) / np.array(M["sd"])
    ok = np.isfinite(X).all(1)
    li = np.full(len(X), np.nan)
    lk = np.full(len(X), np.nan)
    li[ok] = cp["models"]["li"]["gbm"].predict(X[ok])
    lk[ok] = cp["models"]["logK"]["gbm"].predict(X[ok])
    Z = Z.assign(li=li, lk=lk)
    Z["file"] = "mast_shot_" + Z["shot"].astype(str) + ".parquet"
    Z = Z.sort_values(["file", "frame"]).reset_index(drop=True)
    sm = np.full(len(Z), np.nan)
    for _, g in Z.groupby("file", sort=False):
        fr, v = g["frame"].to_numpy(), g["li"].to_numpy()
        for j, i in enumerate(g.index):
            m = (np.abs(fr - fr[j]) <= 2) & np.isfinite(v)
            if m.any():
                sm[i] = float(np.median(v[m]))
    Z["li_sm"] = sm
    cov = np.array(M["cov_gbm"], float)
    np.savez(out, fname=Z["file"].to_numpy().astype(str), frame=Z["frame"].to_numpy().astype(np.int32),
             li=Z["li"].to_numpy(), li_sm=Z["li_sm"].to_numpy(), lk=Z["lk"].to_numpy(), cov_gbm=cov)
    src = {p: sha(os.path.join(SRC, p)) for p in ("cond_model.pkl", "cond_model.json", "mast/real_z.parquet", "zfeat.py",
                                                  "mast_z.py", "fit_prior.py", "gen_param2.py", "cond_prior.py")}
    d = np.diff(Z["li"].to_numpy())
    same = (Z["file"].to_numpy()[1:] == Z["file"].to_numpy()[:-1]) & (np.diff(Z["frame"].to_numpy()) == 1)
    ds = np.diff(Z["li_sm"].to_numpy())
    rep = dict(n_frames=int(len(Z)), n_shots=int(Z["file"].nunique()), n_ok=int(ok.sum()),
               li_pct=np.nanpercentile(li, [5, 25, 50, 75, 95]).round(3).tolist(),
               K_pct=np.exp(np.nanpercentile(lk, [5, 25, 50, 75, 95])).round(3).tolist(),
               frame_to_frame_sd_li=float(np.nanstd(d[same])), frame_to_frame_sd_li_sm=float(np.nanstd(ds[same])),
               resid_sd_li=float(np.sqrt(cov[0, 0])), resid_sd_logK=float(np.sqrt(cov[1, 1])),
               resid_corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), sources_sha256=src)
    json.dump(rep, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
