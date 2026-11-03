#!/usr/bin/env python3
"""Range check: geometry features of generated MAST maps (cond1000 part0 sample, limiter mask) vs DIII-D EFIT."""
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import d3d_critic as dc
import gsolve
LIM = "/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn/geom_mask/mast_mask_limiter.npz"
m = gsolve.load_machine("mast", solenoid_scale=0.5)
L = np.load(LIM)
def init():
    dc.set_grid(L["r"], L["z"], L["mask"])
def one(a):
    return dc.geom(a)
if __name__ == "__main__":
    d = np.load("/fsx/chris.beckham/gsolve_data/cond1000/cond1000_part0.npz")
    idx = np.arange(0, len(d["shot"]), 25)
    args = [(-np.asarray(d["psi"][i], np.float64), 0.0, float(m.fvac(d["tf"][i]))) for i in idx]
    with ProcessPoolExecutor(8, initializer=init) as ex:
        G = pd.DataFrame(list(ex.map(one, args, chunksize=8)))
    E = pd.read_parquet("d3dchk/efit_geom.parquet")
    rows = []
    for k in dc.GEOM:
        lo, hi = np.nanpercentile(E[k], [1, 99]); v = G[k]
        rows.append(dict(f=k, d3d_lo=lo, d3d_hi=hi, mast_med=np.nanmedian(v), mast_p5=np.nanpercentile(v, 5), mast_p95=np.nanpercentile(v, 95),
                         outside=float(np.mean((v < lo) | (v > hi))), nan=float(v.isna().mean())))
    print(len(G), "MAST frames"); print(pd.DataFrame(rows).round(3).to_string())
