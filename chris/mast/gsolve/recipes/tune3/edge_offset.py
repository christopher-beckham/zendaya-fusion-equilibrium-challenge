#!/usr/bin/env python3
"""Systematic LCFS offset of a replay vs EFIT: generated minus EFIT Rmin/Rmax/Zmin/Zmax/a (cm), plus the Thomson-edge
target vs EFIT, on frames the recipe kept.   edge_offset.py TAG [frames parquet]"""
import glob, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import d3d_critic as dc
from lcfs import extract_lcfs

def ext(p):
    C = extract_lcfs(np.asarray(p, np.float64), dc.GR, dc.GZ, dc.MACH, dc.MC, dc.MF, n_points=256)
    return [np.nan] * 4 if C is None else [C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max()]

if __name__ == "__main__":
    tag = sys.argv[1]; fr = sys.argv[2] if len(sys.argv) > 2 else "d3d/val_th.parquet"
    df = pd.read_parquet("d3d/val.parquet"); dt = pd.read_parquet(fr)
    ref = pickle.loads(open("d3d/val_ref.pkl", "rb").read())
    key = {(s, f): i for i, (s, f) in enumerate(zip(df.shot.values, df.frame.values))}
    rows, ps = [], []
    for fn in sorted(glob.glob(f"d3dchk/{tag}_[0-9].npz")):
        z = dict(np.load(fn, allow_pickle=True))
        for j in np.nonzero(z["rank"] == 0)[0]:
            rows.append(key[(z["shot"][j], int(z["frame"][j]))]); ps.append(-np.asarray(z["psi"][j]))
    with ProcessPoolExecutor(8) as ex:
        G = np.array(list(ex.map(ext, ps, chunksize=16)))
    E = np.array([[c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()] if c is not None else [np.nan] * 4 for c in ref["contours"]])[rows]
    d = 100 * (G - E)
    for k, nm in enumerate(["Rmin", "Rmax", "Zmin", "Zmax"]):
        print(f"{tag} {nm}: gen-EFIT cm median {np.nanmedian(d[:, k]):+.2f} p10/p90 {np.nanpercentile(d[:, k], 10):+.2f}/{np.nanpercentile(d[:, k], 90):+.2f}")
    a = 100 * ((G[:, 1] - G[:, 0]) - (E[:, 1] - E[:, 0])) / 2
    print(f"{tag} a: gen-EFIT cm median {np.nanmedian(a):+.2f}; target R_out - EFIT Rmax median {100*np.nanmedian(dt.R_out.values[rows] - E[:, 1]):+.2f}, "
          f"R_in - EFIT Rmin {100*np.nanmedian(dt.R_in.values[rows] - E[:, 0]):+.2f}; EFIT a median {100*np.nanmedian((E[:,1]-E[:,0])/2):.1f}")
