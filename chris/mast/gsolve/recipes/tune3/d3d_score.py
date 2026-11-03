#!/usr/bin/env python3
"""D_label: how far a recipe's generated labels are from the real DIII-D EFIT labels on held-out DIII-D frames.

    python d3d_score.py --tag T [--files d3dchk/T_*.npz] [--workers 12] [--common tagA,tagB,...]

For each frame the rank-0 kept equilibrium is the recipe's "label". Per label: R2 (pooled over frames), bias
(median of generated - EFIT), spread (sd of the difference), correlation. Flux maps: pooled R2psi and the scorer's
boundary distance; the challenge composite S is computed with the vendored scorer functions on the same frames
(DIII-D ground truth is allowed: our own held-out split of diii_d_train). Writes d3dchk/<T>_frames.parquet and
d3dchk/<T>_score.json.
"""
import argparse
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ST = Path("/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter")
sys.path.insert(0, str(ST / "fusion_scoring"))
from common import AXIS_SIGN, CONS_SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402

HERE = Path(__file__).resolve().parent
MK = np.load(ST / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GR, GZ = MK["grid_R"], MK["grid_Z"]
MC = MK["mask_coarse"].astype(bool)
MF = MC.astype(np.float64)
MACH = "DIII-D"


def _ref_one(psi):
    c = extract_lcfs(psi, GR, GZ, MACH, MC, MF, n_points=512)
    v = derive_frame(psi, GR, GZ, MACH, MC, MF, contour=c)
    return c, [v[k] for k in CONS_SCALARS]


def reference(workers):
    f = HERE / "d3d" / "val_ref.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    P = np.load(HERE / "d3d" / "val_psi.npy").astype(np.float64)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_ref_one, P, chunksize=8))
    C = [o[0] for o in out]
    cons = np.array([o[1] for o in out], np.float64)
    rgeo = float(np.mean([major_radius(c) for c in C if c is not None]))
    ref = dict(contours=C, cons=cons, rgeo=rgeo)
    f.write_bytes(pickle.dumps(ref))
    return ref


def _pred_one(args):
    psi, axis_sign, refc = args
    ex, _ = extract_lcfs_with_sign(psi, GR, GZ, MACH, MC, MF, n_iter=22, axis_sign=axis_sign)
    d = np.nan
    if refc is not None:
        d = 1.0 if ex is None else symmetric_hausdorff(ex, np.asarray(refc, np.float64))
    v = derive_frame(psi, GR, GZ, MACH, MC, MF, contour=ex, axis_sign=axis_sign)
    return d, [v[k] for k in CONS_SCALARS]


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if ok.sum() > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--files", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    df = pd.read_parquet(HERE / "d3d" / "val.parquet")
    ref = reference(a.workers)
    P = np.load(HERE / "d3d" / "val_psi.npy", mmap_mode="r")
    files = sorted(glob.glob(a.files or str(HERE / "d3dchk" / f"{a.tag}_[0-9].npz")))
    rows, psis = [], []
    for fn in files:
        z = dict(np.load(fn, allow_pickle=True))
        # map shard rows back to val.parquet rows by (shot, frame)
        key = {(s, f): i for i, (s, f) in enumerate(zip(df.shot.values, df.frame.values))}
        for j in np.nonzero(z["rank"] == 0)[0]:
            i = key[(z["shot"][j], int(z["frame"][j]))]
            rows.append(dict(row=i, q95=z["gs_q95"][j], betaN=z["gs_betaN"][j], li_gs=z["gs_li"][j], K=z["gen_K"][j],
                             an=z["gen_an"][j], am=z["gen_am"][j], li_t=z["gen_li_t"][j], zc=z["gen_zc"][j], z_bal=z["z_bal"][j]))
            psis.append(z["psi"][j].astype(np.float64))
    n_tot = len(df)
    G = pd.DataFrame(rows)
    psis = np.stack(psis)
    gt = P[G.row.values].astype(np.float64)
    mean_psi = float(np.mean(P))
    tot = float(np.sum((gt - mean_psi) ** 2))
    ss = {s: float(np.sum((gt - s * psis) ** 2)) for s in (1.0, -1.0)}
    sgn = min(ss, key=ss.get)
    r2psi = 1 - ss[sgn] / tot
    # offset-corrected: the absolute flux level carries the unknown E-coil / vessel flux offset, the same for every
    # recipe; shift each predicted map by the frame's mean difference (oracle offset) and re-pool
    shp = sgn * psis + (gt.mean(axis=(1, 2)) - (sgn * psis).mean(axis=(1, 2)))[:, None, None]
    r2psi_off = 1 - float(np.sum((gt - shp) ** 2)) / tot
    r2psi_within = 1 - float(np.sum((gt - shp) ** 2)) / float(np.sum((gt - gt.mean(axis=(1, 2), keepdims=True)) ** 2))
    # E-coil calibration (ecoil_cal.json, fitted on TRAIN frames): predicted map + k * I_ec + b (a per-frame constant)
    cf = HERE / "ecoil_cal.json"
    if cf.exists():
        cal = json.loads(cf.read_text())
        add = cal["k"] * (df.c_ECOILA.values * df.sign.values)[G.row.values] + cal["b"]
    else:
        add = np.zeros(len(G))
    shc = sgn * psis + add[:, None, None]
    r2psi_cal = 1 - float(np.sum((gt - shc) ** 2)) / tot
    G_ssr_cal = np.sum((gt - shc) ** 2, axis=(1, 2))
    axis_sign = AXIS_SIGN[MACH] * sgn
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_pred_one, [(sgn * p, AXIS_SIGN[MACH], ref["contours"][i]) for p, i in zip(psis, G.row.values)], chunksize=4))
    G["dlcfs"] = [min(1.0, o[0] / ref["rgeo"]) if np.isfinite(o[0]) else np.nan for o in out]
    G["dlcfs_m"] = [o[0] for o in out]
    for k, name in enumerate(CONS_SCALARS):
        G[f"p_{name}"] = [o[1][k] for o in out]
        G[f"t_{name}"] = ref["cons"][G.row.values, k]
    for k in ("q95", "betaN", "li", "R_ax", "Z_ax", "kappa", "K", "phase", "shot"):
        G[f"e_{k}"] = df[k].values[G.row.values]
    G["r2psi_frame"] = [1 - np.sum((g - sgn * p) ** 2) / np.sum((g - g.mean()) ** 2) for g, p in zip(gt, psis)]
    G["ssr_off"] = np.sum((gt - shp) ** 2, axis=(1, 2))
    G["sst"] = np.sum((gt - mean_psi) ** 2, axis=(1, 2))
    G["ssr_cal"] = G_ssr_cal
    G.to_parquet(HERE / "d3dchk" / f"{a.tag}_frames.parquet")
    # summary
    lab = {"q95": ("q95", "e_q95"), "betaN": ("betaN", "e_betaN"), "li (scorer, from psi)": ("p_li", "t_li"),
           "li (gsolve) vs EFIT li": ("li_gs", "e_li"), "R_axis": ("p_R_axis", "t_R_axis"), "Z_axis": ("p_Z_axis", "t_Z_axis"),
           "kappa": ("p_kappa", "t_kappa"), "tri_top": ("p_tri_top", "t_tri_top"), "tri_bot": ("p_tri_bot", "t_tri_bot"),
           "volume": ("p_volume", "t_volume")}
    summ = {}
    for nm, (p, t) in lab.items():
        d = (G[p] - G[t]).values
        ok = np.isfinite(d)
        summ[nm] = dict(r2=r2(G[t].values, G[p].values), bias=float(np.nanmedian(d)), spread=float(np.nanstd(d)),
                        corr=float(np.corrcoef(G[t].values[ok], G[p].values[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                        gen_median=float(np.nanmedian(G[p])), efit_median=float(np.nanmedian(G[t])))
    # challenge-style composite on these frames (fill failures with the EFIT mean, as the scorer does)
    def r2fill(t, p):
        t, p = np.asarray(t, float), np.asarray(p, float)
        ok = np.isfinite(t)
        p = np.where(np.isfinite(p), p, np.nanmean(t))
        return 1 - np.sum((t[ok] - p[ok]) ** 2) / np.sum((t[ok] - t[ok].mean()) ** 2)
    r2qb = np.mean([r2fill(G.e_q95, G.q95), r2fill(G.e_betaN, G.betaN)])
    cons = np.mean([max(0, r2fill(G[f"t_{k}"], G[f"p_{k}"])) for k in CONS_SCALARS])
    dl = float(G.groupby("e_shot").dlcfs.mean().mean())
    S = 0.55 * max(0, r2psi) + 0.15 * max(0, r2qb) + 0.10 * (1 - min(1, dl)) + 0.20 * cons
    S_off = S + 0.55 * (max(0, r2psi_off) - max(0, r2psi))
    S_cal = S + 0.55 * (max(0, r2psi_cal) - max(0, r2psi))
    res = dict(tag=a.tag, n_frames=n_tot, n_kept=len(G), kept_frac=len(G) / n_tot, psi_sign=sgn, r2psi=r2psi,
               r2psi_off=r2psi_off, r2psi_within=r2psi_within, S_off=float(S_off), r2psi_cal=r2psi_cal, S_cal=float(S_cal),
               r2psi_frame_median=float(np.median(G.r2psi_frame)), dlcfs=dl, dlcfs_cm_median=float(100 * np.nanmedian(G.dlcfs_m)),
               r2qb=float(r2qb), consistency=float(cons), S=float(S), labels=summ)
    (HERE / "d3dchk" / f"{a.tag}_score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "labels"}))
    print(pd.DataFrame(summ).T.round(3).to_string())


if __name__ == "__main__":
    main()
