#!/usr/bin/env python3
"""DIII-D flux-geometry critic: real (z, EFIT map) vs generated (z, gsolve map) on the same held-out DIII-D frames.

Geometry features (normalised coordinates, from the flux map only; same code for both sides):
  LCFS rho(theta)/a Fourier modes 1-4 (cos, sin) about the LCFS box centre, kappa, delta_up, delta_low, squareness,
  (R_ax - Rgeo)/a, (Z_ax - Zc)/a, (Z_ax - 0.77 z_bal)/a, shift and elongation
  of the psi_N 0.3 / 0.6 contours, li, q at psi_N 0.5 / 0.95 (vacuum F = R B_T from the TF current).
Conditioning: the 14 transferable z features (teacher inputs) + z_bal/a.
Classifier: critic.fit_critic (critic_params.json capacity), 5-fold GroupKFold by shot, out-of-fold log loss pooled;
gain = ln2 - log loss, mean over 3 seeds (seed changes the fold assignment and the early-stopping split).

    python d3d_critic.py TAG [TAG ...] [--frames d3d/val.parquet] [--workers 16]   -> d3dchk/<TAG>_geom.parquet, geom_critic.csv
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ST = Path("/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter")
sys.path.insert(0, str(ST / "fusion_scoring"))
from contour import find_contours, grid_to_physical, encloses_point  # noqa: E402
from derive import derive_frame, magnetic_axis  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MK = np.load(ST / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GR, GZ = MK["grid_R"], MK["grid_Z"]
MC = MK["mask_coarse"].astype(bool)
MF = MC.astype(np.float64)
MACH = "DIII-D"
MU0 = 4e-7 * np.pi
ZFE = ["ip_frac", "ph_up", "ph_down", "rr_tau", "qstar", "fgw", "betape", "pk_te", "pk_ne", "pk_pe", "x50_te", "x50_ne",
       "x50_pe", "te_out"]
GEOM = [f"{c}{k}" for k in range(1, 5) for c in ("fc", "fs")] + \
    ["kappa", "d_up", "d_low", "sq", "ax_R", "ax_Z", "ax_Zbal", "s03", "k03", "s06", "k06",
     "li", "q50", "q95"]


def _bilin(f, rq, zq):
    fr = np.clip((rq - GR[0]) / (GR[1] - GR[0]), 0, len(GR) - 1.001)
    fz = np.clip((zq - GZ[0]) / (GZ[1] - GZ[0]), 0, len(GZ) - 1.001)
    i, j = fz.astype(int), fr.astype(int)
    a, b = fz - i, fr - j
    return f[i, j] * (1 - a) * (1 - b) + f[i + 1, j] * a * (1 - b) + f[i, j + 1] * (1 - a) * b + f[i + 1, j + 1] * a * b


def _surface(phi, level, iz, ir):
    """closed contour of phi at level enclosing grid cell (iz, ir), physical (R, Z)"""
    best = None
    for c in find_contours(phi, level):
        if len(c) > 8 and encloses_point(c, iz, ir):
            if best is None or len(c) < len(best):
                best = c
    return None if best is None else grid_to_physical(best, GR, GZ)


def geom(args):
    psi, zbal, F = args          # psi already in the machine's reference orientation; F = R B_T (T m)
    out = {k: np.nan for k in GEOM}
    try:
        psi = np.asarray(psi, np.float64)
        C = extract_lcfs(psi, GR, GZ, MACH, MC, MF, n_points=512)
        if C is None or len(C) < 16:
            return out
        d = derive_frame(psi, GR, GZ, MACH, MC, MF, contour=C)
        Ra, Za, iz, ir = magnetic_axis(psi, GR, GZ, MACH, MC)
        R, Z = C[:, 0], C[:, 1]
        a = 0.5 * (R.max() - R.min()); Rg = 0.5 * (R.max() + R.min()); Zc = 0.5 * (Z.max() + Z.min())
        th = np.arctan2(Z - Zc, R - Rg); rho = np.hypot(R - Rg, Z - Zc) / a
        o = np.argsort(th); tu = np.linspace(-np.pi, np.pi, 256, endpoint=False)
        ru = np.interp(tu, th[o], rho[o], period=2 * np.pi)
        for k in range(1, 5):
            out[f"fc{k}"] = float(np.mean(ru * np.cos(k * tu)) * 2); out[f"fs{k}"] = float(np.mean(ru * np.sin(k * tu)) * 2)
        out["kappa"], out["d_up"], out["d_low"] = d["kappa"], d["tri_top"], d["tri_bot"]
        # squareness: per quadrant, (d_C - d_ell) / (d_box - d_ell) along the quadrant diagonal, averaged
        ku, kl = (Z.max() - Zc) / a, (Zc - Z.min()) / a
        sq = []
        for sx in (1, -1):
            for sz, kk in ((1, ku), (-1, kl)):
                ang = np.arctan2(sz * kk, sx)
                dc = np.interp(ang, tu, ru, period=2 * np.pi)
                dbox = np.hypot(1, kk); dell = dbox / np.sqrt(2)
                sq.append((dc - dell) / (dbox - dell))
        out["sq"] = float(np.mean(sq))
        out["ax_R"], out["ax_Z"], out["ax_Zbal"] = (Ra - Rg) / a, (Za - Zc) / a, (Za - 0.77 * zbal) / a
        out["li"] = d["li"]
        # flux levels: phi with the axis a maximum
        phi = -psi                       # DIII-D reference orientation: axis is a minimum of psi (AXIS_SIGN -1)
        p_ax = float(_bilin(phi, np.array([Ra]), np.array([Za]))[0]); p_b = float(np.mean(_bilin(phi, R, Z)))
        if not p_ax > p_b:
            return out
        gz, gr = np.gradient(psi, GZ, GR)
        for pn, sk, kk in ((0.3, "s03", "k03"), (0.6, "s06", "k06")):
            S = _surface(phi, p_ax - pn * (p_ax - p_b), iz, ir)
            if S is not None:
                out[sk] = (0.5 * (S[:, 0].max() + S[:, 0].min()) - Rg) / a
                out[kk] = (S[:, 1].max() - S[:, 1].min()) / max(S[:, 0].max() - S[:, 0].min(), 1e-6)
        for pn, qk in ((0.5, "q50"), (0.95, "q95")):
            S = _surface(phi, p_ax - pn * (p_ax - p_b), iz, ir)
            if S is not None and np.isfinite(F):
                Rm, Zm = 0.5 * (S[1:, 0] + S[:-1, 0]), 0.5 * (S[1:, 1] + S[:-1, 1])
                dl = np.hypot(np.diff(S[:, 0]), np.diff(S[:, 1]))
                Bp = np.hypot(_bilin(gr, Rm, Zm), _bilin(gz, Rm, Zm)) / Rm
                out[qk] = float(abs(F) / (2 * np.pi) * np.sum(dl / (Rm ** 2 * np.maximum(Bp, 1e-6))))
    except Exception:  # noqa: BLE001
        pass
    return out


def tf_F(tf_kA):
    """R B_T from the DIII-D toroidal coil current (144 turns, as d3d_table)"""
    return MU0 * 144 * np.abs(tf_kA) * 1e3 / (2 * np.pi)


def real_geom(frames, workers):
    f = HERE / "d3dchk" / "efit_geom.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = pd.read_parquet(HERE / "d3d" / frames)
    P = np.load(HERE / "d3d" / "val_psi.npy", mmap_mode="r")
    zb = np.load(HERE / "d3d" / "val_zbal.npy")
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(geom, [(P[i], zb[i], tf_F(df.tf.values[i])) for i in range(len(df))], chunksize=8))
    G = pd.DataFrame(out); G["row"] = np.arange(len(df))
    G.to_parquet(f)
    return G


def gen_geom(tag, workers):
    f = HERE / "d3dchk" / f"{tag}_geom.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = pd.read_parquet(HERE / "d3d" / "val.parquet")
    key = {(s, fr): i for i, (s, fr) in enumerate(zip(df.shot.values, df.frame.values))}
    rows, args = [], []
    zb = np.load(HERE / "d3d" / "val_zbal.npy")
    for fn in sorted(glob.glob(str(HERE / "d3dchk" / f"{tag}_[0-9].npz"))):
        z = dict(np.load(fn, allow_pickle=True))
        for j in np.nonzero(z["rank"] == 0)[0]:
            i = key[(z["shot"][j], int(z["frame"][j]))]
            rows.append(i)
            args.append((-np.asarray(z["psi"][j], np.float64), zb[i], tf_F(df.tf.values[i])))   # gsolve sign -> EFIT sign
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(geom, args, chunksize=8))
    G = pd.DataFrame(out); G["row"] = rows
    G.to_parquet(f)
    return G


def _hgb(params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)


def fit_critic(X, y, g, params, seed=0, es_frac=0.2):
    """copy of critic.fit_critic (critic.py cannot be imported next to fusion_scoring's common.py)"""
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


def critic_gain(Rg, Gg, df, seeds=(0, 1, 2), feats=GEOM, zcols=None):
    from sklearn.metrics import log_loss
    cr = sys.modules[__name__]
    params = json.loads((HERE / "critic_params.json").read_text())
    zcols = zcols if zcols is not None else [f"z_{k}" for k in ZFE]
    zb = np.load(HERE / "d3d" / "val_zbal.npy")
    common = np.intersect1d(Rg.row.values, Gg.row.values)
    R_ = Rg.set_index("row").loc[common]; G_ = Gg.set_index("row").loc[common]
    Z = df[zcols].values[common]
    zz = np.c_[Z, zb[common] / np.clip(0.5 * (df.R_out.values - df.R_in.values)[common], 0.3, 1)]
    X = np.r_[np.c_[zz, R_[feats].values], np.c_[zz, G_[feats].values]].astype(np.float32)
    y = np.r_[np.ones(len(common)), np.zeros(len(common))].astype(int)
    g = np.r_[df.shot.values[common], df.shot.values[common]]
    us = np.unique(g)
    res = []
    for sd in seeds:
        fold = dict(zip(us, np.random.default_rng(100 + sd).permutation(len(us)) % 5))
        fo = np.array([fold[s] for s in g])
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fo != k, fo == k
            m, it = cr.fit_critic(X[tr], y[tr], g[tr], params, seed=sd)
            p[te] = cr.predict_at(m, it, X[te])
        res.append(float(np.log(2) - log_loss(y, np.clip(p, 1e-4, 1 - 1e-4))))
    return dict(gain=float(np.mean(res)), gain_sd_seed=float(np.std(res)), gains=res, n_pairs=int(len(common)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--frames", default="val.parquet")
    ap.add_argument("--out", default="geom_critic.csv")
    a = ap.parse_args()
    df = pd.read_parquet(HERE / "d3d" / a.frames)      # conditioning z (val.parquet oracle edges / val_th.parquet Thomson edges)
    Rg = real_geom("val.parquet", a.workers)
    rows = []
    for t in a.tags:
        Gg = gen_geom(t, a.workers)
        r = critic_gain(Rg, Gg, df)
        # per-feature location/spread of generated vs EFIT on the common frames (diagnostic)
        common = np.intersect1d(Rg.row.values, Gg.row.values)
        dR, dG = Rg.set_index("row").loc[common], Gg.set_index("row").loc[common]
        sd_ratio = {k: float(np.nanstd(dG[k]) / max(np.nanstd(dR[k]), 1e-9)) for k in GEOM}
        r.update(tag=t, sd_ratio=json.dumps({k: round(v, 2) for k, v in sd_ratio.items()}))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        rows.append(r)
    out = HERE / a.out
    pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not out.exists())


if __name__ == "__main__":
    main()


def set_grid(R, Z, mask):
    """use geom() on another machine's grid (e.g. MAST gsolve maps, passed as -psi_gsolve; the DIII-D EFIT orientation
    conventions stay, only grid and LCFS search mask change)"""
    global GR, GZ, MC, MF
    GR, GZ = np.asarray(R, float), np.asarray(Z, float)
    MC = np.asarray(mask, bool); MF = MC.astype(np.float64)
