#!/usr/bin/env python3
"""DIII-D recipe check: run a gsolve recipe (gen_param knobs, optionally the conditional prior) on DIII-D held-out
frames with gsolve's DIII-D machine description, and keep what the MAST generator would keep.

Differences from the MAST generator, forced by the DIII-D inputs (documented in HANDOFF.md):
  - Thomson is a vertical chord + tangential midplane channels: channels are put on flux with their (R, Z) positions.
  - The midplane plasma edges that MAST's Thomson provides (vessel-field target, start shape, selection) are taken
    from the EFIT midplane edges (an "oracle edge": stands in for the Thomson edges of MAST).
  - The pass-1 pressure table uses the Thomson profile on the EFIT-mapped x grid (MAST: x from the Thomson
    peak and edges); pass 2 remaps on the solved surfaces exactly as on MAST.
Everything that sets the hidden physics (K, am, an, w, kp, zc draws, phase factors, selection rule) is the recipe's.

    python gen_d3d.py --frames d3d/val.parquet --knobs '{...}' --out d3dchk/<tag>.npz [--cond cond_model.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import gsolve
from gsolve.forward import FwdConfig, forward_solve, start_current

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_param2 as gp  # noqa: E402   (copy of the MAST generator: DEFAULTS, pressure_table, te_shape, midplane_edges, Draws)
import cond_prior  # noqa: E402
import zfeat  # noqa: E402

E = gp.E
gp.RLINE = np.linspace(0.9, 2.5, 321)          # DIII-D midplane line for gen_param.midplane_edges
NX = gp.NX
XG, RG = gp.XG, gp.RG
COILS = ["F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B", "F6A", "F6B", "F7A", "F7B", "F8A", "F8B",
         "F9A", "F9B", "ECOILA"]


def psi_at(psi2d, R, Z, rq, zq):
    """bilinear psi at points"""
    fr = np.clip((rq - R[0]) / (R[1] - R[0]), 0, len(R) - 1.001)
    fz = np.clip((zq - Z[0]) / (Z[1] - Z[0]), 0, len(Z) - 1.001)
    i, j = fz.astype(int), fr.astype(int)
    a, b = fz - i, fr - j
    return (psi2d[i, j] * (1 - a) * (1 - b) + psi2d[i + 1, j] * a * (1 - b) + psi2d[i, j + 1] * (1 - a) * b
            + psi2d[i + 1, j + 1] * a * b)


def flux_tables2(pch, psax, psb, te, ne, k3=True, mono=True):
    """gen_param.flux_tables with the channel flux given directly (2-D channel positions)."""
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(pch)
    if ok.sum() < 12 or not psax > psb:
        return None
    rho = np.sqrt(np.clip((psax - pch[ok]) / (psax - psb), 0, None))
    h = 0.5 * (RG[1] - RG[0])
    b = np.digitize(rho, np.concatenate([RG - h, [RG[-1] + h]])) - 1
    T = np.array([np.median(te[ok][b == k]) if (b == k).any() else np.nan for k in range(len(RG))])
    N = np.array([np.median(ne[ok][b == k]) if (b == k).any() else np.nan for k in range(len(RG))])
    f = np.isfinite(T)
    if f.sum() < 6 or not (RG[f] < 0.6).any():
        return None
    if k3:
        kk = np.array([0.25, 0.5, 0.25])
        T, N = (np.convolve(np.pad(np.interp(RG, RG[f], v[f]), 1, mode="edge"), kk, mode="valid") for v in (T, N))
    else:
        T, N = (np.interp(RG, RG[f], v[f]) for v in (T, N))
    return (np.maximum.accumulate(T[::-1])[::-1] if mono else T), N


def th_mismatch2(pch, psax, psb, te, ne, groups, tab):
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(pch)
    if ok.sum() < 12 or not psax > psb:
        return np.nan
    d = 0.0
    for q, t in ((te, tab[0]), (ne, tab[1])):
        res, peak = [], 0.0
        for g in groups:
            g = g[ok[g]]
            if len(g) < 5:
                continue
            sm = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(q[g], 2, mode="edge"), 5), axis=1)
            rho = np.sqrt(np.clip((psax - pch[g]) / (psax - psb), 0, None))
            res.append(np.abs(np.interp(rho, RG, t) - sm)); peak = max(peak, sm.max())
        if not res:
            return np.nan
        d += np.median(np.concatenate(res)) / max(peak, 1e-6)
    return float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knobs", default=None)
    ap.add_argument("--cond", default=None, help="conditional prior model JSON (cond_prior.py)")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--n-iter", type=int, default=160)
    ap.add_argument("--pass2-iter", type=int, default=120)
    ap.add_argument("--tol", type=float, default=3e-3)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    kn = dict(gp.DEFAULTS)
    if a.knobs:
        kn.update(json.loads(Path(a.knobs).read_text()) if Path(a.knobs).exists() else json.loads(a.knobs))
    df = pd.read_parquet(a.frames)
    si, sn = (int(v) for v in a.shard.split("/"))
    shots = sorted(df.shot.unique())[si::sn]
    df = df[df.shot.isin(shots)].reset_index(drop=True)
    t0 = time.time()
    m = gsolve.load_machine("diii-d")
    R, Z = m.R, m.Z
    n = len(df)
    rng0 = np.random.default_rng([a.seed, 99])
    _, shot_id = np.unique(df.shot.values, return_inverse=True)
    sign = df.sign.values
    I = df[[f"c_{c}" for c in COILS]].values * sign[:, None]
    ipA = np.abs(df.ip.values) * 1e3
    tf = np.abs(df.tf.values)
    r_in, r_out = df.R_in.values, df.R_out.values
    ph = df.phase.values.astype(int)
    chR = [np.asarray(v, np.float64) for v in df.ch_R]
    chZ = [np.asarray(v, np.float64) for v in df.ch_Z]
    chT = [np.asarray(v, np.float64) for v in df.ch_Te]
    chN = [np.asarray(v, np.float64) for v in df.ch_ne]
    groups = [[np.nonzero(np.isclose(r_, 1.94, atol=1e-3))[0][np.argsort(z_[np.isclose(r_, 1.94, atol=1e-3)])],
               np.nonzero(~np.isclose(r_, 1.94, atol=1e-3))[0][np.argsort(r_[~np.isclose(r_, 1.94, atol=1e-3)])]]
              for r_, z_ in zip(chR, chZ)]
    # pass-1 Thomson tables on XG (EFIT-mapped x for DIII-D; zero beyond x = 1.1)
    tq, nq = np.stack(df.te_q.values).astype(np.float64), np.stack(df.ne_q.values).astype(np.float64)
    XQ = np.linspace(0, 1, 21)
    ext_ = lambda q: np.where(XG <= 1.0, np.interp(XG, XQ, q), np.interp(XG, [1.0, 1.1], [q[-1], 0.0], right=0.0))  # noqa: E731
    te_x = np.stack([ext_(q) for q in tq]); ne_x = np.stack([ext_(q) for q in nq])
    pe_x = te_x * ne_x * 1e19 * 1e3 * E
    ext0 = gsolve.coil_flux(m, I).reshape(n, -1)
    Rc0 = 0.5 * (r_in + r_out)
    a0 = np.clip(0.5 * (r_out - r_in), 0.4, 0.8)
    z_bal = np.concatenate([gsolve.balance_height(m, ext0[j:j + a.chunk], Rc0[j:j + a.chunk], a0[j:j + a.chunk])
                            for j in range(0, n, a.chunk)])
    z_bal = np.nan_to_num(z_bal, nan=0.0)
    ipf = np.clip(df.ip_frac.values, 1 / 3, 1.0)
    fpk = np.select([ph == 0, ph == 2], [ipf ** kn["pk_up"], ipf ** (-kn["pk_down"])], 1.0)
    cp = cond_prior.load(a.cond) if a.cond else None
    if cp is not None:
        Zf = df[[f"z_{k}" for k in zfeat.FEATS_ALL if f"z_{k}" in df.columns]].values
    print(f"{n} DIII-D frames, {len(shots)} shots; knobs {json.dumps(kn)}; cond {a.cond}", flush=True)

    RR = np.meshgrid(R, Z)[0].ravel()
    Vbv = 1e-3 * (RR ** 2 - m.R0 ** 2) / 2
    xq = np.sqrt(np.linspace(0, 1, NX))
    cfg1 = FwdConfig(n_iter=a.n_iter, tol=a.tol, bv_max=40.0)
    cfg2 = FwdConfig(n_iter=a.pass2_iter, tol=a.tol, bv_max=40.0)
    V = int(kn["n_var"]) if kn.get("n_var", 0) > 0 else a.variants
    keys = ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam", "bv", "Rin", "Rout", "K", "am", "an", "zc", "w", "D", "kp",
            "li_t")
    res = {k: np.full((V, n), np.nan) for k in keys}
    psi_all = np.zeros((V, n, R.size * Z.size), np.float32)
    p_all = np.zeros((V, n, NX)); s_all = np.full((V, n, NX), np.nan)
    tab_all = np.full((V, n, 2, len(RG)), np.nan)
    for v in range(V):
        dr = gp.Draws(a.seed, v, shot_id, kn["shot_corr"])
        if cp is not None:
            K, am, an, li_t = cond_prior.draw(cp, Zf, kn, dr, machine="d3d")
        else:
            K = dr.uniform(kn["k_lo"], max(kn["k_hi"], kn["k_lo"]))
            am = dr.uniform(kn["am_lo"], max(kn["am_hi"], kn["am_lo"]))
            an = dr.uniform(kn["an_lo"], max(kn["an_hi"], kn["an_lo"])) * fpk
            li_t = np.full(n, np.nan)
        w = dr.uniform(0.0, kn["w_te"]) if kn["w_te"] > 0 else np.zeros(n)
        r_tgt = r_out + kn["edge_bias"] + kn["edge_sd"] * dr.normal()
        zc = dr.uniform(kn["zc_lo"], max(kn["zc_hi"], kn["zc_lo"])) * z_bal + kn["zc_sd"] * dr.normal()
        kp = dr.uniform(0.0, kn["kp_hi"]) if kn["kp_hi"] > 0 else np.zeros(n)
        fk = (1 + kp[:, None] * (1 - xq[None] ** 2)) / (1 + kp[:, None] / 2)
        p_tab = np.stack([gp.pressure_table(np.interp(xq, XG, pe_x[i]), K[i]) * fk[i] for i in range(n)])
        s_tab = np.full((n, NX), np.nan)
        for i in range(n):
            if w[i] > 0:
                st = gp.te_shape(np.interp(xq, XG, te_x[i]))
                if st is not None:
                    s_tab[i] = st
        xg = np.linspace(0, 1, NX)

        def blend(i_sl):
            out = np.full((len(i_sl), NX), np.nan)
            for q, i in enumerate(i_sl):
                if w[i] > 0 and np.isfinite(s_tab[i]).all():
                    out[q] = (1 - w[i]) * (1 - xg ** am[i]).clip(0) ** an[i] + w[i] * s_tab[i]
            return out
        for i0 in range(0, n, a.chunk):
            sl = slice(i0, min(i0 + a.chunk, n)); B = sl.stop - sl.start
            idx = np.arange(sl.start, sl.stop)
            ext = torch.as_tensor(ext0[sl]).reshape(B, len(Z), len(R))
            J0 = start_current(m, Rc0[sl], zc[sl], a0[sl], np.full(B, 1.8), np.full(B, 0.3))
            prof = dict(mode="tab", p_tab=p_tab[sl], alpha_m=am[sl], alpha_n=an[sl], s_tab=blend(idx))
            rh = r_tgt[sl]
            s1 = forward_solve(m, ext, ipA[sl], prof, cfg1, z_hold=zc[sl], J0=J0, r_hold=rh)
            for j in np.nonzero(s1.status.numpy() == 0)[0]:
                i = sl.start + j
                pch = psi_at(s1.psi[j].numpy(), R, Z, chR[i], chZ[i])
                tb = flux_tables2(pch, float(s1.psi_ax[j]), float(s1.psi_b[j]), chT[i], chN[i], k3=bool(round(kn["remap_k3"])))
                if tb is not None:
                    tbf = tb if kn["remap_mono"] >= 0.5 else flux_tables2(pch, float(s1.psi_ax[j]), float(s1.psi_b[j]), chT[i],
                                                                         chN[i], k3=bool(round(kn["remap_k3"])), mono=False)
                    if tbf is not None:
                        tab_all[v, i] = tbf
                    p_tab[i] = gp.pressure_table(np.interp(xq, RG, tb[0] * tb[1] * 1e19 * 1e3 * E), K[i]) * fk[i]
                    if w[i] > 0:
                        st = gp.te_shape(np.interp(xq, RG, tb[0]))
                        if st is not None:
                            s_tab[i] = st
            prof["p_tab"] = p_tab[sl]; prof["s_tab"] = blend(idx)
            s2 = forward_solve(m, ext, ipA[sl], prof, cfg2, z_hold=zc[sl], J0=J0, r_hold=rh, psi_start=s1.psi, bv0=s1.bv)
            psi_all[v, sl] = s2.psi.reshape(B, -1).numpy()
            for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam"):
                res[k][v, sl] = s2.info[k].double().numpy()
            res["bv"][v, sl] = s2.bv.numpy()
            s_all[v, sl] = prof["s_tab"]
            for j in np.nonzero(s2.status.numpy() == 0)[0]:
                i = sl.start + j
                res["Rin"][v, i], res["Rout"][v, i] = gp.midplane_edges(s2.psi[j].numpy(), float(s2.psi_ax[j]),
                                                                        float(s2.psi_b[j]), R, Z)
                if np.isfinite(tab_all[v, i]).all():
                    pch = psi_at(s2.psi[j].numpy(), R, Z, chR[i], chZ[i])
                    res["D"][v, i] = th_mismatch2(pch, float(s2.psi_ax[j]), float(s2.psi_b[j]), chT[i], chN[i], groups[i],
                                                  tab_all[v, i])
        p_all[v] = p_tab
        res["K"][v], res["am"][v], res["an"][v], res["zc"][v], res["w"][v], res["kp"][v] = K, am, an, zc, w, kp
        res["li_t"][v] = li_t
        print(f"variant {v}: converged {(res['status'][v] == 0).mean():.0%} ({time.time() - t0:.0f}s)", flush=True)

    ok = (res["status"] == 0) & np.isfinite(res["Rout"])
    print(f"edge errors (converged): out {np.nanpercentile(np.where(ok, res['Rout'] - r_out[None], np.nan), [10, 50, 90]).round(3)} in {np.nanpercentile(np.where(ok, res['Rin'] - r_in[None], np.nan), [10, 50, 90]).round(3)}", flush=True)
    dout, din = res["Rout"] - r_out[None], res["Rin"] - r_in[None]
    acc = ok & (np.abs(dout) <= kn["sel_out"]) & (np.abs(din) <= kn["sel_in"])
    Dm = np.where(np.isfinite(res["D"]), res["D"], np.nanmax(np.append(res["D"][np.isfinite(res["D"])], 1.0)))
    if kn["th_tol"] > 0:
        acc &= Dm <= kn["th_tol"]
    err_all = np.abs(dout) + kn["sel_win"] * np.abs(din) + kn["sel_th"] * Dm
    kept, rank = [], []
    for i in range(n):
        c = np.nonzero(acc[:, i])[0]
        c = c[np.argsort(err_all[c, i])][:a.keep]
        kept += [(v, i) for v in c]; rank += list(range(len(c)))
    vv, ii = np.array([k[0] for k in kept], int), np.array([k[1] for k in kept], int)
    N = len(kept)
    prof_k = dict(mode="tab", p_tab=torch.as_tensor(p_all[vv, ii]), alpha_m=torch.as_tensor(res["am"][vv, ii]),
                  alpha_n=torch.as_tensor(res["an"][vv, ii]), s_tab=torch.as_tensor(s_all[vv, ii]))
    sol = gsolve.Solution(m, torch.as_tensor(psi_all[vv, ii].astype(np.float64)).reshape(N, len(Z), len(R)),
                          torch.as_tensor(ipA[ii]), prof_k,
                          {k: torch.as_tensor(res[k][vv, ii]) for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam")})
    lab = gsolve.analyse_all(sol, tf[ii])
    out = dict(row=ii, variant=vv, rank=np.array(rank), err=err_all[vv, ii], dout=dout[vv, ii], din=din[vv, ii], shot=df.shot.values[ii], frame=df.frame.values[ii],
               psi=psi_all[vv, ii].reshape(-1, len(Z), len(R)), z_bal=z_bal[ii],
               **{f"gs_{k}": lab[k] for k in lab}, **{f"gen_{k}": res[k][vv, ii] for k in ("K", "am", "an", "zc", "w", "D", "kp", "bv", "li_t")},
               n_frames=np.int64(n), n_frames_kept=np.int64(len(set(ii.tolist()))), knobs=np.array(json.dumps(kn)),
               conv=np.array([(res["status"][v] == 0).mean() for v in range(V)]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out + ".tmp.npz", **out)
    Path(a.out + ".tmp.npz").replace(a.out)
    print(f"kept {N} equilibria, {len(set(ii.tolist()))}/{n} frames; li {np.nanpercentile(lab['li'], [5, 50, 95]).round(2)} "
          f"betaN {np.nanpercentile(lab['betaN'], [5, 50, 95]).round(2)}; {time.time() - t0:.0f}s; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
