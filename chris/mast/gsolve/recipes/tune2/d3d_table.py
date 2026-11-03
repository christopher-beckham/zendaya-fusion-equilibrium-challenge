#!/usr/bin/env python3
"""DIII-D frame table (training labels allowed): input features z, hidden-physics targets theta, and what the
DIII-D recipe check needs (coils, Ip, TF, Thomson channels with positions, EFIT edges, labels, flux map).

    python d3d_table.py --split train --n 700 --per-shot 16 --workers 16      -> d3d/train.parquet
    python d3d_table.py --split val   --n 250 --per-shot 8  --workers 16 --psi -> d3d/val.parquet + d3d/val_psi.npy

Shots come from our own split (/fsx/chris.beckham/fusion_cache/shots.parquet), all inside diii_d_train.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, "/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_fixes import fix_d3d_ip_times  # noqa: E402
import zfeat  # noqa: E402

DATA = Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/diii_d_train")
OUT = Path(__file__).resolve().parent / "d3d"
COILS = ["F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B", "F6A", "F6B", "F7A", "F7B", "F8A", "F8B",
         "F9A", "F9B", "ECOILA"]
MU0 = 4e-7 * np.pi
RCENTR = 1.6955


def _pip(rq, zq, C):
    """even-odd point in polygon; C (N, 2) [R, Z]"""
    r, z = C[:, 0], C[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    ins = np.zeros(len(rq), bool)
    for a, b, c, d in zip(r, z, r2, z2):
        cond = ((b > zq) != (d > zq)) & (rq < (c - a) * (zq - b) / (d - b + 1e-300) + a)
        ins ^= cond
    return ins


def nearest_profile(times, Te, ne, t, win=20.0):
    times = np.asarray(times, np.float64)
    if not len(times):
        return None
    j = int(np.argmin(np.abs(times - t)))
    if abs(times[j] - t) > win:
        return None
    te, nn = np.asarray(Te[j], np.float64), np.asarray(ne[j], np.float64)
    ok = np.isfinite(te) & np.isfinite(nn) & (te > 1) & (te < 2e4) & (nn > 1e17) & (nn < 3e20)
    return np.where(ok, te / 1e3, np.nan), np.where(ok, nn / 1e19, np.nan)


def one(args):
    stem, per_shot, want_psi = args
    try:
        return _one(stem, per_shot, want_psi)
    except Exception as e:  # noqa: BLE001
        print(f"{stem}: {type(e).__name__} {e}", flush=True)
        return None


def _one(stem, per_shot, want_psi):
    cols = ["efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_beta_n", "efit_li", "efit_q95", "efit_r_axis",
            "efit_z_axis", "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "magnetics_time", "magnetics_plasma_current",
            "magnetics_plasma_current_times", "magnetics_bcoil", "thomson_core_times", "thomson_core_Te",
            "thomson_core_ne", "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "thomson_chord_name",
            "thomson_chord_R", "thomson_chord_Z"] + [f"magnetics_{c}" for c in COILS]
    r = pq.read_table(DATA / f"{stem}.parquet", columns=cols).to_pylist()[0]
    te_t = np.asarray(r["efit_times"], np.float64)
    ipt = fix_d3d_ip_times(r)
    ip = np.asarray(r["magnetics_plasma_current"], np.float64)
    ok = np.isfinite(ipt) & np.isfinite(ip)
    ipt, ip = ipt[ok], ip[ok]
    ipmax = np.nanpercentile(np.abs(ip), 99.5)
    sign = 1.0 if np.median(ip[np.abs(ip) >= 0.8 * ipmax]) >= 0 else -1.0
    hi = np.nonzero(np.abs(ip) >= 0.9 * ipmax)[0]
    t_hi0, t_hi1 = ipt[hi[0]], ipt[hi[-1]]
    ipf = lambda t: np.interp(t, ipt, ip)  # noqa: E731
    tm = np.asarray(r["magnetics_time"], np.float64)
    names = list(r["thomson_chord_name"])
    cR, cZ = np.asarray(r["thomson_chord_R"], np.float64), np.asarray(r["thomson_chord_Z"], np.float64)
    ic = [k for k, nm in enumerate(names) if nm.startswith("TS_core")]
    it = [k for k, nm in enumerate(names) if nm.startswith("TS_tangential")]
    nc_ = len(r["thomson_core_Te"][0]) if r["thomson_core_Te"] else 0
    ne_ = len(r["thomson_edge_Te"][0]) if r["thomson_edge_Te"] else 0
    if nc_ != len(ic) or (ne_ and ne_ != len(it)):
        print(f"{stem}: chord mismatch core {nc_}/{len(ic)} tang {ne_}/{len(it)}", flush=True)
        return None
    Rg, Zg = np.asarray(r["efit_grid_R"]), np.asarray(r["efit_grid_Z"])
    RR, ZZ = np.meshgrid(Rg, Zg)
    rows, psis = [], []
    cand = [i for i in range(len(te_t)) if abs(ipf(te_t[i])) >= 300.0 and r["efit_lcfs_n"][i] and r["efit_lcfs_n"][i] > 20]
    if not cand:
        return None
    pick = np.unique(np.round(np.linspace(0, len(cand) - 1, min(per_shot, len(cand)))).astype(int))
    for k in pick:
        i = cand[k]
        t = te_t[i]
        pc = nearest_profile(r["thomson_core_times"], r["thomson_core_Te"], r["thomson_core_ne"], t) if nc_ else None
        pt = nearest_profile(r["thomson_edge_times"], r["thomson_edge_Te"], r["thomson_edge_ne"], t) if ne_ else None
        if pc is None:
            continue
        chR = np.concatenate([cR[ic], cR[it] if pt is not None else []])
        chZ = np.concatenate([cZ[ic], cZ[it] if pt is not None else []])
        Te = np.concatenate([pc[0], pt[0] if pt is not None else []])
        ne = np.concatenate([pc[1], pt[1] if pt is not None else []])
        psi = np.asarray(r["efit_psirz"][i], np.float64)
        sp = RectBivariateSpline(Zg, Rg, psi)
        rax, zax = r["efit_r_axis"][i], r["efit_z_axis"][i]
        n = int(r["efit_lcfs_n"][i])
        C = np.stack([np.asarray(r["efit_lcfs_r"][i][:n], np.float64), np.asarray(r["efit_lcfs_z"][i][:n], np.float64)], 1)
        C = C[np.isfinite(C).all(1)]
        if len(C) < 20 or not np.isfinite([rax, zax]).all():
            continue
        pax = float(sp(zax, rax)[0, 0])
        pb = float(np.mean(sp.ev(C[:, 1], C[:, 0])))
        if abs(pb - pax) < 1e-6:
            continue
        psn = lambda rq, zq: (sp.ev(zq, rq) - pax) / (pb - pax)  # noqa: E731
        # outboard / inboard midplane rays at the axis height
        ro = np.linspace(rax, Rg[-1], 400); xo = np.maximum.accumulate(psn(ro, np.full_like(ro, zax)))
        ri = np.linspace(rax, Rg[0], 400); xi = np.maximum.accumulate(psn(ri, np.full_like(ri, zax)))
        if not (xo >= 1).any() or not (xi >= 1).any():
            continue
        R_out = float(np.interp(1.0, xo, ro)); R_in = float(np.interp(1.0, xi, ri))
        lcfsR0, lcfsR1 = C[:, 0].min(), C[:, 0].max()
        a = 0.5 * (lcfsR1 - lcfsR0); Rgeo = 0.5 * (lcfsR1 + lcfsR0)
        pn_ch = psn(chR, chZ)
        xr = (ro - rax) / (R_out - rax)
        x_ch = np.where(pn_ch <= 1.0, np.interp(np.clip(pn_ch, 0, 1), xo, xr), np.nan)
        prof = zfeat.profile_on_xq(x_ch, Te, ne)
        if prof is None:
            continue
        tq, nq, xmin = prof
        tB = MU0 * 144 * abs(np.nanmedian(np.asarray(r["magnetics_bcoil"], np.float64)[np.abs(tm - t) <= 2.0])) * 1e3 / (2 * np.pi)
        B0 = tB / Rgeo
        ipk = abs(ipf(t))
        rr = (abs(ipf(t + 20)) - abs(ipf(t - 20))) / 0.040 / max(ipk, 1.0)
        phase = 0 if t < t_hi0 else (2 if t > t_hi1 else 1)
        z = zfeat.features(tq, nq, ipk, ipk / ipmax, phase, rr, B0, a, Rgeo)
        # p_e(psi_N) volume average over the EFIT plasma (cells inside the LCFS, weight R)
        inside_ch = np.isfinite(pn_ch) & (pn_ch <= 1.0) & np.isfinite(Te) & np.isfinite(ne)
        if inside_ch.sum() < 8:
            continue
        rho = np.sqrt(np.clip(pn_ch[inside_ch], 0, 1))
        pe_ch = Te[inside_ch] * ne[inside_ch] * 1e3 * 1e19 * zfeat.E
        bins = np.clip(np.round(rho / 0.05).astype(int), 0, 20)
        tab = np.array([np.median(pe_ch[bins == b]) if (bins == b).any() else np.nan for b in range(21)])
        f = np.isfinite(tab)
        tab = np.interp(np.linspace(0, 1, 21), np.linspace(0, 1, 21)[f], tab[f])
        cell_in = _pip(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
        pn_grid = (psi - pax) / (pb - pax)
        rho_g = np.sqrt(np.clip(pn_grid[cell_in], 0, 1))
        pe_av = float(np.sum(np.interp(rho_g, np.linspace(0, 1, 21), tab) * RR[cell_in]) / np.sum(RR[cell_in]))
        bN = r["efit_beta_n"][i]
        B0c = tB / RCENTR
        betat = bN * (ipk / 1e3) / (a * B0c) / 100
        p_av = betat * B0c ** 2 / (2 * MU0)
        coils = np.array([np.nanmedian(np.asarray(r[f"magnetics_{c}"], np.float64)[np.abs(tm - t) <= 1.0]) for c in COILS])
        row = dict(shot=stem, frame=i, time=t, sign=sign, ip=ipf(t), tf=float(np.nanmedian(np.asarray(r["magnetics_bcoil"], np.float64)[np.abs(tm - t) <= 2.0])),
                   phase=phase, ip_frac=ipk / ipmax, rr=rr, a=a, Rgeo=Rgeo, R_in=R_in, R_out=R_out, B0=B0, xmin=xmin,
                   li=r["efit_li"][i], betaN=bN, q95=r["efit_q95"][i], R_ax=rax, Z_ax=zax,
                   kappa=(C[:, 1].max() - C[:, 1].min()) / (2 * a), pe_av=pe_av, p_av=p_av, K=p_av / max(pe_av, 1e-9),
                   te_q=tq.astype(np.float32), ne_q=nq.astype(np.float32),
                   ch_R=chR.astype(np.float32), ch_Z=chZ.astype(np.float32), ch_Te=Te.astype(np.float32), ch_ne=ne.astype(np.float32),
                   **{f"z_{k}": v for k, v in z.items()}, **{f"c_{c}": v for c, v in zip(COILS, coils)})
        rows.append(row)
        if want_psi:
            psis.append(psi.astype(np.float32))
    return rows, psis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--per-shot", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--psi", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sp = pd.read_parquet("/fsx/chris.beckham/fusion_cache/shots.parquet")
    stems = sorted(sp.loc[sp.split == a.split, "shot"])
    rng = np.random.default_rng(a.seed)
    stems = list(rng.choice(stems, min(a.n, len(stems)), replace=False))
    OUT.mkdir(exist_ok=True)
    rows, psis = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for j, res in enumerate(ex.map(one, [(s, a.per_shot, a.psi) for s in stems], chunksize=2)):
            if res:
                rows += res[0]; psis += res[1]
            if j % 50 == 0:
                print(f"{j}/{len(stems)} shots, {len(rows)} frames", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"{a.split}.parquet")
    if a.psi:
        np.save(OUT / f"{a.split}_psi.npy", np.stack(psis))
    print(f"wrote {len(df)} frames from {df.shot.nunique()} shots", flush=True)


if __name__ == "__main__":
    main()
