#!/usr/bin/env python3
"""Synthetic MAST equilibria seeded by real challenge-format input files (coil currents, Ip, TF, Thomson).

Only input columns are read. Every range below is set from physics, published documentation, DIII-D or the
frame's own inputs; the sources are given next to each setting in the code. For every frame (EFIT time) with
Ip >= 100 kA of each shot file, `--variants` equilibria are solved:
  1. Coils: the frame's real coil currents (solenoid column x --solenoid-scale, default 0.5, see
     gsolve/machines/mast.json) plus a uniform vertical "vessel" field b_v, tuned during the solve so the
     outer midplane plasma edge lands on the frame's Thomson outer edge (+ N(0, 1.5 cm)), |b_v| <= 40 mT.
     Frames without usable Thomson: b_v ~ N(0, 4 mT). (Thomson edges: radii where the midplane Te falls to
     5% of its peak.)
  2. Pressure p(psi_N) = K n_e T_e, K ~ U(1.8, 3.5) (total / electron pressure, bounds from the pressure
     composition, see K_RANGE). Pass 1 places the Thomson profile on a normalised radius; pass 2 places the
     real channels on the pass-1 flux surfaces and solves again.
  3. Current FF' ~ (1 - psi_N^am)^an with am ~ U(1, 3), an ~ U(0.5, 3) around FreeGS's default shape
     (am, an) = (1, 2). The phase of the shot comes from the input Ip only. In the ramp-down the inner current
     decays slowly while Ip falls (li Ip about constant), so an is multiplied by Ip_max / Ip; in the ramp-up the
     current has not yet diffused in, so an is multiplied by Ip / Ip_max (both factors limited to 3). The
     total current equals the measured Ip.
  4. Height: the current centroid is held at Z_c by a fitted radial field. Z_c is drawn uniformly between the
     midplane and the frame's coil force-balance height (gsolve.balance_height: where the measured coil
     currents exert no net vertical force on a nominal plasma), plus N(0, 1.5 cm).
  5. Selection: keep the `--keep` variants whose solved midplane edges lie closest to the Thomson edges (within
     5 cm outer, 8 cm inner); frames without Thomson keep one converged variant.
  6. Equilibrium numbers from each solution (gsolve.analyse): q95, betaN, li, elongation, triangularity, axis.
  7. Thomson remapped onto the solved surfaces: each channel gets the pass-2 flux-surface profile at its own
     rho = sqrt(psi_N), plus noise at the real frame's own scatter. Missing channels stay missing.
The wall-edge fix is on; --no-wall-fix reads a limited plasma's edge on the grid cells next to the wall box.

    python examples/generate_synthetic.py /data/mast_public_test/mast_shot_*.parquet --out synth.npz
Output (.npz, one row per kept equilibrium): shot (file id), frame, time, variant, psi (N, 65, 65) physical
flux on the challenge grid (Wb/rad; for MAST this is also the challenge's stored convention), coils / ip / tf
as in the file, Te (keV) / ne (1e19 m^-3) remapped on the channels at R_ts, the equilibrium numbers, and the
drawn settings (gen_*).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import gsolve
from gsolve.forward import FwdConfig, forward_solve, start_current

E = 1.602176634e-19
NX = 61                              # psi_N grid of the pressure table
XG = np.linspace(0.0, 1.5, 76)       # normalised-radius grid of the pass-1 Thomson table
RG = np.linspace(0.0, 1.5, 31)       # rho grid of the flux-surface tables
RLINE = np.linspace(0.2, 1.55, 271)  # midplane chord for the edge read-out
# K = p_total / p_e = 1 + (n_i / n_e)(T_i / T_e) + p_fast / p_e. Lower bound: thermal plasma only, T_i = T_e and
# ion dilution n_i / n_e = 0.8 (Z_eff about 2 from carbon) gives 1.8. Upper bound: T_i up to 1.5 T_e with beam
# heating and fast beam ions up to a third of the total pressure gives about 3.5. The DIII-D value fitted on
# DIII-D training data, 2.42, lies inside.
K_RANGE = (1.8, 3.5)


# ---------------------------------------------------------------- reading one challenge file (input columns only)

def _interp(t, v, te):
    t, v = np.asarray(t, np.float64), np.asarray(v, np.float64)
    ok = np.isfinite(t) & np.isfinite(v)
    return np.interp(te, t[ok], v[ok]) if ok.sum() >= 2 else np.full(len(te), np.nan)


def _thomson(times, Te, ne, te_frames, window=20.0):
    """Nearest Thomson profile within +-20 ms of each frame; Te keV, ne 1e19 m^-3, invalid -> NaN."""
    Te, ne, times = np.array([np.asarray(r, np.float64) for r in Te]), np.array([np.asarray(r, np.float64) for r in ne]), np.asarray(times, np.float64)
    j = np.clip(np.searchsorted(times, te_frames), 1, len(times) - 1)
    j = np.where(np.abs(te_frames - times[j - 1]) <= np.abs(times[j] - te_frames), j - 1, j)
    te, nn = Te[j], ne[j]
    ok = (np.abs(times[j] - te_frames) <= window)[:, None] & np.isfinite(te) & np.isfinite(nn) \
        & (te > 1) & (te < 2e4) & (nn > 1e17) & (nn < 3e20)
    return np.where(ok, te / 1e3, np.nan), np.where(ok, nn / 1e19, np.nan)


def read_shot(path, m):
    cols = [m.input_format.format(c) for c in m.columns]
    ts = ["thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R", "thomson_edge_times",
          "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial"]
    r = pq.read_table(path, columns=["efit_times", "magnetics_time", "magnetics_plasma_current", m.tf_input] + cols + ts).to_pylist()[0]
    te, tm = np.asarray(r["efit_times"], np.float64), r["magnetics_time"]
    ipraw = np.asarray(r["magnetics_plasma_current"], np.float64)
    ft = np.abs(ipraw) >= 0.8 * np.nanmax(np.abs(ipraw))
    sign = 1.0 if np.nanmedian(ipraw[ft]) >= 0 else -1.0
    out = dict(shot=Path(path).stem.replace("mast_shot_", ""), time=te, sign=sign, ip=_interp(tm, ipraw, te),
               tf=_interp(tm, r[m.tf_input], te), coils=np.stack([_interp(tm, r[c], te) for c in cols], 1))
    Te, ne, R = [], [], []
    if r["thomson_core_Te"]:
        Rc = np.asarray(r["thomson_core_R"], np.float64)
        t_, n_ = _thomson(r["thomson_core_times"], r["thomson_core_Te"], r["thomson_core_ne"], te)
        t_, n_ = (t_[:, 1:], n_[:, 1:]) if t_.shape[1] == len(Rc) + 1 else (t_, n_)   # MAST ghost channel 0
        Te.append(t_); ne.append(n_); R.append(Rc)
    if r["thomson_edge_Te"]:
        t_, n_ = _thomson(r["thomson_edge_times"], r["thomson_edge_Te"], r["thomson_edge_ne"], te)
        if t_.shape[1] == len(r["thomson_edge_spatial"]):
            Te.append(t_); ne.append(n_); R.append(np.asarray(r["thomson_edge_spatial"], np.float64))
    out["Te"], out["ne"] = (np.concatenate(v, 1) if v else np.zeros((len(te), 0)) for v in (Te, ne))
    out["R_ts"] = np.concatenate(R) if R else np.zeros(0)
    return out


# ---------------------------------------------------------------- recipe pieces

def ts_table(te, ne, r):
    """Midplane profile -> (r_peak, r_in, r_out, Te(XG), ne(XG)) against a normalised radius, or None."""
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(r)
    if ok.sum() < 20:
        return None
    o = np.argsort(r[ok]); rr, tt, nn = r[ok][o], te[ok][o], ne[ok][o]
    ts = np.convolve(tt, np.ones(5) / 5, mode="same")
    if ts.max() < 0.05 or (ts >= 0.05 * ts.max()).sum() < 8:
        return None
    hs = np.sort(rr[ts >= 0.05 * ts.max()])
    r_in, r_out, r_pk = hs[2], hs[-3], rr[np.argmax(ts)]
    if not r_in < r_pk < r_out:
        return None
    x = np.where(rr >= r_pk, (rr - r_pk) / (r_out - r_pk), (r_pk - rr) / (r_pk - r_in))
    o = np.argsort(x)
    t3, n3 = np.convolve(tt[o], np.ones(3) / 3, mode="same"), np.convolve(nn[o], np.ones(3) / 3, mode="same")
    t3[0], n3[0] = tt[o][:2].mean(), nn[o][:2].mean()
    return r_pk, r_in, r_out, np.interp(XG, x[o], t3), np.interp(XG, x[o], n3)


def flux_tables(row, psax, psb, Rg, Rch, te, ne):
    """Thomson channels on a solved map's flux surfaces: median per rho bin, gaps filled, light smoothing,
    Te made non-increasing in rho. Returns (Te(RG), ne(RG)) or None."""
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(Rch)
    if ok.sum() < 20 or not psax > psb:
        return None
    rho = np.sqrt(np.clip((psax - np.interp(Rch[ok], Rg, row)) / (psax - psb), 0, None))
    b = np.digitize(rho, np.concatenate([RG - 0.025, [RG[-1] + 0.025]])) - 1
    T = np.array([np.median(te[ok][b == k]) if (b == k).any() else np.nan for k in range(len(RG))])
    N = np.array([np.median(ne[ok][b == k]) if (b == k).any() else np.nan for k in range(len(RG))])
    f = np.isfinite(T)
    if f.sum() < 6 or not (RG[f] < 0.6).any():
        return None
    k3 = np.array([0.25, 0.5, 0.25])
    T, N = (np.convolve(np.pad(np.interp(RG, RG[f], v[f]), 1, mode="edge"), k3, mode="valid") for v in (T, N))
    return np.maximum.accumulate(T[::-1])[::-1], N


def pressure_table(pe_of_x, K):
    pe = np.maximum.accumulate(pe_of_x[::-1])[::-1]               # never rises outward
    return K * np.clip(pe - pe[-1], 0, None)


def midplane_edges(psi, psax, psb, R, Z):
    """Inner and outer plasma edge along Z = 0 (NaN if the axis is not on the chord)."""
    iz = np.searchsorted(Z, 0.0) - 1
    wz = (0.0 - Z[iz]) / (Z[iz + 1] - Z[iz])
    xn = (psax - np.interp(RLINE, R, (1 - wz) * psi[iz] + wz * psi[iz + 1])) / (psax - psb)
    ia = int(np.argmin(xn))
    if not xn[ia] <= 1.0:
        return np.nan, np.nan
    lo, hi = ia, ia
    while lo > 0 and xn[lo - 1] <= 1.0:
        lo -= 1
    while hi < len(RLINE) - 1 and xn[hi + 1] <= 1.0:
        hi += 1
    return RLINE[lo], RLINE[hi]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="challenge-format MAST shot files (.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-iter", type=int, default=160)
    ap.add_argument("--pass2-iter", type=int, default=120)
    ap.add_argument("--tol", type=float, default=3e-3)
    ap.add_argument("--k-range", type=float, nargs=2, default=list(K_RANGE))
    ap.add_argument("--solenoid-scale", type=float, default=0.5,
                    help="factor on the solenoid column (default 0.5, documented in gsolve/machines/mast.json)")
    ap.add_argument("--no-wall-fix", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    m = gsolve.load_machine("mast", solenoid_scale=a.solenoid_scale)
    R, Z = m.R, m.Z
    rng0 = np.random.default_rng(a.seed)

    # frames of all shots
    F = {k: [] for k in ("shot", "frame", "time", "coils", "ip", "tf", "sign", "phase", "ip_frac", "Te", "ne", "R_ts")}
    for f in a.files:
        s = read_shot(f, m)
        aip = np.abs(s["ip"])
        hi = np.nonzero(aip >= 0.9 * np.nanmax(aip))[0]
        phase = np.ones(len(aip), int); phase[:hi[0]] = 0; phase[hi[-1] + 1:] = 2      # ramp-up, flat top, ramp-down
        ip_frac = aip / np.nanmax(aip)                                                   # Ip / Ip_max of the shot
        sel = np.nonzero((aip >= 100.0) & np.isfinite(s["coils"]).all(1) & np.isfinite(s["tf"]))[0][::a.every]
        for i in sel:
            for k, v in (("shot", s["shot"]), ("frame", i), ("time", s["time"][i]), ("coils", s["coils"][i]), ("ip", s["ip"][i]),
                         ("tf", s["tf"][i]), ("sign", s["sign"]), ("phase", phase[i]), ("ip_frac", ip_frac[i]), ("Te", s["Te"][i]),
                         ("ne", s["ne"][i]), ("R_ts", s["R_ts"])):
                F[k].append(v)
    n = len(F["ip"])
    C = max(len(r) for r in F["R_ts"])
    pad = lambda v: np.concatenate([v, np.full(C - len(v), np.nan)])  # noqa: E731
    F = {k: (np.stack([pad(x) for x in v]) if k in ("Te", "ne", "R_ts") else np.array(v)) for k, v in F.items()}
    ph, ipA = F["phase"], np.abs(F["ip"]) * 1e3
    I = F["coils"] * F["sign"][:, None]                               # physical sign (Ip > 0)
    tabs = [ts_table(F["Te"][i], F["ne"][i], F["R_ts"][i]) for i in range(n)]
    has = np.array([t is not None for t in tabs])
    r_in, r_out = (np.array([t[j] if t else np.nan for t in tabs]) for j in (1, 2))
    pe_x = np.stack([t[3] * t[4] * 1e19 * 1e3 * E if t else np.zeros(len(XG)) for t in tabs])
    ext0 = gsolve.coil_flux(m, I).reshape(n, -1)
    # nominal plasma: centre and size from the Thomson edges, else MAST's nominal R 0.88 m, a 0.57 m;
    # elongation 1.8 and triangularity 0.3 as typical MAST shapes (start shape and force-balance shape only)
    Rc0 = np.where(has, 0.5 * (r_in + r_out), 0.88)
    a0 = np.clip(np.where(has, 0.5 * (r_out - r_in), 0.57), 0.3, 0.68)
    z_bal = np.concatenate([gsolve.balance_height(m, ext0[j:j + a.chunk], Rc0[j:j + a.chunk], a0[j:j + a.chunk])
                            for j in range(0, n, a.chunk)])
    z_bal = np.nan_to_num(z_bal, nan=0.0)                            # no balance within +-0.4 m: midplane
    # peaking factor from the phase: ramp-down x Ip_max / Ip (li Ip about constant while Ip falls, as seen on
    # DIII-D shots), ramp-up x Ip / Ip_max (current not yet diffused in); limited to a factor 3
    fpk = np.clip(F["ip_frac"], 1 / 3, 1.0)
    fpk = np.select([ph == 0, ph == 2], [fpk, 1 / fpk], 1.0)
    print(f"{len(a.files)} files, {n} frames (phases {np.bincount(ph, minlength=3)}), Thomson on {has.mean():.0%}, "
          f"force-balance height 5/50/95% {np.percentile(z_bal, [5, 50, 95]).round(3)} m", flush=True)

    RR = np.meshgrid(R, Z)[0].ravel()
    Vbv = 1e-3 * (RR ** 2 - m.R0 ** 2) / 2
    xq = np.sqrt(np.linspace(0, 1, NX))
    cfg1 = FwdConfig(n_iter=a.n_iter, tol=a.tol, wall_fix=not a.no_wall_fix, bv_max=40.0)
    cfg2 = FwdConfig(n_iter=a.pass2_iter, tol=a.tol, wall_fix=not a.no_wall_fix, bv_max=40.0)
    V = a.variants
    keys = ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam", "bv", "Rin", "Rout", "K", "am", "an", "zc")
    res = {k: np.full((V, n), np.nan) for k in keys}
    psi_all = np.zeros((V, n, R.size * Z.size), np.float32)
    p_all = np.zeros((V, n, NX)); tab_all = np.full((V, n, 2, len(RG)), np.nan)
    t_solve = 0.0
    for v in range(V):
        rng = np.random.default_rng([a.seed, v, 0])
        K = rng.uniform(*a.k_range, n)
        am = rng.uniform(1, 3, n)
        an = rng.uniform(0.5, 3, n) * fpk
        bv = np.where(has, 0.0, rng.normal(0, 4.0, n))         # a few % of the ~130 mT coil vertical field
        r_tgt = np.where(has, r_out + rng.normal(0, 0.015, n), np.nan)
        zc = rng.uniform(0, 1, n) * z_bal + rng.normal(0, 0.015, n)
        betap = rng.uniform(0.05, 1.0, n)                      # broad poloidal-beta prior, frames without Thomson
        p_tab = np.zeros((n, NX))
        for i in range(n):
            if has[i]:
                p_tab[i] = pressure_table(np.interp(xq, XG, pe_x[i]), K[i])
            else:                                                     # no Thomson: p0 from beta_p
                Bp = gsolve.geometry.MU0 * ipA[i] / (2 * np.pi * 0.55 * np.sqrt((1 + 1.8 ** 2) / 2))
                p_tab[i] = 2.5 * betap[i] * Bp ** 2 / (2 * gsolve.geometry.MU0) * (1 - xq ** 2) ** 1.5
        for i0 in range(0, n, a.chunk):
            sl = slice(i0, min(i0 + a.chunk, n)); B = sl.stop - sl.start
            ext = torch.as_tensor(ext0[sl] + bv[sl, None] * Vbv[None]).reshape(B, len(Z), len(R))
            J0 = start_current(m, Rc0[sl], zc[sl], a0[sl], np.full(B, 1.8), np.full(B, 0.3))
            prof = dict(mode="tab", p_tab=p_tab[sl], alpha_m=am[sl], alpha_n=an[sl])
            rh = np.where(has[sl], np.nan_to_num(r_tgt[sl], nan=1.3), np.nan)
            t1 = time.time()
            s1 = forward_solve(m, ext, ipA[sl], prof, cfg1, z_hold=zc[sl], J0=J0, r_hold=rh)
            for j in np.nonzero((s1.status.numpy() == 0) & has[sl])[0]:     # pass 2: pressure on the pass-1 surfaces
                i = sl.start + j
                tb = flux_tables(s1.psi[j].numpy()[np.argmin(np.abs(Z))], float(s1.psi_ax[j]), float(s1.psi_b[j]), R,
                                 F["R_ts"][i], F["Te"][i], F["ne"][i])
                if tb is not None:
                    tab_all[v, i] = tb
                    p_tab[i] = pressure_table(np.interp(xq, RG, tb[0] * tb[1] * 1e19 * 1e3 * E), K[i])
            prof["p_tab"] = p_tab[sl]
            s2 = forward_solve(m, ext, ipA[sl], prof, cfg2, z_hold=zc[sl], J0=J0, r_hold=rh, psi_start=s1.psi, bv0=s1.bv)
            t_solve += time.time() - t1
            psi_all[v, sl] = s2.psi.reshape(B, -1).numpy()
            for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam"):
                res[k][v, sl] = s2.info[k].double().numpy()
            res["bv"][v, sl] = bv[sl] + s2.bv.numpy()
            for j in np.nonzero(s2.status.numpy() == 0)[0]:
                res["Rin"][v, sl.start + j], res["Rout"][v, sl.start + j] = midplane_edges(
                    s2.psi[j].numpy(), float(s2.psi_ax[j]), float(s2.psi_b[j]), R, Z)
        p_all[v] = p_tab
        res["K"][v], res["am"][v], res["an"][v], res["zc"][v] = K, am, an, zc
        cv = res["status"][v] == 0
        print(f"variant {v}: converged {cv.mean():.0%} (ramp-up / flat top / ramp-down "
              f"{' / '.join(f'{cv[ph == q].mean():.0%}' if (ph == q).any() else '-' for q in range(3))}) "
              f"({time.time() - t0:.0f}s)", flush=True)

    # selection by the Thomson edges (an input proxy for the plasma size)
    ok = (res["status"] == 0) & np.isfinite(res["Rout"])
    dout, din = res["Rout"] - r_out[None], res["Rin"] - r_in[None]
    acc = ok & (np.abs(dout) <= 0.05) & (np.abs(din) <= 0.08)
    kept = []
    for i in range(n):
        if has[i]:
            c = np.nonzero(acc[:, i])[0]
            kept += [(v, i) for v in c[np.argsort((np.abs(dout) + 0.5 * np.abs(din))[c, i])][:a.keep]]
        elif ok[:, i].any():
            kept.append((int(rng0.choice(np.nonzero(ok[:, i])[0])), i))
    vv, ii = np.array([k[0] for k in kept], int), np.array([k[1] for k in kept], int)
    N = len(kept)
    # equilibrium numbers of each kept equilibrium
    sol = gsolve.Solution(m, torch.as_tensor(psi_all[vv, ii].astype(np.float64)).reshape(N, len(Z), len(R)),
                          torch.as_tensor(ipA[ii]),
                          dict(mode="tab", p_tab=torch.as_tensor(p_all[vv, ii]), alpha_m=torch.as_tensor(res["am"][vv, ii]),
                               alpha_n=torch.as_tensor(res["an"][vv, ii])),
                          {k: torch.as_tensor(res[k][vv, ii]) for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam")})
    lab = gsolve.analyse_all(sol, np.abs(F["tf"][ii]))
    good = np.isfinite(lab["q95"]) & np.isfinite(lab["betaN"]) & (lab["q95"] > 1) & (lab["betaN"] > 0.02)
    # Thomson remapped onto the solved surfaces, with the real frame's own relative scatter
    Te_new, ne_new = np.full((N, C), np.nan), np.full((N, C), np.nan)
    iz = np.argmin(np.abs(Z))
    for j in np.nonzero(good)[0]:
        v, i = vv[j], ii[j]
        if not np.isfinite(tab_all[v, i]).all():
            continue
        have = np.isfinite(F["Te"][i]) & np.isfinite(F["ne"][i]) & np.isfinite(F["R_ts"][i])
        xn = np.clip((res["psi_ax"][v, i] - np.interp(F["R_ts"][i], R, psi_all[v, i].reshape(len(Z), len(R))[iz]))
                     / (res["psi_ax"][v, i] - res["psi_b"][v, i]), 0, None)
        o = np.argsort(F["R_ts"][i][have])
        rel = []
        for q, floor in ((F["Te"][i], 0.02), (F["ne"][i], 0.05)):
            qq = q[have][o]; sm = np.convolve(qq, np.ones(5) / 5, mode="same")
            rel.append(np.clip(np.nanstd((qq - sm)[2:-2] / np.maximum(sm[2:-2], floor)), 0, 0.3) if have.sum() > 10 else 0.05)
        Te_new[j] = np.where(have, np.clip(np.interp(np.sqrt(xn), RG, tab_all[v, i, 0]) * (1 + rel[0] * rng0.standard_normal(C)), 0.001, None), np.nan)
        ne_new[j] = np.where(have, np.clip(np.interp(np.sqrt(xn), RG, tab_all[v, i, 1]) * (1 + rel[1] * rng0.standard_normal(C)), 0.01, None), np.nan)
    g = np.nonzero(good)[0]
    vg, ig = vv[g], ii[g]
    out = dict(shot=F["shot"][ig], frame=F["frame"][ig], time=F["time"][ig], variant=vg, psi=psi_all[vg, ig].reshape(-1, len(Z), len(R)),
               coils=F["coils"][ig], ip=F["ip"][ig], tf=F["tf"][ig], Te=Te_new[g].astype(np.float32), ne=ne_new[g].astype(np.float32),
               R_ts=F["R_ts"][ig].astype(np.float32), phase=ph[ig], z_balance=z_bal[ig],
               **{k: lab[k][g] for k in lab}, **{f"gen_{k}": res[k][vg, ig] for k in ("K", "am", "an", "bv", "zc")})
    np.savez_compressed(a.out, **out)
    nsolve = V * n
    print(f"selected {N}, kept {len(g)} equilibria from {n} frames; q95 5/50/95% {np.percentile(lab['q95'][g], [5, 50, 95]).round(2)}, "
          f"betaN {np.percentile(lab['betaN'][g], [5, 50, 95]).round(2)}, li {np.percentile(lab['li'][g], [5, 50, 95]).round(2)}, "
          f"kappa {np.percentile(lab['kappa'][g], [5, 50, 95]).round(2)}")
    print(f"solver time {t_solve:.0f}s for {nsolve} two-pass solves = {1000 * t_solve / nsolve:.0f}s per 1,000 "
          f"({torch.get_num_threads()} threads); total {time.time() - t0:.0f}s = {1000 * (time.time() - t0) / max(len(g), 1):.0f}s "
          f"per 1,000 kept equilibria; wrote {a.out}")


if __name__ == "__main__":
    main()
