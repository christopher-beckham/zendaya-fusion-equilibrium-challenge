#!/usr/bin/env python3
"""Parameterised copy of gsolve examples/generate_synthetic.py: the same label-free recipe, with its settings
exposed as a vector of "knobs" (JSON) so a real-vs-synthetic critic can tune them.

    PYTHONPATH=/fsx/chris.beckham/gsolve_copy python gen_param.py FILES... --out x.npz --knobs knobs.json [--every 4]

Knobs (defaults reproduce generate_synthetic.py, apart from the random-number streams):
  k_lo, k_hi          pressure ratio K = p_total / p_e ~ U(k_lo, k_hi)
  am_lo, am_hi        current-shape exponent am ~ U
  an_lo, an_hi        current-shape exponent an ~ U (before the phase factor)
  w_te                Te-tied current fraction: FF' shape = (1 - w) (1 - psi^am)^an + w ((Te - Te_edge)/(Te0 - Te_edge))^1.5,
                      w ~ U(0, w_te) per equilibrium (0 = off)
  pk_up, pk_down      exponents on the phase peaking factors (an x (Ip/Ip_max)^pk_up in the ramp-up,
                      x (Ip_max/Ip)^pk_down in the ramp-down; 0 = no phase dependence)
  bv_edge             1: vessel field tuned so the outer edge lands on the Thomson edge (default);
                      0: vessel field drawn from N(bv_mu, bv_sd) mT and the edge left free
  bv_mu, bv_sd        prior of the vessel field when bv_edge = 0 (and for frames without Thomson: N(bv_mu, max(bv_sd, 4)))
  edge_bias, edge_sd  outer-edge target = Thomson edge + N(edge_bias, edge_sd) m (bv_edge = 1)
  zc_lo, zc_hi, zc_sd height Z_c = U(zc_lo, zc_hi) x force-balance height + N(0, zc_sd)
  sel_out, sel_in     selection tolerances on the solved midplane edges vs the Thomson edges (m)
  sel_win             weight of the inner-edge error in the selection ranking
  noise_scale         multiplier on the real frame's own relative Thomson scatter
  out_frac, out_sd    fraction of channels with a log-normal outlier factor exp(N(0, out_sd))
  drop_frac           fraction of in-plasma channels reading cold (Te 5-20 eV)
  ne_shift_mu, ne_shift_sd   density profile shifted outward by N(mu, sd) m relative to Te when remapped
  sel_mode            0: keep, at every frame, the variants whose edges match the Thomson edges best (default; a "case" then
                      strings together unrelated draws from frame to frame); 1: per shot, keep the same variant slots
                      (the `keep` variants accepted most often over the shot) at every frame where they pass the
                      tolerances, so with shot_corr > 0 neighbouring frames share their draws
  rg_step             width of the rho bins of the flux-surface table used to REMAP Thomson (default 0.05; 0.025 keeps
                      more of the real profile's peak shape); the pressure table of the solve keeps 0.05 bins
  remap_side          0: the flux-surface Thomson table is the median of inboard and outboard channels (default);
                      1: outboard channels only (the side the model reads), so the outboard profile keeps its own shape
  noise_ne            extra multiplier on the density noise (Te and ne scatter set separately)
  noise_model         0: relative scatter = std over the whole chord of (value - 5-point mean) / mean (gsolve default; the
                      edge channels inflate it 3-4x over the core scatter); 1: robust core scatter = 1.4826 x median
                      |value - 5-point median| / median over channels above 20% of the peak
  remap_k3            1: the flux-surface Thomson tables get gsolve's extra [1/4, 1/2, 1/4] smoothing over rho (default);
                      0: no extra smoothing (the per-bin median only), which keeps the real profile's peakedness
  shot_corr           share of the variance of every random draw that is common to a shot (0 = independent frames)
  remap_mono          1: the flux-surface Thomson table used to REMAP Thomson is forced non-increasing in rho (default);
                      0: not forced (the pressure / current tables of the solve stay non-increasing), so a real profile
                      whose peak is off the solved axis is not lifted into a flat top
  sel_th              weight (m per unit) of the Thomson-consistency mismatch D in the variant ranking: D = median over
                      channels of |remapped noise-free profile - real profile (5-point median)| / real peak, Te + ne;
                      i.e. prefer equilibria on whose flux surfaces the real Thomson profile is a flux function
  th_tol              if > 0: also reject variants with D > th_tol
  n_var               if > 0: number of variants solved per frame (overrides --variants)
  remap_bal           1: each rho bin of the remap table = mean of the inboard median and the outboard median (equal
                      weight per side, whatever the channel counts); 0: one median over all channels (default)
  need_tab            1: a variant of a frame with Thomson is accepted only if its Thomson remap table exists (otherwise the
                      kept equilibrium carries no Thomson although the real frame has it); 0: old behaviour
  kp_hi               centrally peaked extra pressure (fast ions from the beams): p = K p_e (1 + kp (1 - x^2)) / (1 + kp/2),
                      kp ~ U(0, kp_hi) per equilibrium (x = sqrt(psi_N)); 0 = off (pressure proportional to p_e)
Every range is set from physics / documentation / the frame's own inputs; nothing is fitted to MAST labels.
Only input columns of the shot files are read (efit_times / grid only).
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import norm

import gsolve
import noise_model
import cond_prior
import zfeat
from gsolve.forward import FwdConfig, forward_solve, start_current

E = 1.602176634e-19
NX = 61
XG = np.linspace(0.0, 1.5, 76)
RG = np.linspace(0.0, 1.5, 31)
RLINE = np.linspace(0.2, 1.55, 271)

DEFAULTS = dict(k_lo=1.8, k_hi=3.5, am_lo=1.0, am_hi=3.0, an_lo=0.5, an_hi=3.0, w_te=0.0, pk_up=1.0, pk_down=1.0,
                bv_edge=1, bv_mu=0.0, bv_sd=4.0, edge_bias=0.0, edge_sd=0.015, zc_lo=0.0, zc_hi=1.0, zc_sd=0.015,
                sel_out=0.05, sel_in=0.08, sel_win=0.5, noise_scale=1.0, out_frac=0.0, out_sd=0.5, drop_frac=0.0,
                ne_shift_mu=0.0, ne_shift_sd=0.0, shot_corr=0.0, noise_model=0, remap_k3=1, noise_ne=1.0, sel_mode=0, rg_step=0.05, remap_side=0,
                remap_mono=1, sel_th=0.0, th_tol=0.0, n_var=0, kp_hi=0.0, need_tab=0, remap_bal=0,
                cond=0, cond_model="/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl", d_li=0.0, d_lk=0.0, c_s=1.0, c_model="gbm",
                sel_crit=0.0, sel_crit_model="/fsx/chris.beckham/gsolve_data/tune3/sel_critic.pkl")


# ---------------------------------------------------------------- reading one challenge file (input columns only)

def _interp(t, v, te):
    t, v = np.asarray(t, np.float64), np.asarray(v, np.float64)
    ok = np.isfinite(t) & np.isfinite(v)
    return np.interp(te, t[ok], v[ok]) if ok.sum() >= 2 else np.full(len(te), np.nan)


def _thomson(times, Te, ne, te_frames, window=20.0):
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
    ipa = np.abs(_interp(tm, ipraw, te))
    rr = (np.abs(_interp(tm, ipraw, te + 20.0)) - np.abs(_interp(tm, ipraw, te - 20.0))) / 0.040 / np.maximum(ipa, 1.0)
    out = dict(shot=Path(path).stem.replace("mast_shot_", ""), time=te, sign=sign, ip=_interp(tm, ipraw, te), rr=rr,
               tf=_interp(tm, r[m.tf_input], te), coils=np.stack([_interp(tm, r[c], te) for c in cols], 1),
               ip_trace=(np.asarray(tm, np.float64), ipraw))
    Te, ne, R = [], [], []
    if r["thomson_core_Te"]:
        Rc = np.asarray(r["thomson_core_R"], np.float64)
        t_, n_ = _thomson(r["thomson_core_times"], r["thomson_core_Te"], r["thomson_core_ne"], te)
        t_, n_ = (t_[:, 1:], n_[:, 1:]) if t_.shape[1] == len(Rc) + 1 else (t_, n_)
        Te.append(t_); ne.append(n_); R.append(Rc)
    if r["thomson_edge_Te"]:
        t_, n_ = _thomson(r["thomson_edge_times"], r["thomson_edge_Te"], r["thomson_edge_ne"], te)
        if t_.shape[1] == len(r["thomson_edge_spatial"]):
            Te.append(t_); ne.append(n_); R.append(np.asarray(r["thomson_edge_spatial"], np.float64))
    out["Te"], out["ne"] = (np.concatenate(v, 1) if v else np.zeros((len(te), 0)) for v in (Te, ne))
    out["R_ts"] = np.concatenate(R) if R else np.zeros(0)
    out["n_core"] = len(R[0]) if r["thomson_core_Te"] else 0
    return out


# ---------------------------------------------------------------- recipe pieces (as generate_synthetic.py)

def ts_table(te, ne, r):
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


def flux_tables(row, psax, psb, Rg, Rch, te, ne, k3=True, grid=None, side=0, mono=True, bal=False):
    RG = globals()["RG"] if grid is None else grid
    if side == 1:                                     # outboard of the solved axis only
        te = np.where(np.asarray(Rch) >= Rg[int(np.argmax(row))] - 0.02, te, np.nan)
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(Rch)
    if ok.sum() < 20 or not psax > psb:
        return None
    rho = np.sqrt(np.clip((psax - np.interp(Rch[ok], Rg, row)) / (psax - psb), 0, None))
    h = 0.5 * (RG[1] - RG[0])
    b = np.digitize(rho, np.concatenate([RG - h, [RG[-1] + h]])) - 1
    if bal:
        ob = np.asarray(Rch)[ok] >= Rg[int(np.argmax(row))]
        def bmed(v):
            out = np.full(len(RG), np.nan)
            for k in range(len(RG)):
                m_ = [np.median(v[(b == k) & sd]) for sd in (ob, ~ob) if ((b == k) & sd).any()]
                if m_:
                    out[k] = np.mean(m_)
            return out
        T, N = bmed(te[ok]), bmed(ne[ok])
    else:
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


def th_mismatch(row, psax, psb, Rg, Rch, te, ne, grid, tab):
    """Thomson-consistency mismatch of one solved equilibrium: the real profile remapped through the flux-surface table
    vs the real profile itself (5-point running median along R), median over channels, relative to the peak; Te + ne."""
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(Rch)
    if ok.sum() < 20 or not psax > psb:
        return np.nan
    o = np.argsort(Rch[ok])
    rho = np.sqrt(np.clip((psax - np.interp(Rch[ok][o], Rg, row)) / (psax - psb), 0, None))
    d = 0.0
    for q, t in ((te[ok][o], tab[0]), (ne[ok][o], tab[1])):
        sm = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(q, 2, mode="edge"), 5), axis=1)
        d += np.median(np.abs(np.interp(rho, grid, t) - sm)) / max(sm.max(), 1e-6)
    return float(d)


def pressure_table(pe_of_x, K):
    pe = np.maximum.accumulate(pe_of_x[::-1])[::-1]
    return K * np.clip(pe - pe[-1], 0, None)


def te_shape(te_of_x):
    """((Te - Te_edge) / (Te0 - Te_edge))^1.5 on the psi_N grid, non-increasing; None if degenerate."""
    t = np.maximum.accumulate(np.asarray(te_of_x, np.float64)[::-1])[::-1]
    d = t[0] - t[-1]
    if not np.isfinite(d) or d <= 0:
        return None
    return np.clip((t - t[-1]) / d, 0, 1) ** 1.5


def midplane_edges(psi, psax, psb, R, Z):
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


class Draws:
    """Uniform / normal draws for one variant with a shot-level shared component (Gaussian copula)."""

    def __init__(self, seed, v, shot_ids, corr):
        self.rng = np.random.default_rng([seed, v, 0])
        self.sid = shot_ids
        self.c = float(np.clip(corr, 0.0, 1.0))
        self.ns = int(shot_ids.max()) + 1
        self.n = len(shot_ids)

    def normal(self):
        zs = self.rng.standard_normal(self.ns)[self.sid]
        zf = self.rng.standard_normal(self.n)
        return np.sqrt(self.c) * zs + np.sqrt(1 - self.c) * zf

    def uniform(self, lo, hi):
        return lo + (hi - lo) * norm.cdf(self.normal())


def _sel_init():
    import d3d_critic as dc
    L = np.load("/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn/geom_mask/mast_mask_limiter.npz")
    dc.set_grid(L["r"], L["z"], L["mask"])


def frame_z(F, tabs, has, m, traces=None):
    """(n, len(zfeat.FEATS)) machine-independent input features per frame (NaN without a Thomson table): the Thomson
    profile on x (the ts_table x: distance from the Te peak over the in/out half-widths), a and the centre from the
    Thomson edges, B0 from the TF current at that centre, Ip, its ramp rate and the shot phase."""
    out = np.full((len(has), len(zfeat.FEATS_ALL)), np.nan)      # FEATS, then the optional Ip-history HFEATS
    for i in np.nonzero(has)[0]:
        t = tabs[i]
        a_ = 0.5 * (t[2] - t[1]); R0c = 0.5 * (t[2] + t[1])
        tq, nq = np.interp(zfeat.XQ, XG, t[3]), np.interp(zfeat.XQ, XG, t[4])
        B0 = float(m.fvac(F["tf"][i])) / R0c
        z = zfeat.features(tq, nq, F["ip"][i], F["ip_frac"][i], F["phase"][i], F["rr"][i], B0, a_, R0c)
        out[i, :len(zfeat.FEATS)] = [z[k] for k in zfeat.FEATS]
        if traces is not None and "hidx" in F:
            h = zfeat.hist_features(F["time"][i], *traces[F["hidx"][i]], zfeat.tau_cd(z["tav"], a_))
            out[i, len(zfeat.FEATS):] = [h[k] for k in zfeat.HFEATS]
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--files-from", default=None, help="text file with one shot path per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knobs", default=None, help="JSON file or JSON string of knob overrides")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-iter", type=int, default=160)
    ap.add_argument("--pass2-iter", type=int, default=120)
    ap.add_argument("--tol", type=float, default=3e-3)
    ap.add_argument("--solenoid-scale", type=float, default=0.5)
    ap.add_argument("--no-psi", action="store_true", help="do not store the flux maps (tuning runs)")
    ap.add_argument("--no-labels", action="store_true", help="skip the equilibrium numbers (faster)")
    a = ap.parse_args()
    files = list(a.files)
    if a.files_from:
        files += [ln.strip() for ln in open(a.files_from) if ln.strip()]
    kn = dict(DEFAULTS)
    if a.knobs:
        kn.update(json.loads(Path(a.knobs).read_text()) if Path(a.knobs).exists() else json.loads(a.knobs))
    # rho grid of the Thomson remap (rg_step); the pressure / current tables always use the 0.05 grid RG
    RGF = RG if np.isclose(kn["rg_step"], 0.05) else np.linspace(0.0, 1.5, int(round(1.5 / kn["rg_step"])) + 1)
    t0 = time.time()
    m = gsolve.load_machine("mast", solenoid_scale=a.solenoid_scale)
    R, Z = m.R, m.Z
    rng0 = np.random.default_rng([a.seed, 99])

    F = {k: [] for k in ("shot", "frame", "time", "coils", "ip", "tf", "sign", "phase", "ip_frac", "Te", "ne", "R_ts", "n_core", "rr",
                         "hidx")}
    traces = []
    for f in files:
        s = read_shot(f, m)
        traces.append(s["ip_trace"])
        aip = np.abs(s["ip"])
        hi = np.nonzero(aip >= 0.9 * np.nanmax(aip))[0]
        phase = np.ones(len(aip), int); phase[:hi[0]] = 0; phase[hi[-1] + 1:] = 2
        ip_frac = aip / np.nanmax(aip)
        sel = np.nonzero((aip >= 100.0) & np.isfinite(s["coils"]).all(1) & np.isfinite(s["tf"]))[0][::a.every]
        for i in sel:
            for k, v in (("shot", s["shot"]), ("frame", i), ("time", s["time"][i]), ("coils", s["coils"][i]), ("ip", s["ip"][i]),
                         ("tf", s["tf"][i]), ("sign", s["sign"]), ("phase", phase[i]), ("ip_frac", ip_frac[i]), ("Te", s["Te"][i]),
                         ("ne", s["ne"][i]), ("R_ts", s["R_ts"]), ("n_core", s["n_core"]), ("rr", s["rr"][i]),
                         ("hidx", len(traces) - 1)):
                F[k].append(v)
    n = len(F["ip"])
    C = max(len(r) for r in F["R_ts"])
    pad = lambda v: np.concatenate([v, np.full(C - len(v), np.nan)])  # noqa: E731
    F = {k: (np.stack([pad(x) for x in v]) if k in ("Te", "ne", "R_ts") else np.array(v)) for k, v in F.items()}
    _, shot_id = np.unique(F["shot"], return_inverse=True)
    ph, ipA = F["phase"], np.abs(F["ip"]) * 1e3
    I = F["coils"] * F["sign"][:, None]
    tabs = [ts_table(F["Te"][i], F["ne"][i], F["R_ts"][i]) for i in range(n)]
    has = np.array([t is not None for t in tabs])
    r_in, r_out = (np.array([t[j] if t else np.nan for t in tabs]) for j in (1, 2))
    pe_x = np.stack([t[3] * t[4] * 1e19 * 1e3 * E if t else np.zeros(len(XG)) for t in tabs])
    te_x = np.stack([t[3] if t else np.zeros(len(XG)) for t in tabs])
    zf = frame_z(F, tabs, has, m, traces)
    cp = cond_prior.load(kn["cond_model"]) if kn.get("cond", 0) >= 0.5 else None
    ext0 = gsolve.coil_flux(m, I).reshape(n, -1)
    Rc0 = np.where(has, 0.5 * (r_in + r_out), 0.88)
    a0 = np.clip(np.where(has, 0.5 * (r_out - r_in), 0.57), 0.3, 0.68)
    z_bal = np.concatenate([gsolve.balance_height(m, ext0[j:j + a.chunk], Rc0[j:j + a.chunk], a0[j:j + a.chunk])
                            for j in range(0, n, a.chunk)])
    z_bal = np.nan_to_num(z_bal, nan=0.0)
    ipf = np.clip(F["ip_frac"], 1 / 3, 1.0)
    fpk = np.select([ph == 0, ph == 2], [ipf ** kn["pk_up"], ipf ** (-kn["pk_down"])], 1.0)
    print(f"{len(files)} files, {n} frames (phases {np.bincount(ph, minlength=3)}), Thomson on {has.mean():.0%}; knobs "
          f"{json.dumps(kn)}", flush=True)

    RR = np.meshgrid(R, Z)[0].ravel()
    Vbv = 1e-3 * (RR ** 2 - m.R0 ** 2) / 2
    xq = np.sqrt(np.linspace(0, 1, NX))
    cfg1 = FwdConfig(n_iter=a.n_iter, tol=a.tol, bv_max=40.0)
    cfg2 = FwdConfig(n_iter=a.pass2_iter, tol=a.tol, bv_max=40.0)
    V = int(kn["n_var"]) if kn.get("n_var", 0) > 0 else a.variants
    keys = ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam", "bv", "Rin", "Rout", "K", "am", "an", "zc", "w", "dne", "D", "kp", "li_t")
    res = {k: np.full((V, n), np.nan) for k in keys}
    psi_all = np.zeros((V, n, R.size * Z.size), np.float32)
    p_all = np.zeros((V, n, NX)); s_all = np.full((V, n, NX), np.nan)
    tab_all = np.full((V, n, 2, len(RGF)), np.nan)
    edge_mode = bool(round(kn["bv_edge"]))
    for v in range(V):
        dr = Draws(a.seed, v, shot_id, kn["shot_corr"])
        K = dr.uniform(kn["k_lo"], max(kn["k_hi"], kn["k_lo"]))
        am = dr.uniform(kn["am_lo"], max(kn["am_hi"], kn["am_lo"]))
        an = dr.uniform(kn["an_lo"], max(kn["an_hi"], kn["an_lo"])) * fpk
        li_t = np.full(n, np.nan)
        if cp is not None:                    # conditional prior: (li*, log K) ~ N(f(z) + delta, s^2 Sigma) where z exists
            Kc, amc, anc, lic = cond_prior.draw(cp, zf, kn, dr, machine="mast")
            hz = np.isfinite(lic)
            K, am, an, li_t = np.where(hz, Kc, K), np.where(hz, amc, am), np.where(hz, anc, an), lic
        w = dr.uniform(0.0, kn["w_te"]) if kn["w_te"] > 0 else np.zeros(n)
        bvp = kn["bv_mu"] + kn["bv_sd"] * dr.normal()
        bv = np.where(has, 0.0 if edge_mode else bvp, kn["bv_mu"] + max(kn["bv_sd"], 4.0) * dr.normal())
        r_tgt = np.where(has, r_out + kn["edge_bias"] + kn["edge_sd"] * dr.normal(), np.nan) if edge_mode else np.full(n, np.nan)
        zc = dr.uniform(kn["zc_lo"], max(kn["zc_hi"], kn["zc_lo"])) * z_bal + kn["zc_sd"] * dr.normal()
        dne = kn["ne_shift_mu"] + kn["ne_shift_sd"] * dr.normal()
        betap = dr.uniform(0.05, 1.0)
        kp = dr.uniform(0.0, kn["kp_hi"]) if kn["kp_hi"] > 0 else np.zeros(n)
        fk = (1 + kp[:, None] * (1 - xq[None] ** 2)) / (1 + kp[:, None] / 2)
        p_tab = np.zeros((n, NX)); s_tab = np.full((n, NX), np.nan)
        for i in range(n):
            if has[i]:
                p_tab[i] = pressure_table(np.interp(xq, XG, pe_x[i]), K[i]) * fk[i]
                if w[i] > 0:
                    st = te_shape(np.interp(xq, XG, te_x[i]))
                    if st is not None:
                        s_tab[i] = st
            else:
                Bp = gsolve.geometry.MU0 * ipA[i] / (2 * np.pi * 0.55 * np.sqrt((1 + 1.8 ** 2) / 2))
                p_tab[i] = 2.5 * betap[i] * Bp ** 2 / (2 * gsolve.geometry.MU0) * (1 - xq ** 2) ** 1.5
        xg = np.linspace(0, 1, NX)

        def blend(i_sl):
            """FF' shape table: (1 - w) (1 - x^am)^an + w Te-shape, or NaN rows (pure analytic shape)."""
            out = np.full((len(i_sl), NX), np.nan)
            for q, i in enumerate(i_sl):
                if w[i] > 0 and np.isfinite(s_tab[i]).all():
                    out[q] = (1 - w[i]) * (1 - xg ** am[i]).clip(0) ** an[i] + w[i] * s_tab[i]
            return out
        for i0 in range(0, n, a.chunk):
            sl = slice(i0, min(i0 + a.chunk, n)); B = sl.stop - sl.start
            idx = np.arange(sl.start, sl.stop)
            ext = torch.as_tensor(ext0[sl] + bv[sl, None] * Vbv[None]).reshape(B, len(Z), len(R))
            J0 = start_current(m, Rc0[sl], zc[sl], a0[sl], np.full(B, 1.8), np.full(B, 0.3))
            prof = dict(mode="tab", p_tab=p_tab[sl], alpha_m=am[sl], alpha_n=an[sl], s_tab=blend(idx))
            rh = np.where(has[sl], np.nan_to_num(r_tgt[sl], nan=1.3), np.nan) if edge_mode else None
            s1 = forward_solve(m, ext, ipA[sl], prof, cfg1, z_hold=zc[sl], J0=J0, r_hold=rh)
            for j in np.nonzero((s1.status.numpy() == 0) & has[sl])[0]:
                i = sl.start + j
                tb = flux_tables(s1.psi[j].numpy()[np.argmin(np.abs(Z))], float(s1.psi_ax[j]), float(s1.psi_b[j]), R,
                                 F["R_ts"][i], F["Te"][i], F["ne"][i], k3=bool(round(kn["remap_k3"])),
                                 side=int(round(kn["remap_side"])))
                if tb is not None:
                    tbf = tb if (RGF is RG and kn["remap_mono"] >= 0.5 and kn["remap_bal"] < 0.5) else flux_tables(s1.psi[j].numpy()[np.argmin(np.abs(Z))], float(s1.psi_ax[j]),
                                                           float(s1.psi_b[j]), R, F["R_ts"][i], F["Te"][i], F["ne"][i],
                                                           k3=bool(round(kn["remap_k3"])), grid=RGF,
                                                           side=int(round(kn["remap_side"])),
                                                           mono=kn["remap_mono"] >= 0.5, bal=kn["remap_bal"] >= 0.5)
                    if tbf is not None:
                        tab_all[v, i] = tbf
                    p_tab[i] = pressure_table(np.interp(xq, RG, tb[0] * tb[1] * 1e19 * 1e3 * E), K[i]) * fk[i]
                    if w[i] > 0:
                        st = te_shape(np.interp(xq, RG, tb[0]))
                        if st is not None:
                            s_tab[i] = st
            prof["p_tab"] = p_tab[sl]; prof["s_tab"] = blend(idx)
            s2 = forward_solve(m, ext, ipA[sl], prof, cfg2, z_hold=zc[sl], J0=J0, r_hold=rh, psi_start=s1.psi,
                               bv0=s1.bv if edge_mode else None)
            psi_all[v, sl] = s2.psi.reshape(B, -1).numpy()
            for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam"):
                res[k][v, sl] = s2.info[k].double().numpy()
            res["bv"][v, sl] = bv[sl] + (s2.bv.numpy() if edge_mode else 0.0)
            s_all[v, sl] = prof["s_tab"]
            for j in np.nonzero(s2.status.numpy() == 0)[0]:
                res["Rin"][v, sl.start + j], res["Rout"][v, sl.start + j] = midplane_edges(
                    s2.psi[j].numpy(), float(s2.psi_ax[j]), float(s2.psi_b[j]), R, Z)
                i = sl.start + j
                if has[i] and np.isfinite(tab_all[v, i]).all():
                    res["D"][v, i] = th_mismatch(s2.psi[j].numpy()[np.argmin(np.abs(Z))], float(s2.psi_ax[j]),
                                                 float(s2.psi_b[j]), R, F["R_ts"][i], F["Te"][i], F["ne"][i], RGF, tab_all[v, i])
        p_all[v] = p_tab
        res["K"][v], res["am"][v], res["an"][v], res["zc"][v], res["w"][v], res["dne"][v] = K, am, an, zc, w, dne
        res["kp"][v] = kp
        res["li_t"][v] = li_t
        cv = res["status"][v] == 0
        print(f"variant {v}: converged {cv.mean():.0%} ({time.time() - t0:.0f}s)", flush=True)

    ok = (res["status"] == 0) & np.isfinite(res["Rout"])
    dout, din = res["Rout"] - r_out[None], res["Rin"] - r_in[None]
    acc = ok & (np.abs(dout) <= kn["sel_out"]) & (np.abs(din) <= kn["sel_in"])
    Dm = np.where(np.isfinite(res["D"]), res["D"], np.nanmax(np.append(res["D"][np.isfinite(res["D"])], 1.0)))
    if kn["need_tab"] >= 0.5:
        acc &= ~has[None] | np.isfinite(tab_all).all(axis=(2, 3))
    if kn["th_tol"] > 0:
        acc &= Dm <= kn["th_tol"]
    err_all = np.abs(dout) + kn["sel_win"] * np.abs(din) + kn["sel_th"] * Dm
    if kn.get("sel_crit", 0) > 0:
        # critic-guided selection (tune3/sel_critic.py): err -= sel_crit * logit p(real | z, flux geometry), the DIII-D-trained
        # flux-geometry critic evaluated on every accepted variant's map
        import pickle
        from concurrent.futures import ProcessPoolExecutor
        import d3d_critic as dc
        SM = pickle.loads(Path(kn["sel_crit_model"]).read_bytes())
        vi = np.argwhere(acc)
        args = [(-psi_all[v, i].astype(np.float64).reshape(len(Z), len(R)), 0.0, float(m.fvac(F["tf"][i]))) for v, i in vi]
        with ProcessPoolExecutor(max(1, torch.get_num_threads()), initializer=_sel_init) as ex:
            Gg = list(ex.map(dc.geom, args, chunksize=16))
        zc_ = zf[:, [zfeat.FEATS_ALL.index(k) for k in dc.ZFE]]
        a_s = np.clip(0.5 * (r_out - r_in), 0.3, 1.0)
        Xs = np.c_[zc_[vi[:, 1]], z_bal[vi[:, 1]] / a_s[vi[:, 1]], np.array([[g[k] for k in SM["feats"]] for g in Gg])].astype(np.float32)
        ps = np.clip(SM["model"].predict_proba(Xs)[:, 1], 1e-3, 1 - 1e-3)
        lg = np.full(acc.shape, 0.0); lg[vi[:, 0], vi[:, 1]] = np.log(ps / (1 - ps))
        err_all = err_all - kn["sel_crit"] * lg
        print(f"selection critic on {len(vi)} accepted variants: p(real) median {np.median(ps):.3f}", flush=True)
    na = acc[:, has].sum(0)
    print(f"accepted variants per Thomson frame: {np.bincount(na, minlength=V + 1).tolist()}; D median "
          f"{np.nanmedian(res['D']):.3f}, within-frame sd {np.nanmean(np.nanstd(np.where(acc, res['D'], np.nan), 0)):.3f}", flush=True)
    kept, rank = [], []
    slots = {}
    if int(round(kn["sel_mode"])) == 1:
        err = err_all
        for sid in np.unique(shot_id):
            fr = np.nonzero((shot_id == sid) & has)[0]
            if not len(fr):
                continue
            cnt = acc[:, fr].sum(1)
            me = np.array([np.nanmean(np.where(acc[v, fr], err[v, fr], np.nan)) if cnt[v] else np.inf for v in range(V)])
            order = sorted(range(V), key=lambda v: (-cnt[v], me[v]))
            slots[sid] = [v for v in order[:a.keep] if cnt[v] > 0]
    for i in range(n):
        if has[i]:
            if slots:
                for k_, v in enumerate(slots.get(shot_id[i], [])):
                    if acc[v, i]:
                        kept.append((v, i)); rank.append(k_)
            else:
                c = np.nonzero(acc[:, i])[0]
                c = c[np.argsort(err_all[c, i])][:a.keep]
                kept += [(v, i) for v in c]; rank += list(range(len(c)))
        elif ok[:, i].any():
            kept.append((int(rng0.choice(np.nonzero(ok[:, i])[0])), i)); rank.append(0)
    vv, ii = np.array([k[0] for k in kept], int), np.array([k[1] for k in kept], int)
    rank = np.array(rank, int)
    N = len(kept)
    s_kept = s_all[vv, ii]
    prof_k = dict(mode="tab", p_tab=torch.as_tensor(p_all[vv, ii]), alpha_m=torch.as_tensor(res["am"][vv, ii]),
                  alpha_n=torch.as_tensor(res["an"][vv, ii]), s_tab=torch.as_tensor(s_kept))
    if a.no_labels:
        lab = {k: np.full(N, np.nan) for k in ("R_ax", "Z_ax", "xpt_R", "xpt_Z", "kappa", "delta_upper", "delta_lower",
                                                 "volume", "q95", "betaN", "li")}
        good = np.ones(N, bool)
    else:
        sol = gsolve.Solution(m, torch.as_tensor(psi_all[vv, ii].astype(np.float64)).reshape(N, len(Z), len(R)),
                              torch.as_tensor(ipA[ii]), prof_k,
                              {k: torch.as_tensor(res[k][vv, ii]) for k in ("status", "psi_ax", "psi_b", "R_ax", "Z_ax", "lam")})
        lab = gsolve.analyse_all(sol, np.abs(F["tf"][ii]))
        good = np.isfinite(lab["q95"]) & np.isfinite(lab["betaN"]) & (lab["q95"] > 1) & (lab["betaN"] > 0.02)
    # Thomson remapped onto the solved surfaces, with noise / outliers / cold channels
    Te_new, ne_new = np.full((N, C), np.nan), np.full((N, C), np.nan)
    Te_cl, ne_cl = np.full((N, C), np.nan), np.full((N, C), np.nan)            # noise-free remapped profiles
    relT_k, relN_k, inside_k = np.full(N, np.nan), np.full(N, np.nan), np.zeros((N, C), bool)
    iz = np.argmin(np.abs(Z))
    for j in np.nonzero(good)[0]:
        v, i = vv[j], ii[j]
        if not np.isfinite(tab_all[v, i]).all():
            continue
        have = np.isfinite(F["Te"][i]) & np.isfinite(F["ne"][i]) & np.isfinite(F["R_ts"][i])
        row = psi_all[v, i].reshape(len(Z), len(R))[iz]
        den = res["psi_ax"][v, i] - res["psi_b"][v, i]
        rho_T = np.sqrt(np.clip((res["psi_ax"][v, i] - np.interp(F["R_ts"][i], R, row)) / den, 0, None))
        # density shifted outward by dne relative to the temperature (outboard and inboard alike in R)
        rho_n = np.sqrt(np.clip((res["psi_ax"][v, i] - np.interp(F["R_ts"][i] - res["dne"][v, i], R, row)) / den, 0, None))
        o = np.argsort(F["R_ts"][i][have])
        rel = []
        for q, floor in ((F["Te"][i], 0.02), (F["ne"][i], 0.05)):
            qq = q[have][o]
            if have.sum() <= 10:
                rel.append(0.05)
            elif int(round(kn["noise_model"])) == 0:
                sm = np.convolve(qq, np.ones(5) / 5, mode="same")
                rel.append(np.clip(np.nanstd((qq - sm)[2:-2] / np.maximum(sm[2:-2], floor)), 0, 0.3))
            else:
                sm = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(qq, 2, mode="edge"), 5), axis=1)
                core = sm >= 0.2 * sm.max()
                rel.append(np.clip(1.4826 * np.median(np.abs(qq - sm)[core] / sm[core]), 0, 0.3) if core.sum() > 5 else 0.05)
        cT = np.interp(rho_T, RGF, tab_all[v, i, 0])
        cN = np.interp(rho_n, RGF, tab_all[v, i, 1])
        tT, tN = noise_model.apply(cT, cN, rel[0], rel[1], kn, rho_T < 0.9, rng0)
        Te_cl[j] = np.where(have, cT, np.nan); ne_cl[j] = np.where(have, cN, np.nan)
        relT_k[j], relN_k[j], inside_k[j] = rel[0], rel[1], rho_T < 0.9
        Te_new[j] = np.where(have, tT, np.nan)
        ne_new[j] = np.where(have, tN, np.nan)
    g = np.nonzero(good)[0]
    vg, ig = vv[g], ii[g]
    out = dict(shot=F["shot"][ig], frame=F["frame"][ig], time=F["time"][ig], variant=vg, rank=rank[g],
               coils=F["coils"][ig], ip=F["ip"][ig], tf=F["tf"][ig], Te=Te_new[g].astype(np.float32), ne=ne_new[g].astype(np.float32),
               Te_clean=Te_cl[g].astype(np.float32), ne_clean=ne_cl[g].astype(np.float32), rel_te=relT_k[g], rel_ne=relN_k[g],
               inside=inside_k[g], R_ts=F["R_ts"][ig].astype(np.float32), n_core=F["n_core"][ig], phase=ph[ig], z_balance=z_bal[ig],
               R_ax=res["R_ax"][vg, ig], Z_ax=res["Z_ax"][vg, ig],
               **{k: lab[k][g] for k in lab if k not in ("R_ax", "Z_ax")},
               **{f"gen_{k}": res[k][vg, ig] for k in ("K", "am", "an", "bv", "zc", "w", "dne", "D", "kp", "li_t")},
               **{f"z_{k}": zf[ig, j] for j, k in enumerate(zfeat.FEATS)},
               knobs=np.array(json.dumps(kn)),
               n_frames=np.int64(n), n_has=np.int64(has.sum()), n_frames_kept=np.int64(len(set(ii[g].tolist()))))
    if not a.no_labels:
        out["R_ax"], out["Z_ax"] = lab["R_ax"][g], lab["Z_ax"][g]
    if not a.no_psi:
        out["psi"] = psi_all[vg, ig].reshape(-1, len(Z), len(R))
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    Path(tmp).replace(a.out)
    lq = lambda k: np.percentile(lab[k][g], [5, 50, 95]).round(2) if not a.no_labels and len(g) else None  # noqa: E731
    print(f"kept {len(g)} equilibria from {n} frames ({len(set(ii[g].tolist()))} frames with >=1); q95 {lq('q95')}, "
          f"betaN {lq('betaN')}, li {lq('li')}, kappa {lq('kappa')}; total {time.time() - t0:.0f}s; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
