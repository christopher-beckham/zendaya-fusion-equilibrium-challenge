"""gen4: synthetic MAST shots for training, calibrated on MAST inputs only.

    python gen4.py --test 3                                   # 3 shots in this process, verbose, output under OUT/test
    python gen4.py --run --node-tag n1 --n-workers 60 --n-train-real 8 --n-eval-real 2   # watchdog pool, claims tasks
    python gen4.py --summary

Each task is (realisation r, real mast_public_test shot). Every EFIT frame of that shot is solved in time order with
FreeGSNKE (warm start from the previous frame) on the challenge MAST coil set, and the result is written as ONE parquet
file in the challenge MAST schema: the real input time series (coil groups scaled by small per-shot factors, none for
r = 0), synthetic Thomson profiles consistent with the solved equilibrium, and EFIT-style labels at the frames that solved.

Plasma model per frame (everything not listed is the real input of that frame):
  * pressure  p(x) = (1 + Ti/Te) p_e(x) + p_fast(x), x = normalised flux.
      p_e(x) = e ne(x) Te(x): monotone fits of the frame's REAL Thomson ne and Te mapped onto the solution's own flux
      surfaces (self-consistent: map, solve, re-map, re-solve).
      Ti/Te and the fast-ion share (p_fast = f/(1-f) (1+Ti/Te) p_e(0) (1 - x^2)^nf) are not measured by Thomson: drawn
      per shot, drifting slowly in time (Ornstein-Uhlenbeck, 60 ms).
  * current   FF'(x) = L h(x), h = (1 - w) (1 - x^am)^an + w (Te(x)/Te(0))^1.5 (Spitzer-like part from the same Thomson
      Te fit); w, am, an drawn per shot, w and an drift slowly; L set so the total current equals the frame's Ip.
  * synthetic Thomson at every real Thomson time with a solved frame within 6 ms: the slice's own real ne, Te fitted on
    the nearest solved frame's flux surfaces, evaluated at the chord positions, scrapeoff-layer decay outside the
    boundary, multiplicative noise at the slice's measured scatter, and the real slice's missing channels kept missing.
    Thomson therefore sees only the electron pressure, with real noise and gaps, never the total pressure of the label.
  * solenoid at x0.5 of the column (challenge note: the column reads 2x FAIR-MAST); TF 24 turns.

Rule 3: reads only MAST inputs (magnetics_* currents, thomson_*), geometry (coil_*, thomson_chord_*, efit_grid_R/Z) and
efit_times (time stamps). No efit_* label column of any MAST file, no demo shot, no MAST envelope mask. Every label is
computed from the FreeGSNKE solution. The cases are not fitted to any real shot's magnetic measurements.
Split: shots in gen_m2's 10 % hash pool (the MASTVAL shots) are held out and divided by a second hash into val and test.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/fsx/zara.ercan/more_data/generated/code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mast_machine as mm      # noqa: E402
import synth_common as sc      # noqa: E402

VERSION = "gen4.0"
DATA = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/challenge_data/data/mast_public_test"
MACHINE_DIR = "/fsx/zara.ercan/more_data/generated/machines/MAST_challenge"
OUT = "/fsx/zara.ercan/more_data/m2/gen4"
SIG = mm.COIL_COLUMNS + ["magnetics_plasma_current", "magnetics_tf_current", "magnetics_efps_current"]
VERT = ["magnetics_p4l_current", "magnetics_p4u_current", "magnetics_p5l_current", "magnetics_p5u_current"]
DIV = ["magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current", "magnetics_p3u_current",
       "magnetics_p6l_current", "magnetics_p6u_current"]
SOL_FACTOR = 0.5
SEED = 44001
TOL = 1e-5
MU0 = 4e-7 * np.pi
E = 1.602176634e-19
XS = np.linspace(0.0, 1.0, 51)
TAU_MS = 60.0                 # Ornstein-Uhlenbeck time constant of the drifting profile parameters
TS_MAX_DT = 20.0              # ms: Thomson slice used for an EFIT frame's pressure table
SYN_MAX_DT = 6.0              # ms: Thomson slice synthesised from the nearest solved frame
MIN_CH = 20                   # valid core channels for a usable Thomson slice
VAL_FRAC = 0.10


# ------------------------------------------------------------------------------------------------------------------
# split (identical val pool to gen_m2.is_val_shot, i.e. the MASTVAL shots)
# ------------------------------------------------------------------------------------------------------------------
def split_of(fname: str) -> str:
    if int(hashlib.sha1(fname.encode()).hexdigest()[:8], 16) % 1000 >= int(VAL_FRAC * 1000):
        return "train"
    return "test" if int(hashlib.sha1((fname + "|test").encode()).hexdigest()[:8], 16) % 2 else "val"


# ------------------------------------------------------------------------------------------------------------------
# profile: tables for p(x) and FF'(x)
# ------------------------------------------------------------------------------------------------------------------
def _table_profile_class():
    from freegsnke.jtor_update import ConstrainPaxisIp

    class TableProfile(ConstrainPaxisIp):
        """p(x) = paxis g(x), FF'(x) = mu0 L h(x) (Raxis = 1), J = R p'(psi) + FF'/(mu0 R); L from Ip."""

        def __init__(self, eq, paxis, Ip, fvac, xs, g, h):
            super().__init__(eq, paxis=paxis, Ip=Ip, fvac=fvac, alpha_m=1.0, alpha_n=1.0, Raxis=1.0)
            self.set_tables(paxis, xs, g, h)

        def set_tables(self, paxis, xs, g, h):
            self.paxis = float(paxis)
            self.profile_parameter = self.paxis
            self.xs = np.asarray(xs, float)
            self.g = np.asarray(g, float)
            self.dg = np.gradient(self.g, self.xs)
            self.h = np.asarray(h, float)

        def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
            if psi_bndry is None:
                psi_bndry = psi[0, 0]
            self.psi_bndry = psi_bndry
            self.psi_axis = psi_axis
            dR = R[1, 0] - R[0, 0]
            dZ = Z[0, 1] - Z[0, 0]
            xr = (psi - psi_axis) / (psi_bndry - psi_axis)
            inside = (xr >= 0) & (xr < 1)
            x = np.clip(xr, 0.0, 1.0)
            jp = np.where(inside, R * self.paxis * np.interp(x, self.xs, self.dg) / (psi_bndry - psi_axis), 0.0)
            shape = np.where(inside, np.interp(x, self.xs, self.h), 0.0)
            if mask is not None:
                jp = jp * mask
                shape = shape * mask
            I_R = np.sum(shape / R) * dR * dZ
            Ip_p = np.sum(jp) * dR * dZ
            L = (self.Ip - Ip_p) / I_R if I_R > 0 else 0.0
            J = jp + L / R * shape
            self.jtor = J
            self.L = L
            self.Beta0 = Ip_p / self.Ip
            return J

        def pprime(self, pn):
            pn = np.clip(np.asarray(pn, float), 0, 1)
            return self.paxis * np.interp(pn, self.xs, self.dg) / (self.psi_bndry - self.psi_axis)

        def ffprime(self, pn):
            pn = np.clip(np.asarray(pn, float), 0, 1)
            return MU0 * self.L * np.interp(pn, self.xs, self.h)

        def pressure(self, pn):
            pn = np.clip(np.asarray(pn, float), 0, 1)
            return self.paxis * np.interp(pn, self.xs, self.g)

        def copy(self):
            obj = super().copy()
            for k in ("xs", "g", "dg", "h", "psi_axis", "paxis"):
                if hasattr(self, k):
                    setattr(obj, k, getattr(self, k))
            return obj

    return TableProfile


class Machine:
    """FreeGSNKE equilibrium + static solver on the challenge MAST grid (65 x 65, R 0.06-2.0, Z +-2)."""

    def __init__(self):
        from freegsnke import GSstaticsolver, equilibrium_update
        with sc.quiet():
            self.tok = mm.build_tokamak(MACHINE_DIR)
            self.eq = equilibrium_update.Equilibrium(tokamak=self.tok, Rmin=0.06, Rmax=2.0, Zmin=-2.0, Zmax=2.0, nx=65, ny=65)
            self.gs = GSstaticsolver.NKGSsolver(self.eq)
        self.R1 = self.eq.R[:, 0].copy()
        self.Z1 = self.eq.Z[0, :].copy()
        self.TableProfile = _table_profile_class()

    def set_frame(self, inputs_kA):
        for c, v in mm.coil_currents_A(inputs_kA, SOL_FACTOR).items():
            self.eq.tokamak.set_coil_current(c, v)
        self.Ip = float(inputs_kA["magnetics_plasma_current"]) * 1e3
        self.fvac = mm.fvac_from_tf(inputs_kA["magnetics_tf_current"])

    def profile(self, paxis, g, h):
        return self.TableProfile(self.eq, paxis, self.Ip, self.fvac, XS, g, h)

    def solve(self, prof, max_it=60):
        with sc.quiet():
            self.gs.solve(eq=self.eq, profiles=prof, constrain=None, target_relative_tolerance=TOL,
                          max_solving_iterations=max_it, Picard_handover=0.1)
        rel = float(self.gs.relative_change)
        return rel <= 10 * TOL and bool(np.all(np.isfinite(self.eq.plasma_psi)))

    def psiN_at(self, psi_axis, psi_bndry, R, Z):
        from scipy.interpolate import RectBivariateSpline
        spl = RectBivariateSpline(self.R1, self.Z1, self.eq.psi())
        return (np.asarray(spl(R, Z, grid=False), float) - psi_axis) / (psi_bndry - psi_axis)


# ------------------------------------------------------------------------------------------------------------------
# seed library for cold starts: plasma flux of converged equilibria, looked up by the coil / Ip state
# ------------------------------------------------------------------------------------------------------------------
SEEDLIB = "/fsx/zara.ercan/cache/m2/gen4_seedlib.npz"
SEED_SRC = ["/fsx/zara.ercan/more_data/m2/mast_train/raw", "/fsx/zara.ercan/more_data/m2/mast_train3/raw"]


def seed_key(inp):
    g = lambda a, b: 0.5 * (inp[f"magnetics_{a}_current"] + inp[f"magnetics_{b}_current"])
    return np.array([inp["magnetics_plasma_current"], g("p4l", "p4u"), g("p5l", "p5u"), g("p2l", "p2u"), g("p3l", "p3u"),
                     inp["magnetics_sol_current"], g("p6l", "p6u")], float)


def build_seedlib(n_max=6000):
    """Plasma flux (total - vacuum) of converged gen-1 / gen-3 synthetic equilibria (our own solutions, train-pool shots)."""
    files = []
    for src in SEED_SRC:
        files += sorted(glob.glob(os.path.join(src, "*", "point_*.npz")))
    rng = np.random.default_rng(5)
    files = [files[i] for i in rng.permutation(len(files))]
    M = Machine()
    keys, pp = [], []
    for f in files:
        dd = np.load(f, allow_pickle=True)
        if str(dd["status"]) != "ok":
            continue
        inp = json.loads(str(dd["inputs"]))
        M.set_frame(inp)
        vac = np.asarray(M.eq.tokamak.getPsitokamak(vgreen=M.eq._vgreen)).reshape(65, 65)
        pp.append((np.asarray(dd["psi"], float).T - vac).astype(np.float32))
        keys.append(seed_key(inp))
        if len(keys) >= n_max:
            break
    os.makedirs(os.path.dirname(SEEDLIB), exist_ok=True)
    np.savez(SEEDLIB, keys=np.array(keys), plasma_psi=np.array(pp))
    print(f"seed library: {len(keys)} equilibria -> {SEEDLIB}", flush=True)


class SeedLib:
    def __init__(self, cap=20000):
        d = np.load(SEEDLIB)
        self.keys = list(d["keys"])
        self.psi = list(d["plasma_psi"])
        self.scale = np.std(d["keys"], axis=0) + 1e-6
        self.cap = cap

    def nearest(self, inp, n=2):
        K = np.asarray(self.keys)
        dist = np.sum(((K - seed_key(inp)) / self.scale) ** 2, axis=1)
        return [self.psi[i].astype(np.float64) for i in np.argsort(dist)[:n]]

    def add(self, inp, plasma_psi):
        self.keys.append(seed_key(inp))
        self.psi.append(plasma_psi.astype(np.float32))
        if len(self.keys) > self.cap:
            del self.keys[0], self.psi[0]


# ------------------------------------------------------------------------------------------------------------------
# Thomson: fits on flux surfaces and synthesis
# ------------------------------------------------------------------------------------------------------------------
def pava_dec(y):
    """Unweighted isotonic regression, non-increasing."""
    vals, cnt = [], []
    for yi in -np.asarray(y, float):
        vals.append(yi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            c = cnt[-2] + cnt[-1]
            v = (vals[-2] * cnt[-2] + vals[-1] * cnt[-1]) / c
            vals[-2:] = [v]; cnt[-2:] = [c]
    return -np.concatenate([[v] * c for v, c in zip(vals, cnt)])


def mono_table(x, y, edge_value):
    """Non-increasing y(x) on XS from points with 0 <= x < 1; linear to edge_value(y_last) beyond the last point."""
    o = np.argsort(x)
    x, y = x[o], y[o]
    yi = pava_dec(y)
    t = np.interp(XS, x, yi, left=yi[0], right=yi[-1])
    tail = XS > x[-1]
    if tail.any():
        ye = min(edge_value(yi[-1]), yi[-1])
        t[tail] = yi[-1] + (ye - yi[-1]) * (XS[tail] - x[-1]) / max(1.0 - x[-1], 1e-6)
    s = t.copy()
    s[1:-1] = 0.25 * t[:-2] + 0.5 * t[1:-1] + 0.25 * t[2:]
    return np.minimum.accumulate(np.maximum(s, 0.0))


def kinetic_tables(x, ne, te):
    """ne(x), Te(x) tables (m^-3, eV) from chords inside the boundary, or None."""
    m = np.isfinite(x) & np.isfinite(ne) & np.isfinite(te) & (x >= 0) & (x < 1) & (ne > 0) & (te > 0)
    if m.sum() < 10:
        return None
    ne_t = mono_table(x[m], ne[m], lambda v: 0.6 * v)
    te_t = mono_table(x[m], te[m], lambda v: max(10.0, min(v, 40.0)))
    if ne_t[0] <= 0 or te_t[0] <= 0:
        return None
    return ne_t, te_t


def slice_noise(R, q):
    """Robust relative scatter of a measured profile about its 5-channel running median."""
    m = np.isfinite(q) & (q > 0)
    if m.sum() < 10:
        return 0.1
    r = R[m]; v = q[m]
    o = np.argsort(r); v = v[o]
    med = np.array([np.median(v[max(0, i - 2):i + 3]) for i in range(len(v))])
    rel = (v - med) / np.maximum(med, 1e-30)
    return float(np.clip(1.4826 * np.median(np.abs(rel - np.median(rel))), 0.02, 0.5))


def synth_slice(x, ne_t, te_t, lam_ne, lam_te, sig_ne, sig_te, real_nan, rng):
    """Synthetic (ne, Te) at chords with flux coordinate x; NaN where the real slice is missing."""
    xc = np.clip(x, 0.0, 1.0)
    ne = np.interp(xc, XS, ne_t)
    te = np.interp(xc, XS, te_t)
    out = ~np.isfinite(x) | (x >= 1)
    d = np.where(np.isfinite(x), np.clip(x - 1.0, 0.0, None), 1.0)
    ne = np.where(out, ne_t[-1] * np.exp(-d / lam_ne), ne)
    te = np.where(out, te_t[-1] * np.exp(-d / lam_te), te)
    ne = ne * (1.0 + sig_ne * rng.standard_normal(ne.shape))
    te = np.abs(te * (1.0 + sig_te * rng.standard_normal(te.shape)))
    ne[real_nan] = np.nan
    te[real_nan] = np.nan
    return ne, te


def te_edges(R, te, thr=30.0):
    m = np.isfinite(te) & (te > 0)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan
    r, t = R[m], te[m]
    o = np.argsort(r); r, t = r[o], t[o]
    tm = np.array([np.median(t[max(0, i - 2):i + 3]) for i in range(len(t))])
    hot = np.where(tm > thr)[0]
    if len(hot) < 5:
        return np.nan, np.nan, np.nan
    return float(r[hot[0]]), float(r[hot[-1]]), float(r[int(np.argmax(tm))])


# ------------------------------------------------------------------------------------------------------------------
# per-shot draws
# ------------------------------------------------------------------------------------------------------------------
def shot_rng(r, fname):
    return np.random.default_rng([SEED, r, int(hashlib.sha1(fname.encode()).hexdigest()[:12], 16)])


def ou_path(rng, n, dt_ms, sigma):
    """Stationary OU path (mean 0, sd sigma) sampled at the EFIT times."""
    z = np.zeros(n)
    z[0] = sigma * rng.standard_normal()
    for k in range(1, n):
        a = np.exp(-max(dt_ms[k - 1], 0.0) / TAU_MS)
        z[k] = a * z[k - 1] + sigma * np.sqrt(1 - a * a) * rng.standard_normal()
    return z


def draw_shot(r, fname, t_efit):
    rng = shot_rng(r, fname)
    jit = dict(vert=1.0, div=1.0, sol=1.0, ip=1.0, tf=1.0)
    if r > 0:
        jit = dict(vert=rng.uniform(0.99, 1.01), div=rng.uniform(0.97, 1.03), sol=rng.uniform(0.95, 1.05),
                   ip=rng.uniform(0.985, 1.015), tf=rng.uniform(0.985, 1.015))
    n = len(t_efit)
    dt = np.diff(t_efit) if n > 1 else np.zeros(0)
    tite = np.exp(rng.uniform(np.log(0.7), np.log(1.6)) + ou_path(rng, n, dt, 0.10))
    ff = np.clip(rng.uniform(0.0, 0.5) + ou_path(rng, n, dt, 0.05), 0.0, 0.6)
    nf = rng.uniform(1.0, 3.0)
    w = np.clip(rng.uniform(0.0, 1.0) + ou_path(rng, n, dt, 0.10), 0.0, 1.0)
    am = rng.uniform(1.0, 3.0)
    an = np.exp(rng.uniform(np.log(0.5), np.log(3.5)) + ou_path(rng, n, dt, 0.15))
    lam_ne, lam_te = rng.uniform(0.03, 0.12), rng.uniform(0.02, 0.08)
    return dict(jit=jit, tite=tite, ff=ff, nf=nf, w=w, am=am, an=an, lam_ne=lam_ne, lam_te=lam_te,
                noise_seed=int(rng.integers(2 ** 31)))


def scale_inputs(sig, jit):
    out = dict(sig)
    for c in VERT:
        out[c] = out[c] * jit["vert"]
    for c in DIV:
        out[c] = out[c] * jit["div"]
    out["magnetics_sol_current"] = out["magnetics_sol_current"] * jit["sol"]
    out["magnetics_plasma_current"] = out["magnetics_plasma_current"] * jit["ip"]
    out["magnetics_tf_current"] = out["magnetics_tf_current"] * jit["tf"]
    return out


def profile_tables(ne_t, te_t, tite, ff, nf, w, am, an):
    """(paxis, g, h) from electron tables and the unmeasured parameters."""
    pe = E * ne_t * te_t
    pfast = ff / max(1.0 - ff, 1e-6) * (1.0 + tite) * pe[0] * (1.0 - XS ** 2) ** nf
    p = (1.0 + tite) * pe + pfast
    paxis = float(p[0])
    g = p / paxis
    spitzer = (te_t / te_t[0]) ** 1.5
    h = (1.0 - w) * (1.0 - XS ** am) ** an + w * spitzer
    h = h / max(h[0], 1e-12)
    h[-1] = 0.0
    return paxis, g, h


# ------------------------------------------------------------------------------------------------------------------
# one shot
# ------------------------------------------------------------------------------------------------------------------
def load_shot(fname):
    import pyarrow.parquet as pq
    tab = pq.read_table(os.path.join(DATA, fname))
    row = {c: tab.column(c)[0].as_py() for c in ["efit_times", "magnetics_time", "thomson_core_times", "thomson_core_Te",
                                                 "thomson_core_ne", "thomson_core_R", "thomson_edge_times", "thomson_edge_Te",
                                                 "thomson_edge_ne", "thomson_edge_spatial"] + SIG}
    d = dict(tab=tab, t_efit=np.asarray(row["efit_times"], float), tm=np.asarray(row["magnetics_time"], float))
    d["sig"] = {c: np.asarray(row[c], float) for c in SIG}
    for side in ("core", "edge"):
        d[f"{side}_t"] = np.asarray(row[f"thomson_{side}_times"], float)
        nch = 131 if side == "core" else len(row["thomson_edge_spatial"])
        for q in ("Te", "ne"):
            v = row[f"thomson_{side}_{q}"]
            d[f"{side}_{q}"] = np.asarray(v, float).reshape(len(v), -1) if len(v) else np.zeros((0, nch))
    d["core_R"] = np.asarray(row["thomson_core_R"], float)         # 130 radii; Te/ne rows have 131 (column 0 = ghost)
    d["edge_R"] = np.asarray(row["thomson_edge_spatial"], float)
    return d


def core_slice(d, j):
    return d["core_Te"][j, 1:], d["core_ne"][j, 1:]


def solve_shot(M, fname, r, hang_frames=(), beat=None, verbose=False, lib=None):
    d = load_shot(fname)
    t_efit = d["t_efit"]
    T = len(t_efit)
    P = draw_shot(r, fname, t_efit)
    sig = scale_inputs(d["sig"], P["jit"])
    frames = [dict(status="pending") for _ in range(T)]
    prev = None                      # last accepted frame: (t, plasma_psi, psi_axis, psi_bndry, z_axis, tables)
    R_c = d["core_R"]
    ts_t = d["core_t"]
    for k in range(T):
        t = float(t_efit[k])
        fr = frames[k]
        fr.update(t=t)
        if k in hang_frames:
            fr["status"] = "hang"
            continue
        if beat is not None:
            beat(k)
        t0 = time.time()
        inp = {c: float(np.interp(t, d["tm"], sig[c])) for c in SIG}
        if not (inp["magnetics_plasma_current"] > 100):
            fr["status"] = "low_ip"
            continue
        M.set_frame(inp)
        # Thomson slice for the pressure table
        j = int(np.argmin(np.abs(ts_t - t))) if len(ts_t) else -1
        ts = None
        if j >= 0 and abs(ts_t[j] - t) <= TS_MAX_DT:
            te, ne = core_slice(d, j)
            if np.sum(np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)) >= MIN_CH:
                ts = (te, ne)
        Rin_ts, Rout_ts, Rpk_ts = te_edges(R_c, ts[0]) if ts is not None else (np.nan, np.nan, np.nan)
        pars = dict(tite=P["tite"][k], ff=P["ff"][k], nf=P["nf"], w=P["w"][k], am=P["am"], an=P["an"][k])
        warm_ok = prev is not None and (t - prev["t"]) <= 25.0
        starts = []
        if warm_ok:
            starts.append(("warm", prev["plasma_psi"], prev))
        R0 = float(np.clip(Rpk_ts, 0.7, 1.1)) if np.isfinite(Rpk_ts) else 0.9
        a0 = float(np.clip(0.5 * (Rout_ts - Rin_ts), 0.3, 0.65)) if np.isfinite(Rin_ts) and np.isfinite(Rout_ts) else 0.45
        if lib is not None:
            starts += [("seed", ps, None) for ps in lib.nearest(inp, 2)]
        starts.append(("cold", sc.gaussian_psi_guess(M.eq, R0, 0.0, a0, 1.8, M.Ip), None))
        res = None
        n_solves = 0
        for name, psi0, ref in starts:
            M.eq.plasma_psi = psi0.copy()
            # initial tables: mapped on the reference solution (warm) or the previous tables / a generic shape
            tabs = None
            if ts is not None and ref is not None:
                x = ref_psiN(M, ref, R_c)
                tabs = kinetic_tables(x, ts[1], ts[0])
            if tabs is None and prev is not None:
                tabs = prev["tables"]
            if tabs is None:
                if ts is not None:
                    ne_pk = float(np.nanpercentile(ts[1][ts[1] > 0], 95)); te_pk = float(np.nanpercentile(ts[0][ts[0] > 0], 95))
                else:
                    ne_pk, te_pk = 2e19, 300.0
                tabs = (ne_pk * (1 - 0.6 * XS ** 2), np.maximum(te_pk * (1 - XS ** 1.5) ** 1.2, 10.0))
            ok = False
            for it in range(3):
                paxis, g, h = profile_tables(tabs[0], tabs[1], **pars)
                prof = M.profile(paxis, g, h)
                n_solves += 1
                try:
                    ok = M.solve(prof, max_it=80 if (name != "warm" and it == 0) else 60)
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok or ts is None:
                    break
                x = M.psiN_at(float(prof.psi_axis), float(prof.psi_bndry), R_c, np.zeros_like(R_c))
                new = kinetic_tables(x, ts[1], ts[0])
                if new is None:
                    break
                change = max(np.max(np.abs(new[0] / new[0][0] - tabs[0] / tabs[0][0])),
                             np.max(np.abs(new[1] / new[1][0] - tabs[1] / tabs[1][0])),
                             abs(new[0][0] * new[1][0] / (tabs[0][0] * tabs[1][0]) - 1))
                tabs = new
                if change < 0.02:
                    break
                if it == 2:                      # last re-solve with the final tables
                    paxis, g, h = profile_tables(tabs[0], tabs[1], **pars)
                    prof = M.profile(paxis, g, h)
                    n_solves += 1
                    try:
                        ok = M.solve(prof)
                    except Exception:  # noqa: BLE001
                        ok = False
            if not ok:
                fr.setdefault("tries", []).append(f"{name}:not_converged")
                continue
            if prof.flag_limiter:
                fr.setdefault("tries", []).append(f"{name}:limited")
                continue
            lab = sc.equilibrium_labels(M.eq, prof, M.fvac)
            why = accept(lab, prev if warm_ok else None)
            if why:
                fr.setdefault("tries", []).append(f"{name}:{why}")
                continue
            res = (prof, lab, tabs, paxis, g, h, name)
            break
        fr.update(sec=time.time() - t0, n_solves=n_solves, has_ts=ts is not None, inputs=inp, **{k_: float(v) for k_, v in pars.items()})
        if res is None:
            tries = fr.get("tries", [])
            fr["status"] = tries[-1].split(":", 1)[1] if tries else "failed"
            if verbose:
                print(f"  t {t:6.1f} FAIL {tries} ns {n_solves} {fr['sec']:.1f}s has_ts {ts is not None}", flush=True)
            continue
        prof, lab, tabs, paxis, g, h, name = res
        C = lab.pop("lcfs")
        x_mid = M.psiN_at(float(prof.psi_axis), float(prof.psi_bndry), M.R1, np.zeros_like(M.R1))
        R_in, R_out = lcfs_midplane(M.R1, x_mid, lab["r_axis"])
        fr.update(status="ok", start=name, psi=np.asarray(M.eq.psi()).T.astype(np.float64), lcfs=C, labels=lab,
                  paxis=paxis, pe_axis=float(E * tabs[0][0] * tabs[1][0]), tables=tabs, L=float(prof.L),
                  beta0=float(prof.Beta0), R_lcfs_in=R_in, R_lcfs_out=R_out, R_ts_in=Rin_ts, R_ts_out=Rout_ts, R_ts_pk=Rpk_ts,
                  psi_axis=float(prof.psi_axis), psi_bndry=float(prof.psi_bndry))
        if lib is not None:
            lib.add(inp, M.eq.plasma_psi)
        prev = dict(t=t, plasma_psi=M.eq.plasma_psi.copy(), psi_axis=float(prof.psi_axis), psi_bndry=float(prof.psi_bndry),
                    z_axis=lab["z_axis"], tables=tabs, psi_full=np.asarray(M.eq.psi()).copy())
        if verbose:
            print(f"  t {t:6.1f} {name:4s} ns {n_solves} {fr['sec']:.1f}s R {lab['r_axis']:.3f} Z {lab['z_axis']:+.3f} "
                  f"li {lab['li']:.2f} bN {lab['beta_n']:.2f} q95 {lab['q95']:.1f} k {lab['elongation']:.2f} "
                  f"Rin {R_in:.3f}/{Rin_ts:.3f} Rout {R_out:.3f}/{Rout_ts:.3f} cp {paxis / fr['pe_axis']:.2f}", flush=True)
    return d, P, sig, frames


def ref_psiN(M, ref, R):
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(M.R1, M.Z1, ref["psi_full"])
    return (np.asarray(spl(R, np.zeros_like(R), grid=False), float) - ref["psi_axis"]) / (ref["psi_bndry"] - ref["psi_axis"])


def lcfs_midplane(R1, x, r_axis):
    """Inner / outer midplane radii where the normalised flux crosses 1 (walking out from the axis)."""
    ia = int(np.argmin(np.abs(R1 - r_axis)))
    out = []
    for step in (-1, 1):
        j = ia
        while 0 <= j + step < len(R1) and x[j + step] < 1:
            j += step
        if not (0 <= j + step < len(R1)):
            out.append(np.nan); continue
        x0, x1 = x[j], x[j + step]
        out.append(float(R1[j] + (1 - x0) / (x1 - x0) * (R1[j + step] - R1[j])) if x1 != x0 else float(R1[j]))
    return out[0], out[1]


def accept(lab, prev):
    if lab is None:
        return "no_lcfs"
    if not (0.45 < lab["r_axis"] < 1.3 and abs(lab["z_axis"]) < 0.15):
        return "axis_out_of_range"
    if prev is not None and abs(lab["z_axis"] - prev["z_axis"]) > 0.03:
        return "z_jump"
    if not (1.0 < lab["volume"] < 20.0):
        return "volume_out_of_range"
    if not (np.isfinite(lab["q95"]) and 1.5 < lab["q95"] < 40):
        return "bad_q95"
    if not (np.isfinite(lab["li"]) and 0.3 < lab["li"] < 3.0):
        return "bad_li"
    if not np.isfinite(lab["beta_n"]):
        return "bad_betan"
    return ""


# ------------------------------------------------------------------------------------------------------------------
# synthetic Thomson + parquet
# ------------------------------------------------------------------------------------------------------------------
def synth_thomson(M, d, P, frames):
    """Replace every Thomson slice: synthetic where a solved frame is within SYN_MAX_DT, else all channels NaN."""
    from scipy.interpolate import RectBivariateSpline
    ok = [k for k, f in enumerate(frames) if f["status"] == "ok"]
    t_ok = np.array([frames[k]["t"] for k in ok])
    rng = np.random.default_rng(P["noise_seed"])
    out = {}
    spl_cache = {}
    for side, R in (("core", d["core_R"]), ("edge", d["edge_R"])):
        tt, TE, NE = d[f"{side}_t"], d[f"{side}_Te"], d[f"{side}_ne"]
        off = 1 if side == "core" else 0
        te_new = np.full_like(TE, np.nan)
        ne_new = np.full_like(NE, np.nan)
        for j in range(len(tt)):
            if not len(t_ok):
                break
            i = int(np.argmin(np.abs(t_ok - tt[j])))
            if abs(t_ok[i] - tt[j]) > SYN_MAX_DT:
                continue
            f = frames[ok[i]]
            if ok[i] not in spl_cache:
                spl_cache[ok[i]] = RectBivariateSpline(M.Z1, M.R1, f["psi"])     # stored psi is (Z, R)
            spl = spl_cache[ok[i]]
            xc = (spl(np.zeros_like(R), R, grid=False) - f["psi_axis"]) / (f["psi_bndry"] - f["psi_axis"])
            # this slice's own measured core profile on the solved surfaces (core slice at the same time for edge)
            jc = int(np.argmin(np.abs(d["core_t"] - tt[j])))
            te_c, ne_c = core_slice(d, jc)
            tabs = None
            if abs(d["core_t"][jc] - tt[j]) < 1.0 and np.sum(np.isfinite(te_c) & (te_c > 0)) >= MIN_CH:
                xcore = (spl(np.zeros_like(d["core_R"]), d["core_R"], grid=False) - f["psi_axis"]) / (f["psi_bndry"] - f["psi_axis"])
                tabs = kinetic_tables(xcore, ne_c, te_c)
            if tabs is None:
                tabs = f["tables"]
            te_r, ne_r = TE[j, off:], NE[j, off:]
            real_nan = ~(np.isfinite(te_r) & np.isfinite(ne_r))
            if real_nan.all():
                continue
            sig_te = slice_noise(R, te_r) if side == "core" else 0.08
            sig_ne = slice_noise(R, ne_r) if side == "core" else 0.08
            ne_s, te_s = synth_slice(xc, tabs[0], tabs[1], P["lam_ne"], P["lam_te"], sig_ne, sig_te, real_nan, rng)
            te_new[j, off:] = te_s
            ne_new[j, off:] = ne_s
            if off:
                te_new[j, 0] = TE[j, 0]
                ne_new[j, 0] = NE[j, 0]
        out[side] = (te_new, ne_new)
    return out


LABEL_KEYS = [("efit_beta_n", "beta_n"), ("efit_li", "li"), ("efit_q95", "q95"), ("efit_r_axis", "r_axis"),
              ("efit_z_axis", "z_axis"), ("efit_beta_p", "beta_p"), ("efit_elongation", "elongation"),
              ("efit_psi_axis", "psi_axis"), ("efit_psi_boundary", "psi_boundary"), ("efit_geom_axis_r", "geom_axis_r"),
              ("efit_geom_axis_z", "geom_axis_z")]
FRAME_DIAG = ["t", "status", "start", "sec", "n_solves", "has_ts", "tite", "ff", "nf", "w", "am", "an", "paxis", "pe_axis", "L",
              "beta0", "R_lcfs_in", "R_lcfs_out", "R_ts_in", "R_ts_out", "R_ts_pk"]


def write_parquet(path, d, P, sig, frames, thomson, fname, r):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tab = d["tab"]
    ok = [f for f in frames if f["status"] == "ok"]

    def put(name, arr):
        nonlocal tab
        i = tab.schema.get_field_index(name)
        if i >= 0:
            tab = tab.set_column(i, pa.field(name, arr.type), arr)
        else:
            tab = tab.append_column(pa.field(name, arr.type), arr)

    for c in SIG:
        put(c, pa.array([sig[c].tolist()], pa.list_(pa.float64())))
    for side in ("core", "edge"):
        te, ne = thomson[side]
        put(f"thomson_{side}_Te", pa.array([te.tolist()], pa.list_(pa.list_(pa.float64()))))
        put(f"thomson_{side}_ne", pa.array([ne.tolist()], pa.list_(pa.list_(pa.float64()))))
    put("efit_times", pa.array([[f["t"] for f in ok]], pa.list_(pa.float32())))
    put("efit_psirz", pa.array([[f["psi"].tolist() for f in ok]], pa.list_(pa.list_(pa.list_(pa.float64())))))
    for col, key in LABEL_KEYS:
        put(col, pa.array([[float(f["labels"][key]) for f in ok]], pa.list_(pa.float64())))
    put("efit_lcfs_n", pa.array([[float(len(f["lcfs"])) for f in ok]], pa.list_(pa.float64())))
    put("efit_lcfs_r", pa.array([[f["lcfs"][:, 0].tolist() for f in ok]], pa.list_(pa.list_(pa.float64()))))
    put("efit_lcfs_z", pa.array([[f["lcfs"][:, 1].tolist() for f in ok]], pa.list_(pa.list_(pa.float64()))))
    put("source", pa.array(["MAST_synthetic_freegsnke_gen4"], pa.string()))
    meta = dict(version=VERSION, origin_file=fname, realisation=r, split=split_of(fname), sol_factor=SOL_FACTOR,
                jitter=P["jit"], nf=P["nf"], am=P["am"], lam_ne=P["lam_ne"], lam_te=P["lam_te"])
    put("synth_meta", pa.array([json.dumps(meta)], pa.string()))
    diag = [{k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item()) for k, v in f.items() if k in FRAME_DIAG}
            for f in frames]
    for f, g in zip(frames, diag):
        if f["status"] == "ok":
            g.update(volume=f["labels"]["volume"], minor_radius=f["labels"]["minor_radius"], beta_t=f["labels"]["beta_t"])
        if "tries" in f:
            g["tries"] = f["tries"]
    put("synth_frames", pa.array([json.dumps(diag)], pa.string()))
    tmp = path + ".tmp"
    pq.write_table(tab, tmp, compression="zstd")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------------
# task queue with claims (shared file system; several nodes)
# ------------------------------------------------------------------------------------------------------------------
def all_files():
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(DATA, "*.parquet")))


def task_list(n_train_real, n_eval_real):
    files = all_files()
    tasks = []
    for r in range(max(n_train_real, n_eval_real)):
        perm = np.random.default_rng([SEED, 99, r]).permutation(len(files))
        for i in perm:
            f = files[i]
            sp = split_of(f)
            if r < (n_train_real if sp == "train" else n_eval_real):
                tasks.append((r, f))
    return tasks


def out_path(r, fname):
    sp = split_of(fname)
    stem = fname.replace(".parquet", "").replace("mast_shot_", "")
    return os.path.join(OUT, f"mast_g4_{sp}", f"mast_shot_g4r{r}_{stem}.parquet")


def task_id(r, fname):
    return f"r{r}_{fname.replace('.parquet', '')}"


def hang_frames_of(tid):
    p = os.path.join(OUT, "hangs", tid + ".txt")
    if not os.path.exists(p):
        return set()
    return {int(x) for x in open(p).read().split()}


def worker(k, nw, node_tag, n_train_real, n_eval_real):
    from pool import beat as pool_beat
    hb = os.path.join(OUT, "hb", node_tag)
    claims = os.path.join(OUT, "claims")
    os.makedirs(claims, exist_ok=True)
    me = f"{node_tag}:{k}"
    M = Machine()
    lib = SeedLib()
    tasks = task_list(n_train_real, n_eval_real)
    # resume a task this worker slot held before a watchdog restart
    mine = [t for t in tasks if os.path.exists(os.path.join(claims, task_id(*t))) and
            open(os.path.join(claims, task_id(*t))).read().strip() == me and not done(*t)]
    t_start = time.time(); n_sh = n_fr = n_ok = 0
    for r, fname in mine + tasks:
        tid = task_id(r, fname)
        cp = os.path.join(claims, tid)
        if done(r, fname):
            continue
        if not (os.path.exists(cp) and open(cp).read().strip() == me):
            try:
                fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, me.encode()); os.close(fd)
        t0 = time.time()
        try:
            d, P, sig, frames = solve_shot(M, fname, r, hang_frames_of(tid), beat=lambda fi: pool_beat(hb, k, f"{tid}|{fi}"), lib=lib)
            pool_beat(hb, k, f"{tid}|write")
            th = synth_thomson(M, d, P, frames)
            nok = sum(f["status"] == "ok" for f in frames)
            os.makedirs(os.path.dirname(out_path(r, fname)), exist_ok=True)
            if nok:
                write_parquet(out_path(r, fname), d, P, sig, frames, th, fname, r)
            st = {}
            for f in frames:
                st[f["status"]] = st.get(f["status"], 0) + 1
            secs = [f["sec"] for f in frames if "sec" in f]
            with open(os.path.join(OUT, "done", tid + ".json"), "w") as fh:
                json.dump(dict(r=r, file=fname, split=split_of(fname), n=len(frames), ok=nok, status=st,
                               sec=float(np.sum(secs)), node=me, wall=time.time() - t0), fh)
            n_sh += 1; n_fr += len(frames); n_ok += nok
            print(f"[{me}] {tid} {nok}/{len(frames)} ok {time.time() - t0:.0f}s | total {n_sh} shots {n_ok}/{n_fr} frames "
                  f"{(time.time() - t_start) / max(n_fr, 1):.2f} s/frame", flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            with open(os.path.join(OUT, "done", tid + ".json"), "w") as fh:
                json.dump(dict(r=r, file=fname, split=split_of(fname), error=traceback.format_exc()[-400:], node=me), fh)
            M = Machine()
    print(f"[{me}] finished: {n_sh} shots, {n_ok}/{n_fr} frames ok in {(time.time() - t_start) / 3600:.2f} h", flush=True)


def done(r, fname):
    return os.path.exists(os.path.join(OUT, "done", task_id(r, fname) + ".json"))


def run(a):
    from pool import run_pool
    for sub in ("done", "hangs", "claims", os.path.join("hb", a.node_tag)):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    def on_hang(k, tag):
        tid, fi = tag.split("|")
        if fi == "write":
            return
        with open(os.path.join(OUT, "hangs", tid + ".txt"), "a") as fh:
            fh.write(f"{fi}\n")

    n = run_pool(worker, a.n_workers, (a.node_tag, a.n_train_real, a.n_eval_real), os.path.join(OUT, "hb", a.node_tag),
                 stall=a.stall, on_hang=on_hang)
    print(f"pool finished, {n} hang restarts", flush=True)


def summary():
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "done", "*.json"))]
    agg = {}
    for x in recs:
        key = (x["split"], x["r"])
        a = agg.setdefault(key, dict(shots=0, frames=0, ok=0, sec=0.0, errors=0, status={}))
        a["shots"] += 1
        if "error" in x:
            a["errors"] += 1
            continue
        a["frames"] += x["n"]; a["ok"] += x["ok"]; a["sec"] += x["sec"]
        for s, v in x["status"].items():
            a["status"][s] = a["status"].get(s, 0) + v
    for key in sorted(agg):
        a = agg[key]
        print(f"{key[0]:5s} r{key[1]}: {a['shots']} shots, {a['ok']}/{a['frames']} frames ok "
              f"({a['ok'] / max(a['frames'], 1):.2f}), {a['sec'] / max(a['frames'], 1):.2f} s/frame, errors {a['errors']}, "
              f"{json.dumps(dict(sorted(a['status'].items(), key=lambda kv: -kv[1])))}")
    tot = sum(a["ok"] for a in agg.values())
    print(f"total accepted frames: {tot}")


def test(n, r, seed=7):
    files = [f for f in all_files() if split_of(f) == "train"]
    pick = [files[i] for i in np.random.default_rng(seed).choice(len(files), n, replace=False)]
    global OUT
    OUT = os.path.join(OUT, "test")
    M = Machine()
    lib = SeedLib()
    for fname in pick:
        t0 = time.time()
        print(f"== {fname} r{r}", flush=True)
        d, P, sig, frames = solve_shot(M, fname, r, verbose=True, lib=lib)
        th = synth_thomson(M, d, P, frames)
        st = {}
        for f in frames:
            st[f["status"]] = st.get(f["status"], 0) + 1
        print(f"   status {st}; {time.time() - t0:.0f} s for {len(frames)} frames; draws tite0 {P['tite'][0]:.2f} ff0 {P['ff'][0]:.2f} "
              f"w0 {P['w'][0]:.2f} am {P['am']:.2f} an0 {P['an'][0]:.2f}", flush=True)
        os.makedirs(os.path.join(OUT, "mast_g4_test"), exist_ok=True)
        if any(f["status"] == "ok" for f in frames):
            p = os.path.join(OUT, "mast_g4_test", os.path.basename(out_path(r, fname)))
            write_parquet(p, d, P, sig, frames, th, fname, r)
            print(f"   wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--real", type=int, default=1, help="realisation for --test")
    ap.add_argument("--seed", type=int, default=7, help="shot choice for --test")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--build-seedlib", action="store_true")
    ap.add_argument("--node-tag", default=socket.gethostname())
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--n-train-real", type=int, default=8)
    ap.add_argument("--n-eval-real", type=int, default=2)
    ap.add_argument("--stall", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = a.out
    if a.build_seedlib:
        build_seedlib()
    elif a.summary:
        summary()
    elif a.test:
        test(a.test, a.real, a.seed)
    elif a.run:
        run(a)


if __name__ == "__main__":
    main()
