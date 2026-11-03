"""Machine-independent input features z(c, y) shared by DIII-D (teacher) and MAST (generator).

All features are computed from:
  - a Thomson profile on a normalised-radius grid XQ (x = 0 axis / profile peak, 1 = plasma edge), built per machine
    (MAST: midplane channels, x from the Te peak and the 5%-of-peak edges, as gen_param.ts_table;
     DIII-D: core vertical chord + tangential channels mapped onto the outboard midplane radius of their flux surface),
  - Ip (kA), its normalised ramp rate, the shot phase, the vacuum toroidal field, and the minor radius a and centre R
    of the plasma (MAST: from the Thomson edges; DIII-D: from the EFIT midplane edges, which the MAST Thomson edges
    stand in for).
No feature uses an equilibrium label of MAST.
"""
import numpy as np

MU0 = 4e-7 * np.pi
E = 1.602176634e-19
XQ = np.linspace(0.0, 1.0, 21)
KAPPA0 = 1.8          # nominal elongation for the poloidal-field scale (same constant on both machines)

FEATS = ["ip_frac", "ph_up", "ph_down", "rr", "rr_tau", "qstar", "te0", "tav", "fgw", "betape",
         "pk_te", "pk_ne", "pk_pe", "x50_te", "x50_ne", "x50_pe", "te_out"]


def profile_on_xq(x, te, ne):
    """Median of channels in x-bins of width 0.05 (x <= 1.05), interpolated on XQ; flat outside the covered range.
    Returns (te_q, ne_q, xmin) or None."""
    x, te, ne = (np.asarray(v, np.float64) for v in (x, te, ne))
    ok = np.isfinite(x) & np.isfinite(te) & np.isfinite(ne) & (x <= 1.05) & (x >= 0)
    if ok.sum() < 8:
        return None
    b = np.clip(np.round(x[ok] / 0.05).astype(int), 0, 20)
    T = np.array([np.median(te[ok][b == k]) if (b == k).any() else np.nan for k in range(21)])
    N = np.array([np.median(ne[ok][b == k]) if (b == k).any() else np.nan for k in range(21)])
    f = np.isfinite(T)
    if f.sum() < 6 or XQ[f].min() > 0.35 or XQ[f].max() < 0.7:
        return None
    return np.interp(XQ, XQ[f], T[f]), np.interp(XQ, XQ[f], N[f]), float(XQ[f].min())


def _x50(q):
    q0 = q[:3].mean()
    if not q0 > 0:
        return np.nan
    below = np.nonzero(q < 0.5 * q0)[0]
    below = below[below > 0]
    if not len(below):
        return 1.0
    k = below[0]
    return float(np.interp(0.5 * q0, [q[k], q[k - 1]], [XQ[k], XQ[k - 1]]))


def features(te_q, ne_q, ip_kA, ip_frac, phase, rr, B0, a, R0):
    """te_q in keV, ne_q in 1e19 m^-3 on XQ; ip in kA; rr = (dIp/dt)/Ip in 1/s; B0 vacuum field (T) at R0; a, R0 (m).
    Returns a dict of FEATS."""
    w = 2 * XQ                                 # cylindrical volume weight on x in [0, 1]
    av = lambda q: float(np.trapezoid(q * w, XQ))  # noqa: E731
    pe = te_q * ne_q * 1e3 * 1e19 * E           # Pa
    te0, ne0, pe0 = te_q[:3].mean(), ne_q[:3].mean(), pe[:3].mean()
    tav, nav, pav = av(te_q), av(ne_q), av(pe)
    Ip = abs(ip_kA) * 1e3
    Bp = MU0 * Ip / (2 * np.pi * a * np.sqrt((1 + KAPPA0 ** 2) / 2))
    eta = 1.65e-9 * 17.0 * 2.0 / max(tav, 0.02) ** 1.5      # Spitzer, ln Lambda 17, Z_eff 2 (ohm m)
    tau = MU0 * a ** 2 / eta                                 # current diffusion time (s)
    return dict(ip_frac=float(ip_frac), ph_up=float(phase == 0), ph_down=float(phase == 2), rr=float(np.clip(rr, -20, 20)),
                rr_tau=float(np.clip(rr * tau, -20, 20)),
                qstar=float(5 * a ** 2 * B0 * (1 + KAPPA0 ** 2) / 2 / (R0 * Ip / 1e6)),
                te0=float(te0), tav=float(tav), fgw=float(nav / 10 / (Ip / 1e6 / (np.pi * a ** 2))),
                betape=float(2 * MU0 * pav / Bp ** 2),
                pk_te=float(te0 / max(tav, 1e-6)), pk_ne=float(ne0 / max(nav, 1e-6)), pk_pe=float(pe0 / max(pav, 1e-9)),
                x50_te=_x50(te_q), x50_ne=_x50(ne_q), x50_pe=_x50(pe), te_out=float(te_q[16] / max(te0, 1e-6)))


# ---------------------------------------------------------------- Ip-history features (li memory), optional
HFEATS = ["h01", "h025", "h05", "tsince", "ew"]
FEATS_ALL = FEATS + HFEATS


def tau_cd(tav, a):
    """current diffusion time (s), as in features()"""
    eta = 1.65e-9 * 17.0 * 2.0 / max(tav, 0.02) ** 1.5
    return MU0 * a ** 2 / eta


def hist_features(t, tr, ip_tr, tau_s):
    """t (ms) frame time; tr (ms), ip_tr (|Ip|, any unit) the Ip trace; tau_s current diffusion time (s).
    Ip(t - k tau)/Ip(t) for k = 0.1, 0.25, 0.5; time since |Ip| first exceeded 100 kA-equivalent (10% of max) over tau;
    exponentially weighted past Ip over Ip(t) (window 3 tau)."""
    tr, ip_tr = np.asarray(tr, np.float64), np.abs(np.asarray(ip_tr, np.float64))
    ok = np.isfinite(tr) & np.isfinite(ip_tr)
    tr, ip_tr = tr[ok], ip_tr[ok]
    out = {k: np.nan for k in HFEATS}
    if len(tr) < 5 or not np.isfinite(tau_s):
        return out
    o = np.argsort(tr); tr, ip_tr = tr[o], ip_tr[o]
    tau = float(np.clip(tau_s, 0.005, 10.0)) * 1e3
    f = lambda q: np.interp(q, tr, ip_tr, left=0.0)  # noqa: E731
    ipt = float(f(t))
    if not ipt > 0:
        return out
    for k, nm in ((0.1, "h01"), (0.25, "h025"), (0.5, "h05")):
        out[nm] = float(np.clip(f(t - k * tau) / ipt, 0, 3))
    on = np.nonzero(ip_tr > 0.1 * ip_tr.max())[0]
    t_on = tr[on[0]] if len(on) else tr[0]
    out["tsince"] = float(np.clip((t - t_on) / tau, 0, 20))
    s = np.linspace(0, 3 * tau, 61)
    out["ew"] = float(np.clip(np.trapezoid(f(t - s) * np.exp(-s / tau), s) / tau / ipt, 0, 3))
    return out
