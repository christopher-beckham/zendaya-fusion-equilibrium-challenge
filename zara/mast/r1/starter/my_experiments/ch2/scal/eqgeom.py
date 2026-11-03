"""Per-frame equilibrium geometry and physics scalars from one 65x65 flux map (machine-agnostic, numpy only).

    frame_physics(phi, R, Z, mask, F_vac, ip, ts=None) -> dict

phi    flux map with the magnetic axis at a MAXIMUM (= axis_sign * psi_stored), per-radian units (Wb/rad).
mask   the machine's coarse envelope mask (scorer convention); used for the O-point search and the LCFS bisection.
F_vac  vacuum R B_T = mu0 N_tf I_tf / (2 pi) [T m];  ip  plasma current [A].
ts     optional Thomson: dict(R, Z, te [keV], ne [1e19 m^-3]) at the frame (1-D arrays, nan = missing).

Everything is computed twice where it matters: on the scorer's coarse grid (LCFS bisection exactly as the scorer does it)
and on a 3x cubic-spline refinement (smooth contours and gradients for q and the volume integrals).

q95 estimators (all with F = F_vac; the edge diamagnetic correction is below 0.3 % on every set checked):
  q95_int    F/(2 pi) loop-integral dl / (R |grad psi|) on psi_N = 0.95 (fine grid, spline gradient)
  q95_dA     F/(2 pi) d(Phi_tor/F)/d psi from the 1/R-weighted areas inside psi_N = 0.93 and 0.97 (positions only)
  q95_amp    q95_int x (loop integral of B_p on the LCFS) / (mu0 Ip)   (Ampere normalisation of the gradient scale)
  q95_uck    Uckan / ITER: 5 a^2 B0 / (R Ip[MA]) (1 + k^2 (1 + 2 d^2 - 1.2 d^3)) / 2 x (1.17 - 0.65 e) / (1 - e^2)^2
  qstar      5 a^2 B0 (1 + k^2) / (2 R Ip[MA])
Kinetic (Thomson): electron pressure p_e = n_e T_e mapped onto psi_N, binned in rho = sqrt(psi_N), volume averaged.
  betaN_e = 100 * 2 mu0 <p_e>_V a / (B0 Ip[MA]),  B0 = F_vac / R_geo  (the synthetic-set / EFIT definition with B at R_geo)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.interpolate import RectBivariateSpline
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parents[3]   # the starter folder of this package
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from o_point import find_o_point                                    # noqa: E402
from contour import is_closed, encloses_point, grid_to_physical, resample_closed   # noqa: E402
from derive import _shape_from_contour, internal_inductance         # noqa: E402

MU0 = 4e-7 * math.pi
QE = 1.602176634e-19
PE_UNIT = 1e3 * QE * 1e19          # Te[keV] * ne[1e19] -> Pa
UP = 3                              # refinement factor of the spline grid
RHO_EDGES = np.linspace(0.0, 1.0, 11)


# ------------------------------------------------------------------------------------------------ contours
def _in_mask(c, mask_f):
    nr, nc = mask_f.shape
    rows = np.clip(c[:, 0], 0, nr - 1); cols = np.clip(c[:, 1], 0, nc - 1)
    r0 = np.floor(rows).astype(int); c0 = np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, nr - 1); c1 = np.minimum(c0 + 1, nc - 1)
    fr = rows - r0; fc = cols - c0
    v = (mask_f[r0, c0] * (1 - fr) * (1 - fc) + mask_f[r1, c0] * fr * (1 - fc) + mask_f[r0, c1] * (1 - fr) * fc
         + mask_f[r1, c1] * fr * fc)
    return bool(np.all(v > 0.5))


def _closed_at(phi, level, iz, ir, mask_f=None):
    for c in find_contours(phi, level):
        if len(c) < 4 or not is_closed(c) or not encloses_point(c, iz, ir):
            continue
        if mask_f is not None and not _in_mask(c, mask_f):
            continue
        return c
    return None


def lcfs_bisect(phi, iz, ir, mask_f, n_iter=24, bracket=None):
    """Scorer LCFS logic (lowest closed, axis-enclosing, in-envelope level) returning (contour_rc, level).
    bracket=(lo, hi): optional narrower start (used if hi is valid and lo is not)."""
    if bracket is not None:
        lo, hi = bracket
        best = _closed_at(phi, hi, iz, ir, mask_f)
        if best is not None and _closed_at(phi, lo, iz, ir, mask_f) is None:
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                c = _closed_at(phi, mid, iz, ir, mask_f)
                if c is not None:
                    hi, best = mid, c
                else:
                    lo = mid
            return best, hi
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo = float(np.min(border)); hi = float(phi[iz, ir]) - 1e-9
    if not hi > lo:
        return None, np.nan
    best = _closed_at(phi, hi, iz, ir, mask_f)
    if best is None:
        return None, np.nan
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = _closed_at(phi, mid, iz, ir, mask_f)
        if c is not None:
            hi, best = mid, c
        else:
            lo = mid
    return best, hi


def _poly_metrics(xy):
    """Closed polygon (N, 2) physical -> dict(perimeter, area, A_R = integral dA / R, R-moment volume)."""
    R, Z = xy[:, 0], xy[:, 1]
    R2, Z2 = np.roll(R, -1), np.roll(Z, -1)
    seg = np.hypot(R2 - R, Z2 - Z)
    cross = R * Z2 - R2 * Z
    area = 0.5 * cross.sum()
    sgn = 1.0 if area >= 0 else -1.0
    # integral over the enclosed area of 1/R dR dZ = loop integral ln(R) dZ (Green, counter-clockwise)
    lnR = np.log(np.clip(0.5 * (R + R2), 1e-6, None))
    A_R = sgn * float(np.sum(lnR * (Z2 - Z)))
    return float(seg.sum()), abs(float(area)), A_R


def midplane_crossings(C, z0=0.0):
    """Innermost / outermost R where the closed polygon C crosses Z = z0 (nan if it does not)."""
    R, Z = C[:, 0], C[:, 1]
    R2, Z2 = np.roll(R, -1), np.roll(Z, -1)
    s = (Z - z0) * (Z2 - z0) <= 0
    s &= Z != Z2
    if not s.any():
        return np.nan, np.nan
    t = (z0 - Z[s]) / (Z2[s] - Z[s])
    Rx = R[s] + t * (R2[s] - R[s])
    return float(Rx.min()), float(Rx.max())


def ts_edges(tR, tZ, te, thr):
    """Inner / outer radius where the midplane (|Z| < 5 cm) Thomson Te crosses thr [keV] (outermost crossings of the
    hot region around the Te maximum; linear interpolation; nan if not bracketed)."""
    m = np.abs(tZ) < 0.05
    if m.sum() < 10:
        return np.nan, np.nan
    r, t = tR[m], te[m]
    o = np.argsort(r); r, t = r[o], t[o]
    k = int(np.argmax(t))
    if t[k] < 2 * thr:
        return np.nan, np.nan
    ri = ro = np.nan
    below = np.where(t[:k] < thr)[0]
    if len(below):
        j = below[-1]
        ri = r[j] + (thr - t[j]) * (r[j + 1] - r[j]) / (t[j + 1] - t[j])
    below = np.where(t[k:] < thr)[0]
    if len(below):
        j = k + below[0]
        ro = r[j - 1] + (thr - t[j - 1]) * (r[j] - r[j - 1]) / (t[j] - t[j - 1])
    return float(ri), float(ro)


# ------------------------------------------------------------------------------------------------ main
class Grid:
    """Static per-machine grid objects (coarse + fine)."""

    def __init__(self, R, Z, mask):
        self.R = np.asarray(R, np.float64); self.Z = np.asarray(Z, np.float64)
        self.mask = np.asarray(mask, bool); self.mask_f = self.mask.astype(np.float64)
        nR, nZ = len(self.R), len(self.Z)
        self.Rf = np.linspace(self.R[0], self.R[-1], (nR - 1) * UP + 1)
        self.Zf = np.linspace(self.Z[0], self.Z[-1], (nZ - 1) * UP + 1)
        self.RRf, self.ZZf = np.meshgrid(self.Rf, self.Zf)
        # envelope on the fine grid (bilinear of the float mask)
        sp = RectBivariateSpline(self.Z, self.R, self.mask_f, kx=1, ky=1)
        self.mask_ff = np.clip(sp(self.Zf, self.Rf), 0, 1)
        self.dRf = self.Rf[1] - self.Rf[0]; self.dZf = self.Zf[1] - self.Zf[0]
        self.dVf = 2 * math.pi * self.RRf * self.dRf * self.dZf
        self.pts_f = np.c_[self.RRf.ravel(), self.ZZf.ravel()]


def _to_phys_f(c, g):
    return np.column_stack([np.interp(c[:, 1], np.arange(len(g.Rf)), g.Rf), np.interp(c[:, 0], np.arange(len(g.Zf)), g.Zf)])


def frame_physics(phi, g: Grid, F_vac, ip, ts=None, levels_q=(0.9, 0.95), with_li=True, limiter=None):
    """limiter=(R_cc, Z_half): centre-column limiter (machine geometry). If the predicted separatrix surface reaches inside
    R_cc for |Z| <= Z_half, the boundary flux becomes the limiter value (EFIT's limited-boundary rule) for q95 and the
    volume integrals; the scorer-style shape (kappa, a, ...) is unchanged."""
    out = {}
    phi = np.asarray(phi, np.float64)
    if not np.isfinite(phi).all():
        phi = np.nan_to_num(phi, nan=np.nanmin(phi), posinf=np.nanmax(phi), neginf=np.nanmin(phi))
    Fv = abs(float(F_vac)); Ip = abs(float(ip))
    out["F_vac"] = Fv; out["ip"] = Ip
    Ra, Za, iz, ir, _ = find_o_point(phi, g.R, g.Z, g.mask if g.mask.any() else None)
    out["R_axis"], out["Z_axis"] = Ra, Za
    # coarse LCFS (scorer)
    c_rc, lev_c = lcfs_bisect(phi, iz, ir, g.mask_f)
    if c_rc is None:
        out["ok"] = 0
        return out
    C = resample_closed(grid_to_physical(c_rc, g.R, g.Z), 512)
    out.update(_shape_from_contour(C))
    Rmax, Rmin = C[:, 0].max(), C[:, 0].min(); Zmax, Zmin = C[:, 1].max(), C[:, 1].min()
    a = 0.5 * (Rmax - Rmin); Rg = 0.5 * (Rmax + Rmin)
    out.update(a=a, R_geo=Rg, Z_geo=0.5 * (Zmax + Zmin), eps=a / Rg)
    lp, area, _ = _poly_metrics(C)
    out["lp"], out["area"] = lp, area
    out["Z_low"], out["Z_up"] = Zmin, Zmax
    out["R_mid_in"], out["R_mid_out"] = midplane_crossings(C, 0.0)
    out["R_low"], out["R_up"] = float(C[np.argmin(C[:, 1]), 0]), float(C[np.argmax(C[:, 1]), 0])
    if with_li:
        try:
            out["li"] = internal_inductance(phi, g.R, g.Z, "MAST", C)
        except Exception:
            out["li"] = np.nan
    # fine spline grid
    S = RectBivariateSpline(g.Z, g.R, phi, kx=3, ky=3)
    phf = S(g.Zf, g.Rf)
    izf, irf = iz * UP, ir * UP
    # refine the axis value: maximum of the spline in a small window
    w = phf[max(izf - UP, 0):izf + UP + 1, max(irf - UP, 0):irf + UP + 1]
    k = np.unravel_index(np.argmax(w), w.shape)
    izf, irf = max(izf - UP, 0) + k[0], max(irf - UP, 0) + k[1]
    pa = float(phf[izf, irf])
    d0 = pa - lev_c
    cf_rc, lev_f = lcfs_bisect(phf, izf, irf, g.mask_ff, n_iter=12, bracket=(lev_c - 0.04 * d0, lev_c + 0.04 * d0))
    if cf_rc is None:
        cf_rc, lev_f = None, lev_c
    pb = lev_f
    out["limited"] = 0
    if limiter is not None:
        zz = np.linspace(-limiter[1], limiter[1], 161)
        plim = float(np.max(S.ev(zz, np.full_like(zz, limiter[0]))))
        if plim > pb and plim < pa:
            c_lim = _closed_at(phf, plim + 1e-12 * abs(pa), izf, irf, None)
            if c_lim is not None:
                out["limited"] = 1; out["psi_b_sep"] = pb
                pb, cf_rc = plim, c_lim
    out["psi_axis"], out["psi_b"], out["psi_b_coarse"], out["dpsi"] = pa, pb, lev_c, pa - pb
    out["ok"] = 1
    dps = pa - pb
    if not dps > 0:
        out["ok"] = 0
        return out
    B0 = Fv / Rg
    out["B0"] = B0
    IpMA = Ip / 1e6
    # Ampere on the LCFS (fine contour if available)
    Cf = _to_phys_f(cf_rc, g) if cf_rc is not None else C

    def gradmag(xy):
        gz = S.ev(xy[:, 1], xy[:, 0], dx=1); gr = S.ev(xy[:, 1], xy[:, 0], dy=1)
        return np.sqrt(gz ** 2 + gr ** 2)

    def loop(xy, f):
        mid = 0.5 * (xy + np.roll(xy, -1, 0))
        dl = np.hypot(*(np.roll(xy, -1, 0) - xy).T)
        return float(np.sum(f(mid) * dl))

    G_b = loop(Cf, lambda m: gradmag(m) / m[:, 0])
    out["mu0Ip_map"] = G_b
    out["amp_ratio"] = G_b / (MU0 * Ip) if Ip > 0 else np.nan
    # q on surfaces
    A_R = {}
    for x in sorted(set(levels_q) | {0.93, 0.97}):
        c = _closed_at(phf, pa - x * dps, izf, irf, None)
        if c is None:
            continue
        xy = _to_phys_f(c, g)
        if x in levels_q:
            gm = np.maximum(gradmag(0.5 * (xy + np.roll(xy, -1, 0))), 1e-12)
            mid = 0.5 * (xy + np.roll(xy, -1, 0)); dl = np.hypot(*(np.roll(xy, -1, 0) - xy).T)
            out[f"q{int(round(100 * x))}_int"] = Fv / (2 * math.pi) * float(np.sum(dl / (mid[:, 0] * gm)))
            out[f"G{int(round(100 * x))}"] = float(np.sum(gm / mid[:, 0] * dl))
        A_R[x] = _poly_metrics(xy)[2]
    if 0.93 in A_R and 0.97 in A_R:
        out["q95_dA"] = Fv / (2 * math.pi) * (A_R[0.97] - A_R[0.93]) / (0.04 * dps)
    # q95 with the separatrix boundary as well (limited frames): q95_int_sep, and the geometric mean of both (q95_half)
    out["q95_int_sep"] = out.get("q95_int", np.nan)
    if out.get("limited", 0) == 1 and "q95_int" in out:
        pbs = out["psi_b_sep"]
        c = _closed_at(phf, pa - 0.95 * (pa - pbs), izf, irf, None)
        if c is not None:
            xy = _to_phys_f(c, g)
            mid = 0.5 * (xy + np.roll(xy, -1, 0)); dl = np.hypot(*(np.roll(xy, -1, 0) - xy).T)
            gm = np.maximum(gradmag(mid), 1e-12)
            out["q95_int_sep"] = Fv / (2 * math.pi) * float(np.sum(dl / (mid[:, 0] * gm)))
    if "q95_int" in out:
        out["q95_half"] = math.sqrt(out["q95_int"] * out["q95_int_sep"]) if out["q95_int_sep"] > 0 else out["q95_int"]
    if "q95_int" in out:
        out["q95_amp"] = out["q95_int"] * out["amp_ratio"]
    # shape formulas
    kap, dl_ = out.get("kappa", np.nan), 0.5 * (out.get("tri_top", np.nan) + out.get("tri_bot", np.nan))
    e = a / Rg
    out["qstar"] = 5 * a ** 2 * B0 * (1 + kap ** 2) / (2 * Rg * IpMA)
    fe = (1.17 - 0.65 * e) / max(1 - e ** 2, 0.05) ** 2
    out["q95_uck"] = 5 * a ** 2 * B0 / (Rg * IpMA) * (1 + kap ** 2 * (1 + 2 * dl_ ** 2 - 1.2 * dl_ ** 3)) / 2 * fe
    out["q_cyl_lp"] = Fv * lp ** 2 / (2 * math.pi * Rg ** 2 * MU0 * Ip) if Ip > 0 else np.nan   # cylinder with a_eff = lp / 2 pi
    # plasma volume on the fine grid
    psin_f = (pa - phf) / dps
    poly = Cf if cf_rc is not None else C
    bb = (g.RRf >= poly[:, 0].min() - 1e-9) & (g.RRf <= poly[:, 0].max() + 1e-9) & (g.ZZf >= poly[:, 1].min() - 1e-9) & (g.ZZf <= poly[:, 1].max() + 1e-9)
    inside = np.zeros(phf.shape, bool)
    inside[bb] = MPath(poly).contains_points(np.c_[g.RRf[bb], g.ZZf[bb]])
    inside &= (psin_f >= 0) & (psin_f <= 1.0)
    V = float(g.dVf[inside].sum())
    out["V_fine"] = V
    out["beta_pol_den"] = (MU0 * Ip / lp) ** 2 / (2 * MU0) if Ip > 0 else np.nan     # B_pa^2 / 2 mu0
    if ts is None or V <= 0:
        return out
    # ---------------------------------------------------------------- kinetic (Thomson)
    tR, tZ = np.asarray(ts["R"], float), np.asarray(ts["Z"], float)
    te, ne = np.asarray(ts["te"], float), np.asarray(ts["ne"], float)
    okp = np.isfinite(tR) & np.isfinite(tZ) & np.isfinite(te) & np.isfinite(ne) & (te > 0.003) & (ne > 0.01)
    okp &= (tR > g.R[0]) & (tR < g.R[-1]) & (tZ > g.Z[0]) & (tZ < g.Z[-1])
    out["n_ts"] = int(okp.sum())
    if okp.sum() < 5:
        return out
    tR, tZ, te, ne = tR[okp], tZ[okp], te[okp], ne[okp]
    for thr in (0.03, 0.1):
        ri, ro = ts_edges(tR, tZ, te, thr)
        out[f"R_ts_in{int(thr * 1000)}"], out[f"R_ts_out{int(thr * 1000)}"] = ri, ro
    psin = (pa - S.ev(tZ, tR)) / dps
    rho = np.sqrt(np.clip(psin, 0, None))
    pe = te * ne * PE_UNIT
    core = rho < 1.0
    out["n_ts_core"] = int(core.sum())
    out["n_ts_rho05"] = int((rho < 0.5).sum())
    if core.sum() < 5:
        return out

    def prof(v):
        cen, val = [], []
        for b in range(len(RHO_EDGES) - 1):
            lo, hi = RHO_EDGES[b], RHO_EDGES[b + 1]
            sel = (rho >= lo) & (rho < (hi if b < len(RHO_EDGES) - 2 else 1.0))
            cen.append(0.5 * (lo + hi))
            val.append(np.median(v[sel]) if sel.sum() else np.nan)
        cen, val = np.array(cen), np.array(val)
        okb = np.isfinite(val)
        edge_sel = (rho >= 0.97) & (rho < 1.05)
        v_edge = float(np.median(v[edge_sel])) if edge_sel.sum() else 0.0
        if okb.sum() == 0:
            return None, 0
        xs = np.concatenate([[0.0], cen[okb], [1.0]])
        ys = np.concatenate([[val[okb][0]], val[okb], [min(v_edge, val[okb][-1])]])
        return (xs, ys), int(okb.sum())

    pp, nb = prof(pe)
    out["n_bins"] = nb
    if pp is None:
        return out
    rho_c = np.sqrt(np.clip(psin_f[inside], 0, 1))
    dV = g.dVf[inside]
    pe_c = np.interp(rho_c, *pp)
    pe_avg = float(np.sum(pe_c * dV) / V)
    out["pe_avg"] = pe_avg
    out["pe0"] = float(pp[1][0])
    out["pe_peak"] = out["pe0"] / max(pe_avg, 1e-9)
    out["W_e"] = 1.5 * pe_avg * V
    out["pe_r09"] = float(np.interp(0.9, *pp)) / max(pe_avg, 1e-9)
    out["pe_r05"] = float(np.interp(0.5, *pp)) / max(pe_avg, 1e-9)
    # density / temperature profiles
    pn, _ = prof(ne); pt, _ = prof(te)
    if pn is not None:
        ne_c = np.interp(rho_c, *pn)
        out["ne_avg"] = float(np.sum(ne_c * dV) / V)       # 1e19
        out["ne0"] = float(pn[1][0])
        out["ne_peak"] = out["ne0"] / max(out["ne_avg"], 1e-9)
        out["fGW"] = out["ne_avg"] / 10 / (IpMA / (math.pi * a ** 2))
    if pt is not None:
        out["te0"] = float(pt[1][0])
        out["te_avg_n"] = pe_avg / PE_UNIT / max(out.get("ne_avg", np.nan), 1e-9)
        out["te_peak"] = out["te0"] / max(out["te_avg_n"], 1e-9)
    # in / out consistency of the mapped profile (mapping quality): median |log ratio| over shared bins
    side = tR < Ra
    if side.sum() >= 5 and (~side).sum() >= 5:
        lr = []
        for b in range(len(RHO_EDGES) - 1):
            s = (rho >= RHO_EDGES[b]) & (rho < RHO_EDGES[b + 1])
            a_, b_ = pe[s & side], pe[s & ~side]
            if len(a_) >= 2 and len(b_) >= 2 and np.median(a_) > 0 and np.median(b_) > 0:
                lr.append(math.log(np.median(a_) / np.median(b_)))
        out["inout_lr"] = float(np.median(lr)) if lr else np.nan
        out["inout_abs"] = float(np.median(np.abs(lr))) if lr else np.nan
    # gradients
    xs, ys = pp
    gr = np.diff(ys) / np.maximum(np.diff(xs), 1e-6)
    out["pe_gradmax"] = float(-gr.min()) / max(pe_avg, 1e-9)
    out["pe_rho_gradmax"] = float(0.5 * (xs[np.argmin(gr)] + xs[np.argmin(gr) + 1]))
    # betas (electron)
    out["betaN_e"] = 100 * 2 * MU0 * pe_avg * a / (B0 * IpMA) if IpMA > 0 else np.nan
    out["betat_e"] = 2 * MU0 * pe_avg / B0 ** 2
    out["betap_e"] = pe_avg / out["beta_pol_den"] if out["beta_pol_den"] > 0 else np.nan
    return out
