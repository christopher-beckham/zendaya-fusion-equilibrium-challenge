"""Label-free physics checks of predicted flux maps against MAST's own inputs, calibrated on DIII-D truth.

usage: python physics_suite.py CKPT [CKPT ...]

Each check is computed for DIII-D val true maps (should be near the physical value), each model's DIII-D maps, each
model's MAST maps and, as a reference, MAST coil picture + prior. Per frame, the scorer's own axis and boundary give
ψ_N = (φ_axis − φ) / (φ_axis − φ_boundary), φ = −ψ, read at every Thomson channel (bilinear).

  edge       hot points (Te ≥ 300 eV) outside the boundary (ψ_N > 1.02), and cold points (Te ≤ 30 eV) deep inside
             (ψ_N < 0.85): shares of such channels. Both are unphysical.
  midplane   where the boundary crosses Z ≈ Z_axis on each side, against where Te along the midplane chord falls
             through 50 eV on that side (MAST: both sides; DIII-D: outboard only when the chord reaches it). cm.
  order      Te, ne and p = ne·Te should fall with ψ_N: share of channel pairs with value_i > 1.3·value_j but
             ψ_N,i > ψ_N,j + 0.02.
  betaN      βN from Thomson: p_e(ψ_N) interpolated from the channels, averaged over the plasma volume (∫ p R dA /
             ∫ R dA), β_t = 2 μ0 <p_e> / B0², βN = 100 β_t a B0 / Ip[MA] (electron pressure only). Bt from the
             q_ITER feature, inverted. Reported as the ratio to the label (DIII-D) or to the βN head (MAST).
  q95        q95 from the map, q = F/(2π) ∮ dl / (R |∇ψ|) at ψ_N = 0.95, F = R0·B0: ratio to the label (DIII-D)
             or to the q95 head (MAST).
Uses MAST inputs only; no MAST label and no demo shot is read.
"""
import os
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]   # the starter tree this file sits in
sys.path.insert(0, str(W / "my_experiments" / "nn"))
import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as MPath
from scipy.interpolate import RectBivariateSpline

import features_v2 as fv
import features_v5 as f5
from common import IP_COL, checkpoint_pictures, external_flux, stack_history
from model import load_checkpoint, predict_raw, prior_from_table
from physics_loss import MastInputs

sys.modules.pop("common")
sys.path.insert(0, str(W / "fusion_scoring"))
from common import AXIS_SIGN  # noqa: E402
from contour import encloses_point, find_contours, grid_to_physical, is_closed  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
from o_point import find_o_point  # noqa: E402

C = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
DEV = torch.device("cpu")
MU0 = 4e-7 * np.pi
S = AXIS_SIGN["DIII-D"]
KAPPA = fv.KAPPA


def bt_from_qiter(q_iter, ip_ka, r0, a):
    eps = a / r0
    shape = (1 + KAPPA ** 2) / 2 * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2
    ip_a = np.maximum(np.abs(ip_ka), fv.IP_FLOOR_KA) * 1000.0
    return q_iter / shape * MU0 * r0 * ip_a / (2 * np.pi * a ** 2)


class Machine:
    def __init__(self, name):
        key = "d3d" if name == "DIII-D" else "mast"
        g = fv.geometry(key)
        self.name, self.r, self.z, self.r0, self.a = name, g["r"], g["z"], g["R0"], g["a"]
        if key == "mast":   # the region mast_region.py selects (FEC_MAST_MASK); DIII-D keeps its envelope
            import mast_region
            self.mask = mast_region.mask().astype(bool)
        else:
            self.mask = np.load(W / f"fusion_scoring/masks/{key}_envelope.npz")["mask_coarse"].astype(bool)
        rr, zz = np.meshgrid(self.r, self.z, indexing="xy")
        self.RR, self.pts = rr, np.column_stack([rr.ravel(), zz.ravel()])
        self.dA = (self.r[1] - self.r[0]) * (self.z[1] - self.z[0])
        split = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # MAST Thomson rows follow the test set
        sfx = "" if key == "d3d" or split == "mast_public_test" else f"_{split}"
        te = np.load(C / f"thomson_raw_{key}{sfx}.npz"); ne = np.load(C / f"thomson_raw_ne_{key}{sfx}.npz")
        self.te, self.ne, self.cr, self.cz = te["te"], ne["te"], te["r"], te["z"]


def frame_checks(m: Machine, psi, row, ip_ka, bt):
    """One frame: dict of per-check values, or None without a boundary."""
    psi = np.asarray(psi, np.float64).reshape(65, 65)
    phi = S * psi
    lc, _ = extract_lcfs_with_sign(psi, m.r, m.z, m.name, m.mask, m.mask.astype(float), axis_sign=S)
    if lc is None:
        return None
    ra, za, iz, ir, _ = find_o_point(phi, m.r, m.z, m.mask)
    sp = RectBivariateSpline(m.z, m.r, phi)
    ax = float(sp(za, ra)[0, 0]); bd = float(np.mean(sp(lc[:, 1], lc[:, 0], grid=False)))
    span = ax - bd
    te = m.te[row].astype(np.float64); ne = m.ne[row].astype(np.float64)
    cr = m.cr[row].astype(np.float64); cz = m.cz[row].astype(np.float64)
    ok = np.isfinite(te) & (cr >= m.r[0]) & (cr <= m.r[-1]) & (cz >= m.z[0]) & (cz <= m.z[-1])
    out = {}
    if ok.sum() >= 4:
        psin = (ax - sp(cz[ok], cr[ok], grid=False)) / span
        t, n = te[ok], ne[ok]
        hot, cold = t >= 300, t <= 30
        out["hot_out"] = float(np.mean(psin[hot] > 1.02)) if hot.any() else np.nan
        out["cold_in"] = float(np.mean(psin[cold] < 0.85)) if cold.any() else np.nan
        rng = np.random.default_rng(int(row))
        i, j = rng.integers(0, len(t), 400), rng.integers(0, len(t), 400)
        for q, v in (("te", t), ("ne", n), ("p", t * n)):
            vi, vj = v[i], v[j]
            sel = np.isfinite(vi) & np.isfinite(vj) & (vi > 1.3 * vj)
            out[f"order_{q}"] = float(np.mean(psin[i][sel] > psin[j][sel] + 0.02)) if sel.any() else np.nan
        # midplane boundary vs Te = 50 eV crossing, per side
        mid = np.abs(cz[ok] - za) < 0.1
        zl = za
        c = lc[np.argsort(lc[:, 0])]
        near = np.abs(c[:, 1] - zl) < 0.03
        if near.any() and mid.sum() >= 4:
            rb_in, rb_out = c[near, 0].min(), c[near, 0].max()
            rm, tm = cr[ok][mid], t[mid]
            o = np.argsort(rm); rm, tm = rm[o], tm[o]
            for side, sel, rb in (("in", rm < ra, rb_in), ("out", rm > ra, rb_out)):
                rs, ts = rm[sel], tm[sel]
                if len(rs) >= 3 and ts.min() < 50 < ts.max():
                    k = np.nonzero(np.diff(np.sign(ts - 50)))[0]
                    if len(k):
                        kk = k[0] if side == "out" else k[-1]
                        r50 = rs[kk] + (50 - ts[kk]) * (rs[kk + 1] - rs[kk]) / (ts[kk + 1] - ts[kk])
                        out[f"mid_{side}_cm"] = float((rb - r50) * 100)
        # βN from Thomson electron pressure averaged over the plasma volume
        pe = t * n * 1e19 * 1.602e-19
        good = np.isfinite(pe) & (psin >= 0) & (psin <= 1.05)
        inside = MPath(lc).contains_points(m.pts).reshape(65, 65)
        if good.sum() >= 4 and inside.sum() > 10 and np.isfinite(bt) and bt > 0:
            o = np.argsort(psin[good])
            pn, pv = psin[good][o], pe[good][o]
            pix = np.clip((ax - phi[inside]) / span, 0, 1)
            pp = np.interp(pix, pn, pv, left=pv[0], right=pv[-1])
            pavg = np.sum(pp * m.RR[inside]) / np.sum(m.RR[inside])
            beta_t = 2 * MU0 * pavg / bt ** 2
            out["betaN_th"] = float(100 * beta_t * m.a * bt / (abs(ip_ka) / 1000))
    # q95 from the map
    lev = ax - 0.95 * span
    for cont in find_contours(phi, lev):
        if len(cont) < 8 or not is_closed(cont) or not encloses_point(cont, iz, ir):
            continue
        p = grid_to_physical(cont, m.r, m.z)
        midp = 0.5 * (p[1:] + p[:-1]); dl = np.hypot(*np.diff(p, axis=0).T)
        gz = sp(midp[:, 1], midp[:, 0], dx=1, grid=False); gr = sp(midp[:, 1], midp[:, 0], dy=1, grid=False)
        if np.isfinite(bt):
            out["q95_map"] = float(m.r0 * bt / (2 * np.pi) * np.sum(dl / (midp[:, 0] * np.hypot(gr, gz))))
        break
    return out


def summarise(name, rows, ref=None):
    df = pd.DataFrame(rows)
    cells = [f"frames {len(df)}"]
    for k in ("hot_out", "cold_in", "order_te", "order_ne", "order_p"):
        if k in df:
            cells.append(f"{k} {df[k].mean():.3f}")
    for k in ("mid_in_cm", "mid_out_cm"):
        if k in df and df[k].notna().any():
            cells.append(f"{k} {df[k].median():+.1f} (|.| {df[k].abs().median():.1f}, n {df[k].notna().sum()})")
    for k, lab in (("betaN_th", "betaN_ref"), ("q95_map", "q95_ref")):
        if k in df and lab in df:
            rr = df[k] / df[lab]
            cells.append(f"{k}/{lab[:-4]} median {rr.median():.2f} (10–90% {rr.quantile(.1):.2f}–{rr.quantile(.9):.2f})")
            if ref == "label":
                y, p = df[lab], df[k]
                okk = y.notna() & p.notna()
                cc = np.corrcoef(y[okk], p[okk])[0, 1] if okk.sum() > 3 else np.nan
                cells.append(f"corr({k},label) {cc:.2f}")
    print(f"  {name:36s} " + " | ".join(cells), flush=True)


def d3d_rows(m, net, ck, val, model):
    norm = ck["norm"]; pics = checkpoint_pictures(ck["args"])
    fm = np.load(C / "features_v5.npy", mmap_mode="r"); co = np.load(C / "coils.npy", mmap_mode="r")
    ps = np.load(C / "psi.npy", mmap_mode="r"); sc = np.load(C / "scal.npy", mmap_mode="r")
    rows = []
    for r in val.itertuples():
        sl = slice(int(r.start), int(r.start + r.n))
        raw = np.array(fm[sl], np.float32); coils = np.asarray(co[sl], np.float32); ip = coils[:, IP_COL]
        bt = bt_from_qiter(raw[:, f5.FEATURES.index("q_ITER")], ip, m.r0, m.a)
        if model:
            maps, qh, bh = predict_raw(net, stack_history(raw[:, list(norm["feature_columns"])], int(norm.get("history_frames", 0))),
                                       external_flux(coils, pics["green"], pics["ecoil"]), norm, DEV, ip_ka=ip,
                                       rax_th=raw[:, f5.FEATURES.index("R_ax_th")])
        else:
            maps = np.asarray(ps[sl], np.float64)
        for k in range(0, len(ip), 4):
            if abs(ip[k]) < 300:
                continue
            o = frame_checks(m, maps[k], int(r.start) + k, ip[k], bt[k])
            if o is not None:
                o["betaN_ref"], o["q95_ref"] = float(sc[sl][k, 1]), float(sc[sl][k, 0])
                rows.append(o)
    return rows


if __name__ == "__main__":
    ckpts = [Path(p) for p in sys.argv[1:]]
    d3d, mast = Machine("DIII-D"), Machine("MAST")
    sh = pd.read_parquet(C / "shots.parquet"); val = sh[sh.split == "val"].sample(10, random_state=5)
    net, ck = load_checkpoint(ckpts[0], DEV)
    print("## DIII-D val (10 shots, every 4th frame); betaN and q95 ratios are to the labels")
    summarise("true maps", d3d_rows(d3d, net, ck, val, model=False), ref="label")
    fq = f5.FEATURES.index("q_ITER")
    for n, ckpt in enumerate(ckpts):
        net, ck = load_checkpoint(ckpt, DEV); net.eval()
        print(f"## {ckpt.parent.name}/{ckpt.stem}")
        summarise("DIII-D model", d3d_rows(d3d, net, ck, val, model=True), ref="label")
        variant = ck["args"].get("mast_variant", "") or ""
        mi = MastInputs(C, ck["norm"], DEV, variant=variant)
        torch.manual_seed(0); idx = torch.randint(0, mi.n, (600,))
        import mast_region
        feat = np.load(C / f"features_v5_mast_public_test{variant}{mast_region.suffix()}.npy", mmap_mode="r")
        rows_c = mi.rows[idx.numpy()]
        ip = mi.ip[idx].numpy()
        bt = bt_from_qiter(np.asarray(feat[rows_c])[:, fq], ip, mast.r0, mast.a)
        with torch.no_grad():
            pn, sc = net(mi.coils[idx], mi.cur[idx] @ mi.green, mi.ip[idx], mi.rax[idx], coord=mi.coord,
                         prior=prior_from_table(mi.prior_table, mi.ip[idx], mi.rax[idx]), r0=mi.r0)
        phys = (pn * net.psi_std + net.pixel_mean).numpy()
        heads = sc.numpy() * ck["norm"]["scal_std"] + ck["norm"]["scal_mean"]
        rows = []
        for k in range(len(idx)):
            o = frame_checks(mast, phys[k], int(rows_c[k]), ip[k], bt[k])
            if o is not None:
                o["q95_ref"], o["betaN_ref"] = float(heads[k, 0]), float(heads[k, 1])
                rows.append(o)
        summarise(f"MAST model ({variant or 'x1'}), vs heads", rows)
        if n == 0:
            base = (mi.cur[idx] @ mi.green + prior_from_table(mi.prior_table, mi.ip[idx], mi.rax[idx])).numpy()
            rows = [o for k in range(len(idx)) if (o := frame_checks(mast, base[k], int(rows_c[k]), ip[k], bt[k]))]
            summarise("MAST coil picture + prior", rows)
