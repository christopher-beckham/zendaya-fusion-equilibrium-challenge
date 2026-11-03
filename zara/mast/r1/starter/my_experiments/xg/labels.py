"""Profile labels for every DIII-D training frame, read off the EFIT flux map (DIII-D labels only).

    python labels.py --workers 4            # -> $R1_ROOT/cache/xg/labels_d3d/<stem>.npz

Per frame, inside the EFIT LCFS: J_phi = -Delta* psi / (mu0 R) (5-point GS stencil on the EFIT grid; closes to
the measured Ip within 0.2 %), fitted by least squares with EFIT's basis
    J = R (pc0 + pc1 x) + (fc0 + fc1 x + fc2 x^2) / (mu0 R),        x = psi_N
(relative residual 0.12 % median). Stored per frame (T,):
    coef (T, 5)   pc0, pc1, fc0, fc1, fc2 (SI, physical convention: Ip > 0, axis = max)
    fit_rel, I_in (A), psi_axis, psi_b (physical), beta_p, li (scorer form with the EFIT LCFS perimeter),
    frac_p (share of Ip carried by the p' term), R_geo, Z_geo, a, kappa, delta_u, delta_l, lp, area, volume,
    W_th (thermal energy from the fitted pressure, J)
NaN where the frame has no usable LCFS.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.interpolate import RegularGridInterpolator
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

MU0 = 4e-7 * np.pi
CACHE = Path(f"{R1_ROOT}/cache/nn/diii_d_train")
OUT = Path(f"{R1_ROOT}/cache/xg/labels_d3d")
KEYS = ["fit_rel", "I_in", "psi_axis", "psi_b", "beta_p", "li", "frac_p", "R_geo", "Z_geo", "a", "kappa",
        "delta_u", "delta_l", "lp", "area", "volume", "W_th"]


def gs_op(psi, R, dR, dZ):
    out = np.zeros_like(psi)
    Rp = 0.5 * (R[1:-1] + R[2:]); Rm = 0.5 * (R[1:-1] + R[:-2])
    c = psi[..., 1:-1, 1:-1]
    out[..., 1:-1, 1:-1] = (R[None, 1:-1] * ((psi[..., 1:-1, 2:] - c) / Rp - (c - psi[..., 1:-1, :-2]) / Rm) / dR ** 2
                            + (psi[..., 2:, 1:-1] - 2 * c + psi[..., :-2, 1:-1]) / dZ ** 2)
    return out


# physical ranges: frames outside them are degenerate (Ip ~ 0, tiny or ill-defined LCFS) and are set to NaN
RANGES = {"li": (0.3, 3.0), "beta_p": (1e-3, 6.0), "frac_p": (-5.0, 5.0), "fit_rel": (0.0, 0.05)}
MIN_IP, MIN_AREA, IIN_RANGE = 1e5, 0.1, (0.8, 1.25)


def sanitize(coef, out, ip):
    """NaN-out frames with degenerate or out-of-range labels. Returns (coef, out, counts per rule)."""
    bad = ~np.isfinite(coef).all(1)
    counts = {"nonfinite_coef": int(bad.sum())}
    for k, (lo, hi) in RANGES.items():
        v = out[k]
        b = ~np.isfinite(v) | (v < lo) | (v > hi)
        counts[f"{k}_out_of_range"] = int((b & ~bad).sum()); bad |= b
    b = ~(ip > MIN_IP); counts["ip_below_1e5"] = int((b & ~bad).sum()); bad |= b
    b = ~(out["area"] > MIN_AREA); counts["area_below_0.1"] = int((b & ~bad).sum()); bad |= b
    r = out["I_in"] / np.where(np.abs(ip) > 0, ip, np.nan)
    b = ~((r > IIN_RANGE[0]) & (r < IIN_RANGE[1])); counts["I_in_over_ip_out"] = int((b & ~bad).sum()); bad |= b
    coef = coef.copy(); coef[bad] = np.nan
    out = {k: np.where(bad, np.nan, v) for k, v in out.items()}
    counts["frames"] = int(len(ip)); counts["dropped"] = int(bad.sum())
    return coef, out, counts


def one(stem: str) -> str:
    d = np.load(CACHE / f"{stem}.npz")
    gR = np.load(CACHE / "basis.npz")["grid_R"].astype(np.float64)
    gZ = np.load(CACHE / "basis.npz")["grid_Z"].astype(np.float64)
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    RR, ZZ = np.meshgrid(gR, gZ)
    pts = np.stack([RR.ravel(), ZZ.ravel()], 1)
    T = len(d["ip"])
    coef = np.full((T, 5), np.nan, np.float64)
    out = {k: np.full(T, np.nan, np.float64) for k in KEYS}
    psi_all = -d["psi"].astype(np.float64)
    J_all = -gs_op(psi_all, gR, dR, dZ) / (MU0 * RR)
    gz_all, gr_all = np.gradient(psi_all, dZ, dR, axis=(1, 2))
    for k in range(T):
        lr, lz = d["lcfs_r"][k], d["lcfs_z"][k]
        ok = np.isfinite(lr) & np.isfinite(lz)
        if ok.sum() < 10:
            continue
        poly = np.stack([lr[ok], lz[ok]], 1)
        inside = MPath(poly).contains_points(pts).reshape(RR.shape)
        if inside.sum() < 30:
            continue
        psi, J = psi_all[k], J_all[k]
        ip = float(d["ip"][k])
        itp = RegularGridInterpolator((gZ, gR), psi)
        psi_b = float(np.median(itp(np.stack([lz[ok], lr[ok]], 1))))
        psi_ax = float(psi[inside].max())
        sw = psi_ax - psi_b
        if not sw > 0:
            continue
        x = np.clip((psi_ax - psi[inside]) / sw, 0, 1)
        Rin = RR[inside]
        A = np.stack([Rin, Rin * x, 1 / (MU0 * Rin), x / (MU0 * Rin), x * x / (MU0 * Rin)], 1)
        c, *_ = np.linalg.lstsq(A, J[inside], rcond=None)
        fit = A @ c
        coef[k] = c
        out["fit_rel"][k] = np.linalg.norm(fit - J[inside]) / np.linalg.norm(J[inside])
        out["I_in"][k] = J[inside].sum() * dR * dZ
        out["psi_axis"][k], out["psi_b"][k] = psi_ax, psi_b
        p = sw * (c[0] * (1 - x) + 0.5 * c[1] * (1 - x * x))
        vw = 2 * np.pi * Rin * dR * dZ
        V = vw.sum()
        seg = np.diff(np.vstack([poly, poly[:1]]), axis=0)
        lp = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        Bpbar = MU0 * ip / lp
        out["beta_p"][k] = 2 * MU0 * (p * vw).sum() / V / Bpbar ** 2
        Bp2 = (gr_all[k] ** 2 + gz_all[k] ** 2) / RR ** 2
        out["li"][k] = (Bp2[inside] * vw).sum() / V / Bpbar ** 2
        out["frac_p"][k] = (A[:, :2] @ c[:2]).sum() * dR * dZ / ip
        out["W_th"][k] = 1.5 * (p * vw).sum()
        Rmin, Rmax = poly[:, 0].min(), poly[:, 0].max()
        Zmin, Zmax = poly[:, 1].min(), poly[:, 1].max()
        a = 0.5 * (Rmax - Rmin); Rg = 0.5 * (Rmax + Rmin)
        out["R_geo"][k], out["Z_geo"][k], out["a"][k] = Rg, 0.5 * (Zmax + Zmin), a
        out["kappa"][k] = (Zmax - Zmin) / (2 * a)
        out["delta_u"][k] = (Rg - poly[np.argmax(poly[:, 1]), 0]) / a
        out["delta_l"][k] = (Rg - poly[np.argmin(poly[:, 1]), 0]) / a
        out["lp"][k] = lp
        R2, Z2 = np.roll(poly[:, 0], -1), np.roll(poly[:, 1], -1)
        cr = poly[:, 0] * Z2 - R2 * poly[:, 1]
        Ar = 0.5 * cr.sum()
        out["area"][k] = abs(Ar)
        out["volume"][k] = 2 * np.pi * abs((cr * (poly[:, 0] + R2)).sum() / 6.0)
    with np.errstate(all="ignore"):
        coef, out, _ = sanitize(coef, out, d["ip"].astype(np.float64))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{stem}.tmp.npz"
    np.savez(tmp, coef=coef.astype(np.float32), **{k: v.astype(np.float32) for k, v in out.items()})
    os.replace(tmp, OUT / f"{stem}.npz")
    return stem


def sanitize_existing(stem: str):
    """Apply sanitize() to an already written label file in place; returns its counts."""
    f = OUT / f"{stem}.npz"
    L = dict(np.load(f))
    ip = np.load(CACHE / f"{stem}.npz")["ip"].astype(np.float64)
    with np.errstate(all="ignore"):
        coef, out, counts = sanitize(L["coef"].astype(np.float64), {k: L[k].astype(np.float64) for k in KEYS}, ip)
    tmp = OUT / f"{stem}.tmp.npz"
    np.savez(tmp, coef=coef.astype(np.float32), **{k: v.astype(np.float32) for k, v in out.items()})
    os.replace(tmp, f)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--sanitize-only", action="store_true", help="apply the physical-range checks to existing label files")
    args = ap.parse_args()
    if args.sanitize_only:
        stems = sorted(p.stem for p in OUT.glob("*.npz") if not p.stem.endswith(".tmp"))
        tot = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for c in ex.map(sanitize_existing, stems, chunksize=32):
                for k, v in c.items():
                    tot[k] = tot.get(k, 0) + v
        print(f"{len(stems)} label files sanitised:", json.dumps(tot, indent=1), flush=True)
        return 0
    stems = sorted(p.stem for p in CACHE.glob("*.npz") if p.stem not in ("basis", "meta"))
    stems = [s for s in stems if not (OUT / f"{s}.npz").exists()][: args.max_shots or None]
    print(f"{len(stems)} shots to label", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(one, stems, chunksize=8)):
            if i % 500 == 0:
                print(f"  {i}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
