"""Shape calibration that leaves X-point corners in place (ml/ch1k).

Two warps psi'(r) = psi(r - u(r)) (bicubic spline of the map), applied only at a SMOOTH end of the predicted LCFS (top or
bottom extreme point without a saddle: |grad psi| there >= --xthr times its median on the LCFS); an end at an X-point is
never moved, so the corner where the D_LCFS maximum usually sits stays where the map put it.

  end   vertical move of the smooth end to a target Zmax* (Zmin*): u_Z = d s(zeta) w(r), zeta = (Z - Z_axis) / (Z_end - Z_axis)
        clipped to [0, 1], s = smoothstep, w = 1 inside the LCFS and exp(-dist^2 / 2 lam^2) outside. The axis level, the
        midplane points and the other end stay in place. Changes kappa (and volume); tri unchanged.
  tilt  local tilt of the smooth end toward a triangularity target: u_Z = s (R - R_end) exp(-|r - P_end|^2 / 2 L^2). On a
        flat top / bottom a slope of a few per cent moves the extreme point (R at Zmax / Zmin) by cm while the boundary moves by
        ~1 mm. s is chosen on the map's own contour (grid search; the warp moves the contour to C + u(C) to first order) and
        checked with the scorer's extraction of the warped map.
Targets: per-shot files (targets.py) with Zmax / Zmin (end) and tri_top / tri_bot (tilt), NaN = no target. A frame is kept
only if the targeted quantities moved closer to their targets and the magnetic axis moved < 1 mm.

    python shapek.py --preds DIR --out DIR --targets DIR --modes end,tilt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapecal import MC, RR, ZZ, R, Z, lcfs  # noqa: E402


def tri_of(C):
    Rmax, Rmin = C[:, 0].max(), C[:, 0].min()
    a, Rg = 0.5 * (Rmax - Rmin), 0.5 * (Rmax + Rmin)
    return (Rg - C[np.argmax(C[:, 1]), 0]) / a, (Rg - C[np.argmin(C[:, 1]), 0]) / a


def end_grad_ratio(psi, C):
    from derive import _bilinear
    gz, gR = np.gradient(psi, Z, R)
    b = _bilinear(np.hypot(gz, gR), R, Z, C[:, 0], C[:, 1])
    med = np.median(b)
    return b[np.argmax(C[:, 1])] / med, b[np.argmin(C[:, 1])] / med


def tilt_slope(C, top, tgt, L, smax=0.6, n=121, W=64):
    """slope s whose tilt moves this end's triangularity closest to tgt, on the contour (0 if nothing better)."""
    P_ = len(C)
    j = int(np.argmax(C[:, 1]) if top else np.argmin(C[:, 1]))
    P = C[j]
    Rmax, Rmin = C[:, 0].max(), C[:, 0].min()
    a, Rg = 0.5 * (Rmax - Rmin), 0.5 * (Rmax + Rmin)
    w = C[(j + np.arange(-W, W + 1)) % P_]
    dR, dZ = w[:, 0] - P[0], w[:, 1] - P[1]
    prof = dR * np.exp(-(dR ** 2 + dZ ** 2) / (2 * L * L))
    grid = np.linspace(-smax, smax, n)
    grid = grid[np.argsort(np.abs(grid), kind="stable")]
    best_s, best_e = 0.0, abs((Rg - P[0]) / a - tgt)
    for sv in grid[1:]:
        zn = w[:, 1] + sv * prof
        jj = int(np.argmax(zn) if top else np.argmin(zn))
        e = abs((Rg - w[jj, 0]) / a - tgt)
        if e < best_e - 1e-4:
            best_s, best_e = sv, e
    return best_s, P


def cal_frame(psi, tg, modes, xthr=0.3, L=0.15, lam=0.10, maxd=0.03, smax=0.6, end_which="smooth"):
    """tg: dict of targets (Zmax, Zmin, tri_top, tri_bot; NaN = none) -> (new psi or None, info)."""
    from derive import magnetic_axis, _points_in_polygon
    from scipy.interpolate import RectBivariateSpline
    C = lcfs(psi)
    if C is None:
        return None, None
    rt, rb = end_grad_ratio(psi, C)
    smooth_t, smooth_b = rt >= xthr, rb >= xthr
    Ra, Za = magnetic_axis(psi, R, Z, "DIII-D", MC)[:2]
    Zmax, Zmin = C[:, 1].max(), C[:, 1].min()
    t0, b0 = tri_of(C)
    uZ = np.zeros_like(RR)
    did = {}
    if "end" in modes:
        mt = {"smooth": smooth_t, "x": not smooth_t, "all": True}[end_which]
        mb = {"smooth": smooth_b, "x": not smooth_b, "all": True}[end_which]
        dt = tg["Zmax"] - Zmax if mt and np.isfinite(tg["Zmax"]) else 0.0
        db = tg["Zmin"] - Zmin if mb and np.isfinite(tg["Zmin"]) else 0.0
        dt, db = float(np.clip(dt, -maxd, maxd)), float(np.clip(db, -maxd, maxd))
        if abs(dt) > 2e-4 or abs(db) > 2e-4:
            ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
            Cs = C[::4]
            dist = np.sqrt(((RR.ravel()[:, None] - Cs[None, :, 0]) ** 2 + (ZZ.ravel()[:, None] - Cs[None, :, 1]) ** 2).min(1)).reshape(RR.shape)
            w = np.where(ins, 1.0, np.exp(-dist ** 2 / (2 * lam ** 2)))
            zt = np.clip((ZZ - Za) / max(Zmax - Za, 1e-3), 0, 1)
            zb = np.clip((Za - ZZ) / max(Za - Zmin, 1e-3), 0, 1)
            uZ += (dt * zt * zt * (3 - 2 * zt) + db * zb * zb * (3 - 2 * zb)) * w
            did.update(dt=dt, db=db)
    if "tilt" in modes:
        for top, key, ok in ((True, "tri_top", smooth_t), (False, "tri_bot", smooth_b)):
            if not ok or not np.isfinite(tg[key]):
                continue
            Cc = C.copy()
            if np.any(uZ != 0):                 # contour after the end move (first order)
                from derive import _bilinear
                Cc[:, 1] += _bilinear(uZ, R, Z, C[:, 0], C[:, 1])
            sv, P = tilt_slope(Cc, top, tg[key], L, smax=smax)
            if sv != 0.0:
                uZ += sv * (RR - P[0]) * np.exp(-((RR - P[0]) ** 2 + (ZZ - P[1]) ** 2) / (2 * L * L))
                did["s_top" if top else "s_bot"] = sv
    if not did:
        return None, None
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    new = spl.ev(ZZ - uZ, RR).astype(np.float32).astype(np.float64)
    C1 = lcfs(new)
    info = {"rt": rt, "rb": rb, **did}
    if C1 is None:
        info["kept"] = False
        return None, info
    t1, b1 = tri_of(C1)
    val0 = {"Zmax": Zmax, "Zmin": Zmin, "tri_top": t0, "tri_bot": b0}
    val1 = {"Zmax": C1[:, 1].max(), "Zmin": C1[:, 1].min(), "tri_top": t1, "tri_bot": b1}
    keys = ([k for k, d in (("Zmax", "dt"), ("Zmin", "db")) if d in did and did[d] != 0.0]
            + [k for k, d in (("tri_top", "s_top"), ("tri_bot", "s_bot")) if d in did])
    sc = {"Zmax": 1.0, "Zmin": 1.0, "tri_top": 0.02, "tri_bot": 0.02}       # tri error in units of ~1 cm / a
    e0 = sum(abs(tg[k] - val0[k]) / sc[k] for k in keys)
    e1 = sum(abs(tg[k] - val1[k]) / sc[k] for k in keys)
    a1 = magnetic_axis(new, R, Z, "DIII-D", MC)[:2]
    info.update({k + "_0": val0[k] for k in keys})
    info.update({k + "_1": val1[k] for k in keys})
    info["axis_move_mm"] = 1e3 * float(np.hypot(a1[0] - Ra, a1[1] - Za))
    info["kept"] = bool(e1 < e0 and info["axis_move_mm"] < 1.0)
    return (new.astype(np.float32) if info["kept"] else None), info


def one_shot(args):
    src, dst, tgf, modes, kw = args
    p = dict(np.load(src))
    z = np.load(tgf)
    T = len(p["psi"])
    tg = {k: (z[k].astype(np.float64) if k in z.files else np.full(T, np.nan)) for k in ("Zmax", "Zmin", "tri_top", "tri_bot")}
    infos = []
    for k in range(T):
        new, info = cal_frame(p["psi"][k].astype(np.float64), {n: v[k] for n, v in tg.items()}, modes, **kw)
        if info is not None:
            info["k"] = k
            infos.append(info)
        if new is not None:
            p["psi"][k] = new
    np.savez(dst, **p)
    return Path(src).stem, infos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--modes", default="end,tilt")
    ap.add_argument("--xthr", type=float, default=0.3)
    ap.add_argument("--L", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.10)
    ap.add_argument("--maxd", type=float, default=0.03)
    ap.add_argument("--smax", type=float, default=0.6)
    ap.add_argument("--end-which", default="smooth", choices=["smooth", "x", "all"],
                    help="which ends the end move may touch (x = only X-point ends; all = both ends)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.preds.glob("*.npz"))
    modes = set(a.modes.split(","))
    kw = dict(xthr=a.xthr, L=a.L, lam=a.lam, maxd=a.maxd, smax=a.smax, end_which=a.end_which)
    t0 = time.time()
    jobs = [(str(f), str(a.out / f.name), str(a.targets / f.name), modes, kw) for f in files]
    rep = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for stem, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
    n_t = sum(len(v) for v in rep.values())
    n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "shapek_report.json").write_text(json.dumps(rep, default=float))
    print(f"{len(files)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
