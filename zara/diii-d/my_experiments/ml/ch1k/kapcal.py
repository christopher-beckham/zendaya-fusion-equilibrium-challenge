"""Elongation calibration: an area-preserving stretch of the plasma that moves the scorer's kappa of a predicted map toward
a target, leaving triangularity, volume (to first order) and the magnetic axis essentially unchanged.

Forward displacement u (r' = r + u):  u_Z = +(eps / 2) (Z - Z_axis) w(r),  u_R = -(eps / 2) (R - R_geo) w(r)
with w = 1 inside the map's own LCFS and exp(-d^2 / 2 lambda^2) outside (d = distance to the LCFS, lambda = 0.10 m), so the
LCFS is stretched by (1 + eps / 2) vertically and compressed by (1 - eps / 2) horizontally about its geometric centre:
kappa -> kappa (1 + eps), tri unchanged, area / volume unchanged to first order. psi'(r) = psi(r - u(r)) (bicubic spline).
eps is refined by fixed-point steps on the scorer's own kappa of the warped map; kept only if kappa moved closer to the
target and the axis moved < 1 mm.

Target: kappa_map + alpha (T - kappa_map), T = mean of the available estimates: the heads' kappa ("cons" outputs of --src
dirs) and the mean kappa of the member maps (--members; the ensemble-mean map blurs the X-point region).

    python kapcal.py --preds <ens preds> --out <dir> --src <head dirs> --members <member preds dirs> --alpha 0.5
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
from shapecal import MC, RR, ZZ, R, Z, geo, lcfs, map_geo  # noqa: E402


def kappa_of(g):
    return (g[0] - g[1]) / (g[2] - g[3])


def shape3(g):
    """kappa, tri_top, tri_bot, a from LCFS extremes (scorer definitions)."""
    a = 0.5 * (g[2] - g[3])
    Rg = 0.5 * (g[2] + g[3])
    return np.array([(g[0] - g[1]) / (2 * a), (Rg - g[4]) / a, (Rg - g[5]) / a]), a


def cal_frame(psi, tgt, lam=0.10, n_iter=4, tol=2e-4, max_eps=0.05, max_shear=0.04, upper_only=False, max_axis_mm=1.0,
              retry=False, anchor=False, xthr=0.3):
    """tgt = (kappa*, tri_top*, tri_bot*), NaN = not targeted. kappa by the area-preserving stretch (eps); tri_top / tri_bot by
    a shear of the upper / lower plasma, u_R = delta zeta^2 (zeta = (Z - Z_axis) / (Z_extreme - Z_axis) >= 0), which moves the
    top / bottom point horizontally by delta and leaves the midplane, Zmax / Zmin and the axis in place."""
    from derive import magnetic_axis
    from scipy.interpolate import RectBivariateSpline
    from derive import _points_in_polygon
    tgt = np.asarray(tgt, dtype=np.float64)
    tm = np.isfinite(tgt)
    C = lcfs(psi)
    if C is None or not tm.any():
        return None, None
    g0 = geo(C)
    s0, a0 = shape3(g0)
    t = np.where(tm, tgt, s0)
    if np.all(np.abs(t - s0) < tol):
        return None, None
    Ra, Za = magnetic_axis(psi, R, Z, "DIII-D", MC)[:2]
    Rg = 0.5 * (g0[2] + g0[3])
    Zc, Rc = Za, Rg                         # stretch centre: axis height, geometric centre (default)
    if anchor:                              # (ch1k) single X-point end: stretch about the X-point (it stays in place)
        from shapek import end_grad_ratio
        rt, rb = end_grad_ratio(psi, C)
        if (rt < xthr) != (rb < xthr):
            j = int(np.argmax(C[:, 1]) if rt < xthr else np.argmin(C[:, 1]))
            Rc, Zc = float(C[j, 0]), float(C[j, 1])
    ins = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    Cs = C[::4]
    dist = np.sqrt(((RR.ravel()[:, None] - Cs[None, :, 0]) ** 2 + (ZZ.ravel()[:, None] - Cs[None, :, 1]) ** 2).min(1)).reshape(RR.shape)
    w = np.where(ins, 1.0, np.exp(-dist ** 2 / (2 * lam ** 2)))
    zu = np.clip((ZZ - Za) / max(g0[0] - Za, 1e-3), 0.0, None) ** 2          # upper shear profile (1 at Zmax)
    zl = np.clip((Za - ZZ) / max(Za - g0[1], 1e-3), 0.0, None) ** 2          # lower shear profile (1 at Zmin)
    if upper_only:          # X-point / divertor half frozen: no lower shear, kappa through the top only
        tm[2] = False
        t[2] = s0[2]
        x = np.clip(2.0 * (ZZ - Za) / max(g0[0] - Za, 1e-3), 0.0, 1.0)
        pu = x * x * (3.0 - 2.0 * x)                                          # 0 below the axis, 1 in the upper half
        hu = max(g0[0] - Za, 1e-3)
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    if upper_only:
        eps = (t[0] - s0[0]) * 2 * a0 / hu if tm[0] else 0.0                  # relative vertical stretch of the upper half
    else:
        eps = t[0] / s0[0] - 1.0 if tm[0] else 0.0
    dt = -a0 * (t[1] - s0[1]) if tm[1] else 0.0
    db = -a0 * (t[2] - s0[2]) if tm[2] else 0.0
    best, best_err, s_best = None, np.abs(t - s0)[tm].sum(), s0
    for _ in range(n_iter):
        eps = float(np.clip(eps, -max_eps, max_eps))
        dt, db = float(np.clip(dt, -max_shear, max_shear)), float(np.clip(db, -max_shear, max_shear))
        if upper_only:
            uZ = eps * (ZZ - Za) * pu * w
            uR = dt * zu * w
        else:
            uZ = 0.5 * eps * (ZZ - Zc) * w
            uR = (-0.5 * eps * (RR - Rc) + dt * zu + db * zl) * w
        new = spl.ev(ZZ - uZ, RR - uR)
        C1 = lcfs(new)
        if C1 is None:
            break
        s1, a1_ = shape3(geo(C1))
        err = np.abs(t - s1)[tm].sum()
        if err < best_err:
            best, best_err, s_best = new, err, s1
        if np.all(np.abs(t - s1)[tm] < tol):
            break
        if tm[0] and upper_only:
            eps = eps + (t[0] - s1[0]) * 2 * a1_ / hu
        elif tm[0]:
            eps = eps + (t[0] / s1[0] - 1.0)
        if tm[1]:
            dt = dt - a1_ * (t[1] - s1[1])
        if tm[2]:
            db = db - a1_ * (t[2] - s1[2])
    if best is None and retry and not upper_only and tm[0] and not (tm[1] or tm[2]):
        # (ch1k) the full step lost the LCFS or moved kappa away (often a second X-point taking over): smaller single steps
        e0 = float(np.clip(t[0] / s0[0] - 1.0, -max_eps, max_eps))
        for f in (0.5, 0.25):
            ee = f * e0
            new = spl.ev(ZZ - 0.5 * ee * (ZZ - Zc) * w, RR + 0.5 * ee * (RR - Rc) * w)
            C1 = lcfs(new)
            if C1 is None:
                continue
            s1, _ = shape3(geo(C1))
            err = np.abs(t - s1)[tm].sum()
            if err < best_err:
                best, best_err, s_best, eps = new, err, s1, ee
                break
    info = {"s0": s0.tolist(), "t": t.tolist(), "s1": s_best.tolist(), "eps": eps, "dt": dt, "db": db}
    if best is None:
        info["kept"] = False
        return None, info
    best = best.astype(np.float32).astype(np.float64)
    a1 = magnetic_axis(best, R, Z, "DIII-D", MC)[:2]
    info["axis_move_mm"] = 1e3 * float(np.hypot(a1[0] - Ra, a1[1] - Za))
    info["kept"] = bool(info["axis_move_mm"] < max_axis_mm)
    return (best.astype(np.float32) if info["kept"] else None), info


def one_shot(args):
    src, dst, tk, upper = args[:4]
    kw = args[4] if len(args) > 4 else {}
    p = dict(np.load(src))
    infos = []
    for k in range(len(p["psi"])):
        new, info = cal_frame(p["psi"][k].astype(np.float64), tk[k], upper_only=upper, **kw)
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
    ap.add_argument("--src", type=Path, nargs="*", default=[])
    ap.add_argument("--members", type=Path, nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.5, help="blend weight of the kappa target")
    ap.add_argument("--alpha-tri", type=float, default=0.0, help="blend weight of the tri_top / tri_bot targets (0 = off)")
    ap.add_argument("--upper-only", action="store_true",
                    help="freeze the lower (X-point / divertor) half: kappa through a stretch of the upper half, tri_top shear only")
    ap.add_argument("--agree-gate", action="store_true",
                    help="target a quantity only where heads and member maps both move it in the same direction")
    ap.add_argument("--max-axis-mm", type=float, default=1.0, help="(ch1k) reject a frame whose axis moved more (mm)")
    ap.add_argument("--max-eps", type=float, default=0.05, help="(ch1k) largest relative stretch")
    ap.add_argument("--retry", action="store_true", help="(ch1k) retry frames whose full step failed with 1/2 and 1/4 steps")
    ap.add_argument("--anchor", action="store_true", help="(ch1k) stretch about the X-point when exactly one end is an X-point")
    ap.add_argument("--targets", type=Path, default=None,
                    help="(ch1k) dir of per-shot target files with absolute 'kappa' (and optional 'tri_top', 'tri_bot') (T,)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.preds.glob("*.npz"))
    assert a.src or a.members or a.targets
    t0 = time.time()

    def s3(g):          # (T, 6) extremes -> (T, 3) kappa, tri_top, tri_bot
        aa = 0.5 * (g[:, 2] - g[:, 3])
        Rg = 0.5 * (g[:, 2] + g[:, 3])
        return np.c_[(g[:, 0] - g[:, 1]) / (2 * aa), (Rg - g[:, 4]) / aa, (Rg - g[:, 5]) / aa]

    al = np.array([a.alpha, a.alpha_tri, a.alpha_tri])
    if a.targets is not None:
        jobs = []
        for f in files:
            z = np.load(a.targets / f.name)
            T = len(z["kappa"])
            tg = np.c_[z["kappa"], z["tri_top"] if "tri_top" in z.files else np.full(T, np.nan),
                       z["tri_bot"] if "tri_bot" in z.files else np.full(T, np.nan)].astype(np.float64)
            jobs.append((str(f), str(a.out / f.name), tg, a.upper_only, {"max_axis_mm": a.max_axis_mm, "max_eps": a.max_eps, "retry": a.retry, "anchor": a.anchor}))
        rep = {}
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for stem, infos in pool.map(one_shot, jobs, chunksize=2):
                if infos:
                    rep[stem] = infos
        n_t = sum(len(v) for v in rep.values())
        n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
        (a.out.parent / "kapcal_report.json").write_text(json.dumps(rep, default=float))
        print(f"{len(files)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")
        return
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        mg = list(pool.map(map_geo, [str(f) for f in files], chunksize=2))
        mems = None
        if a.members:
            memg = [list(pool.map(map_geo, [str(md / f.name) for f in files], chunksize=2)) for md in a.members]
            mems = [np.nanmean(np.stack([s3(mm[i]) for mm in memg]), 0) for i in range(len(files))]
        jobs = []
        for i, (f, m) in enumerate(zip(files, mg)):
            sm = s3(m)
            est = []
            if a.src:
                est.append(np.mean([np.load(sd / f.name)["cons"][:, 2:5].astype(np.float64) for sd in a.src], 0))
            if mems is not None:
                est.append(mems[i])
            T = np.nanmean(np.stack(est), 0)
            tg = sm + al[None] * (T - sm)
            tg[:, al == 0] = np.nan
            if a.agree_gate and len(est) == 2:
                # only where the heads and the member maps move the map value in the same direction
                agree = np.sign(est[0] - sm) == np.sign(est[1] - sm)
                tg = np.where(agree, tg, np.nan)
            jobs.append((str(f), str(a.out / f.name), tg, a.upper_only))
        rep = {}
        for stem, infos in pool.map(one_shot, jobs, chunksize=2):
            if infos:
                rep[stem] = infos
    n_t = sum(len(v) for v in rep.values())
    n_k = sum(sum(1 for i in v if i.get("kept")) for v in rep.values())
    (a.out.parent / "kapcal_report.json").write_text(json.dumps(rep, default=float))
    print(f"{len(files)} shots, {n_t} frames targeted, {n_k} modified, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
