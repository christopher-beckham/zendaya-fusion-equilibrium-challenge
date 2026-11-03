"""Ceiling of the hard-constrained representation psi = sum_c g_c I_c M_c + G J (+ harmonic correction) on DIII-D.

    python ceiling.py --n-shots 100 --out runs/ceiling_100

J is the TRUE plasma current, read off the EFIT map (J = -Delta* psi / (mu0 R) inside the EFIT LCFS, on the
coil-derived domain), so the only errors left are the representation's: missing vessel currents, coil-current
errors, discretisation. Variants written as separate prediction folders (scored with the competition scorer):
    exact     coils as shipped (machine spec) + G J_true
    harm      + per-frame least-squares vacuum harmonics (toroidal multipoles, 7 terms) fitted on domain cells
    gain      + global per-coil gains fitted by least squares on these frames (ceiling of the gain idea)
    gainharm  both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "c1" / "nn"))
from machine import d3d, D3D_CACHE, MU0  # noqa: E402
from train_v5 import split_stems, DATA_DIR  # noqa: E402


def gs_op(psi, R, dR, dZ):
    out = np.zeros_like(psi)
    Rp = 0.5 * (R[1:-1] + R[2:]); Rm = 0.5 * (R[1:-1] + R[:-2])
    c = psi[..., 1:-1, 1:-1]
    out[..., 1:-1, 1:-1] = (R[None, 1:-1] * ((psi[..., 1:-1, 2:] - c) / Rp - (c - psi[..., 1:-1, :-2]) / Rm) / dR ** 2
                            + (psi[..., 2:, 1:-1] - 2 * c + psi[..., :-2, 1:-1]) / dZ ** 2)
    return out


def harmonics(RR, ZZ, R0, Z0):
    """Polynomial solutions of Delta* h = 0 about (R0, Z0), x = R/R0, y = (Z - Z0)/R0. (7, nZ, nR)"""
    x, y = RR / R0, (ZZ - Z0) / R0
    return np.stack([np.ones_like(x), x ** 2, y, x ** 2 * y, x ** 4 - 4 * x ** 2 * y ** 2,
                     3 * x ** 4 * y - 4 * x ** 2 * y ** 3, y ** 2 - x ** 2 * np.log(x)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shots", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = d3d()
    gR, gZ = m.grid_R.astype(np.float64), m.grid_Z.astype(np.float64)
    RR, ZZ = np.meshgrid(gR, gZ)
    pts = np.stack([RR.ravel(), ZZ.ravel()], 1)
    _, ho = split_stems(DATA_DIR, 0.10, 0, 0)
    stems = ho[: args.n_shots]
    dom = m.domain
    t0 = time.time()
    recs = []
    for s in stems:
        d = np.load(D3D_CACHE["train"] / f"{s}.npz")
        psi_st = d["psi"].astype(np.float64)
        phys = m.from_stored(psi_st)
        J = -gs_op(phys, gR, m.dR, m.dZ) / (MU0 * RR)
        T = len(d["ip"])
        Jc = np.zeros((T, len(m.cell_iz)))
        for k in range(T):
            lr, lz = d["lcfs_r"][k], d["lcfs_z"][k]
            ok = np.isfinite(lr) & np.isfinite(lz)
            if ok.sum() < 10:
                continue
            inside = MPath(np.stack([lr[ok], lz[ok]], 1)).contains_points(pts).reshape(RR.shape)
            Jk = np.where(inside & dom, J[k], 0.0)
            Jc[k] = Jk[m.cell_iz, m.cell_ir]
        recs.append(dict(stem=s, psi=psi_st, I=d["I"].astype(np.float64), ip=d["ip"].astype(np.float64), Jc=Jc,
                         q95=d["q95"], betaN=d["betaN"], r_axis=d["r_axis"], z_axis=d["z_axis"]))
    print(f"{len(stems)} shots loaded in {time.time() - t0:.0f}s", flush=True)
    M = m.M.astype(np.float64)                                  # (C, P)
    G = np.asarray(m.G_cells, np.float64)                       # (K, P)
    Iall = np.concatenate([r["I"] for r in recs]); Jall = np.concatenate([r["Jc"] for r in recs])
    tgt = np.concatenate([m.from_stored(r["psi"]).reshape(len(r["ip"]), -1) for r in recs])
    ipall = np.concatenate([r["ip"] for r in recs])
    base = Iall @ M + (Jall * (m.dR * m.dZ)) @ G
    res = tgt - base
    Iin = Jall.sum(1) * m.dR * m.dZ
    print(f"J_true closes to Ip: median ratio {np.median(Iin / ipall):.4f}", flush=True)
    domf = dom.ravel()
    # global per-coil gains: res ~ sum_c dg_c I_c M_c (+ per-frame offset), fitted on domain points
    sub = np.arange(0, len(tgt), 7)
    A = (Iall[sub, :, None] * M[None, :, domf]).transpose(0, 2, 1).reshape(-1, M.shape[0])
    A = np.concatenate([A, np.ones((A.shape[0], 1))], 1)
    lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
    reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
    dg = np.linalg.solve(A.T @ A + reg, A.T @ res[sub][:, domf].ravel())[:-1]
    print("per-coil gain corrections:", np.round(dg, 4).tolist(), flush=True)
    variants = {"exact": base, "gain": base + (Iall * dg[None]) @ M}
    # per-frame harmonic fit on domain cells (axis of the EFIT equilibrium as the expansion centre)
    rax = np.concatenate([r["r_axis"] for r in recs]); zax = np.concatenate([r["z_axis"] for r in recs])
    for name in ("exact", "gain"):
        out = variants[name].copy()
        r = tgt - out
        for i in range(len(out)):
            H = harmonics(RR, ZZ, rax[i] if np.isfinite(rax[i]) else 1.7, zax[i] if np.isfinite(zax[i]) else 0.0).reshape(7, -1)
            c, *_ = np.linalg.lstsq(H[:, domf].T, r[i, domf], rcond=None)
            out[i] += c @ H
        variants[name.replace("exact", "") + "harm"] = out
    metrics = {}
    vt = ((tgt - tgt.mean()) ** 2).sum()
    for name, ps in variants.items():
        e = ps - tgt
        metrics[name] = {"r2_psi_grid": float(1 - (e ** 2).sum() / vt),
                         "rms_dom": float(np.sqrt((e[:, domf] ** 2).mean())),
                         "rms_all": float(np.sqrt((e ** 2).mean()))}
        print(name, metrics[name], flush=True)
        od = args.out / name / "preds"
        od.mkdir(parents=True, exist_ok=True)
        o = 0
        for r in recs:
            T = len(r["ip"])
            st = m.to_stored(ps[o:o + T]).reshape(T, len(gZ), len(gR)).astype(np.float32)
            np.savez(od / f"{r['stem']}.npz", psi=st, q95=r["q95"].astype(np.float32), betaN=r["betaN"].astype(np.float32))
            o += T
        (args.out / name / "stems.txt").write_text("\n".join(stems))
    (args.out / "metrics.json").write_text(json.dumps({"metrics": metrics, "dg": dg.tolist()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
