"""Label-free per-frame scorer functionals and LCFS extremes of one or more prediction dirs (holdout or public test).

    python mfunc.py --preds DIR [DIR ...] --out FILE.npz

FILE.npz: stems, shot, k (frame order: sorted stems, frames in order), cons (D, N, 7) scorer functionals (li derived always;
NaN where the scorer would NaN them), li_raw (D, N) (li before the plausibility cut), geo (D, N, 6) LCFS extremes
(Zmax, Zmin, Rmax, Rmin, R at Zmax, R at Zmin) of the scorer's contour; D = number of dirs, in the given order.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def one(path):
    import liguard as L
    from common import AXIS_SIGN, CONS_SCALARS
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    from fanat import geo
    psi = np.load(path)["psi"].astype(np.float64)
    T = len(psi)
    cv = np.full((T, 7), np.nan)
    gv = np.full((T, 6), np.nan)
    li_raw = np.full(T, np.nan)
    s = AXIS_SIGN["DIII-D"]
    for k in range(T):
        ex, _ = extract_lcfs_with_sign(psi[k], L.R, L.Z, "DIII-D", L.MC, L.MF, n_iter=22, axis_sign=s)
        v = derive_frame(psi[k], L.R, L.Z, "DIII-D", L.MC, L.MF, contour=ex, with_li=True, axis_sign=s)
        cv[k] = [v[c] for c in CONS_SCALARS]
        if ex is not None:
            li_raw[k] = L.raw_li(psi[k], ex)
            gv[k] = geo(ex)
    return cv, li_raw, gv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", type=Path, nargs="*", default=None,
                    help="instead of computing: concatenate these mfunc files (same stems) along the dir axis into --out")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.merge:
        zs = [np.load(f) for f in a.merge]
        for z in zs[1:]:
            assert (z["stems"] == zs[0]["stems"]).all()
        np.savez(a.out, stems=zs[0]["stems"], shot=zs[0]["shot"], k=zs[0]["k"],
                 cons=np.concatenate([z["cons"] for z in zs]), li_raw=np.concatenate([z["li_raw"] for z in zs]),
                 geo=np.concatenate([z["geo"] for z in zs]), dirs=np.concatenate([z["dirs"] for z in zs]))
        print(f"merged {len(zs)} files -> {a.out}")
        return
    stems = sorted(p.stem for p in a.preds[0].glob("*.npz"))
    C, LR, G = [], [], []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for d in a.preds:
            t0 = time.time()
            res = list(pool.map(one, [str(d / f"{s}.npz") for s in stems], chunksize=2))
            C.append(np.concatenate([r[0] for r in res])); LR.append(np.concatenate([r[1] for r in res]))
            G.append(np.concatenate([r[2] for r in res]))
            print(f"{d}: {time.time() - t0:.0f}s", flush=True)
            T = np.array([len(r[1]) for r in res])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, stems=np.array(stems), shot=np.repeat(np.arange(len(stems)), T),
             k=np.concatenate([np.arange(t) for t in T]), cons=np.stack(C), li_raw=np.stack(LR), geo=np.stack(G),
             dirs=np.array([str(d) for d in a.preds]))
    print(f"{len(stems)} shots, {T.sum():,} frames -> {a.out}")


if __name__ == "__main__":
    main()
