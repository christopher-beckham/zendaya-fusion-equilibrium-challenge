"""Boundary targets from the TRUE maps of every official training shot (labels for the boundary model; diagnostic columns too).

Per frame: psi_ax (spline value at the scorer O-point), psi_sep (mean spline psi on the scorer-extracted LCFS = last closed surface),
psi_efit (mean spline psi on the EFIT LCFS of the cache), x_b = (psi_efit - psi_ax) / (psi_sep - psi_ax) (NaN if undefined),
min spline psi on the inner wall lines R = 1.020 / 1.033 / 1.046 m (|Z| <= 0.3), scorer-LCFS Rmin and its Z.
    python xb_targets.py   -> $FEQ_CACHE/ml/ch1s/xb_all.npz (frames of all 7,041 shots in sorted-stem order)
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s_common import CACHE, OUTC, all_stems

ZL = np.linspace(-0.3, 0.3, 25)
COLS = ["psi_ax", "psi_sep", "psi_efit", "x_b", "wall1020", "wall1033", "wall1046", "Rmin_sep", "Z_Rmin_sep"]


def job(s):
    import physfeat as pf
    from scipy.interpolate import RectBivariateSpline
    d = np.load(CACHE / f"{s}.npz")
    T = len(d["ip"])
    out = np.full((T, len(COLS)), np.nan)
    for k in range(T):
        try:
            psi = d["psi"][k].astype(np.float64)
            sp = RectBivariateSpline(pf.GZ, pf.GR, psi, kx=3, ky=3)
            Ra, Za, iz, ir, _ = pf.find_o_point(-psi, pf.GR, pf.GZ, pf.MASK)
            pa = float(sp.ev(Za, Ra)); out[k, 0] = pa
            C, _ = pf.extract_lcfs_with_sign(psi, pf.GR, pf.GZ, "DIII-D", pf.MASK, pf.MASKF, n_iter=22, axis_sign=-1.0)
            lr, lz = d["lcfs_r"][k], d["lcfs_z"][k]; m = np.isfinite(lr)
            if m.sum() >= 8:
                out[k, 2] = float(np.mean(sp.ev(lz[m], lr[m])))
            for j, Rw in enumerate((1.020, 1.033, 1.046)):
                out[k, 4 + j] = float(np.min(sp.ev(ZL, np.full_like(ZL, Rw))))
            if C is not None:
                ps = float(np.mean(sp.ev(C[:, 1], C[:, 0]))); out[k, 1] = ps
                out[k, 3] = (out[k, 2] - pa) / (ps - pa) if ps > pa else np.nan
                j = int(np.argmin(C[:, 0])); out[k, 7], out[k, 8] = C[j, 0], C[j, 1]
        except Exception:
            pass
    return out


if __name__ == "__main__":
    stems = all_stems()
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(job, stems, chunksize=4)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    A = np.concatenate(res)
    nT = np.array([len(r) for r in res])
    np.savez(OUTC / "xb_all.npz", A=A, cols=np.array(COLS), stems=np.array(stems), nT=nT)
    xb = A[:, 3]
    print(f"{len(A):,} frames in {time.time() - t0:.0f}s; x_b finite {np.isfinite(xb).mean():.4f}; "
          f"x_b < 0.999: {np.mean(xb < 0.999):.4f}; quantiles {np.nanquantile(xb, [0.001, 0.01, 0.02, 0.05, 0.5, 0.99, 0.999])}")
