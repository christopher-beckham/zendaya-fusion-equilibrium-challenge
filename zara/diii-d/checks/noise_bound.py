"""Worst-case score change between two DIII-D prediction sets of the same shots, with the official scorer code.

    python zara/diii-d/checks/noise_bound.py OUT.json NAME=PKG_DIR:REF_DIR [...]

PKG_DIR / REF_DIR hold per-shot files <stem>.npz (psi (T,65,65), q95 (T,), betaN (T,)). REF is what a submitted file was
built from, PKG what the package regenerated from the shipped weights. The reference set is scored as if it were the ground
truth (local_score.build_reference / score_shot / finalize_machine, the scorer's own modules), which gives how far apart the
two sets are on every term of S: 1 - R2(pkg vs ref) for psi, q95, betaN and the seven consistency functionals, and the
LCFS distance D(pkg, ref).

From those, a bound on how much S(pkg) and S(ref) can differ against the real labels:
  pooled R2 terms: ||a - t|| <= ||b - t|| + ||a - b||, so |dR2| <= 2 sqrt((1 - R2_b) d) + d with d = 1 - R2(a vs b) (the
    reference's own spread stands in for the label spread; R2_b = the Development board value of the same model family);
  D_LCFS: per-frame Hausdorff distances obey the triangle inequality, so |dD| <= D(a, b);
  S = 0.55 R2_psi + 0.15 mean(R2_q95, R2_betaN) + 0.10 (1 - D) + 0.20 mean(R2_cons).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]   # zara/diii-d
sys.path.insert(0, str(PKG / "fusion_scoring"))
sys.path.insert(0, str(PKG))
import local_score as ls  # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, SCALARS, W_CONS, W_LCFS, W_PSI, W_QB  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

# v13 on the Development board (t2b / b2 / r1 detailed results): the error level each bound is taken around
BOARD = {"r2_psi": 0.999937, "q95": 0.99883, "betaN": 0.998112, "R_axis": 0.985245, "Z_axis": 0.994621, "kappa": 0.98525,
         "tri_top": 0.995861, "tri_bot": 0.995272, "volume": 0.993251, "li": 0.97631}
SAFETY = 2.0          # take the label error twice as large as on the board (other recipes, other test shots)

mask = np.load(PKG / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = mask["grid_R"], mask["grid_Z"]
MC = mask["mask_coarse"].astype(bool)
MF = MC.astype(np.float64)


def load(f):
    with np.load(f) as z:
        return {"psi": z["psi"].astype(np.float32), "q95": z["q95"].astype(np.float64), "betaN": z["betaN"].astype(np.float64)}


def ref_job(f):
    s = load(f)
    return s, ls.build_reference(s["psi"], R, Z, MC, MF)


def score_job(args):
    s, ref, pf, sign, means = args
    p = load(pf)
    pred = {"psirz": p["psi"].astype(np.float64), "q95": p["q95"], "betaN": p["betaN"]}
    return ls.score_shot(s, ref, pred, R, Z, MC, MF, sign, means)


def one(name, pkg_dir, ref_dir, workers):
    stems = sorted(f.name for f in Path(pkg_dir).glob("*.npz"))
    assert stems, f"no files in {pkg_dir}"
    ident = sum(Path(pkg_dir, s).read_bytes() == Path(ref_dir, s).read_bytes() for s in stems)
    with Pool(workers) as pool:
        refs = pool.map(ref_job, [Path(ref_dir, s) for s in stems])
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for s, ref in refs:
        g = s["psi"].astype(np.float64); fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, nm in enumerate(SCALARS):
            v = s[nm][np.isfinite(s[nm])]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        for j in range(N_CONS):
            v = ref[1][ref[2][:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n, "scal_sum": scal_sum, "scal_sumsq": scal_sumsq,
                 "scal_n": scal_n, "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    means = (psi_sum / psi_n, np.where(scal_n > 0, scal_sum / np.maximum(scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.maximum(cons_n, 1), 0.0))
    acc = Accum(); acc.psi_sign = 1.0
    with Pool(workers) as pool:
        for part in pool.map(score_job, [(s, ref, Path(pkg_dir, st), 1.0, means) for (s, ref), st in zip(refs, stems)]):
            acc.add(part)
    res = finalize_machine(acc, ref_stats)
    # distances of pkg from ref, per term
    d_psi = float(acc.ss_res_psi[1.0] / (psi_sumsq - psi_sum ** 2 / psi_n))      # unrounded (finalize_machine rounds to 6 digits)
    d_scal = {nm: float(acc.ss_res_scal[j] / (scal_sumsq[j] - scal_sum[j] ** 2 / scal_n[j])) for j, nm in enumerate(SCALARS)}
    d_cons = {}
    for j, nm in enumerate(CONS_SCALARS):
        tot = cons_sumsq[j] - cons_sum[j] ** 2 / cons_n[j] if cons_n[j] else 0.0
        d_cons[nm] = float(acc.ss_res_cons[j] / tot) if tot > 0 else 0.0
    d_lcfs = float(np.mean(acc.shot_dlcfs))
    bound = lambda r2b, d: 2.0 * np.sqrt(SAFETY * (1.0 - r2b) * d) + d
    b_psi = bound(BOARD["r2_psi"], d_psi)
    b_qb = np.mean([bound(BOARD[nm], d_scal[nm]) for nm in SCALARS])
    b_cons = np.mean([bound(BOARD[nm], d_cons[nm]) for nm in CONS_SCALARS])
    b_S = W_PSI * b_psi + W_QB * b_qb + W_LCFS * d_lcfs + W_CONS * b_cons
    out = {"shots": len(stems), "frames": int(sum(s["psi"].shape[0] for s, _ in refs)), "byte_identical_shots": int(ident),
           "S_pkg_vs_ref": res["S"], "dist": {"psi": d_psi, **d_scal, **d_cons, "D_lcfs": d_lcfs},
           "max_abs": {k: float(max(np.max(np.abs(load(Path(pkg_dir, st))[k] - load(Path(ref_dir, st))[k])) for st in stems))
                       for k in ("psi", "q95", "betaN")},
           "bound_dS": {"psi_term": W_PSI * b_psi, "qb_term": W_QB * b_qb, "lcfs_term": W_LCFS * d_lcfs,
                        "cons_term": W_CONS * b_cons, "total": b_S}}
    print(f"[{name}] {json.dumps(out)}", flush=True)
    return out


if __name__ == "__main__":
    out_json, pairs = Path(sys.argv[1]), sys.argv[2:]
    workers = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "4"))
    res = {}
    for p in pairs:
        name, rest = p.split("=", 1)
        pkg_dir, ref_dir = rest.split(":")
        res[name] = one(name, pkg_dir, ref_dir, workers)
        out_json.write_text(json.dumps(res, indent=1))
