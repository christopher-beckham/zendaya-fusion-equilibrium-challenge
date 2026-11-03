"""Score several prediction sets of the same labelled DIII-D training shots with the official scorer code, at full precision.

    python score_lab.py OUT.json LABEL_DIR NAME=PRED_DIR [...] [--pair A,B ...]

LABEL_DIR holds the shots' parquet files (official training data, so EFIT labels are present); PRED_DIR holds per-shot
<stem>.npz (psi, q95, betaN), as the package's predict_*.sh write them under pred/.../final. Every set is scored on the same
shots (the stems of LABEL_DIR). --pair A,B reports S(A) - S(B) term by term.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]   # zara/diii-d
sys.path.insert(0, str(PKG / "fusion_scoring"))
sys.path.insert(0, str(PKG))
import local_score as ls  # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, SCALARS, W_CONS, W_LCFS, W_PSI, W_QB  # noqa: E402
from experiments import _as_psirz_stack  # noqa: E402
from metrics import Accum  # noqa: E402

mask = np.load(PKG / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R, Z = mask["grid_R"], mask["grid_Z"]
MC = mask["mask_coarse"].astype(bool)
MF = MC.astype(np.float64)


def ref_job(f):
    row = pd.read_parquet(f, columns=["efit_psirz", "efit_q95", "efit_beta_n"]).iloc[0].to_dict()
    s = {"psi": _as_psirz_stack(row["efit_psirz"]).astype(np.float32), "q95": np.asarray(row["efit_q95"], dtype=np.float64),
         "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64)}
    return s, ls.build_reference(s["psi"], R, Z, MC, MF)


def score_job(args):
    s, ref, pf, sign, means = args
    with np.load(pf) as z:
        pred = {"psirz": z["psi"].astype(np.float64), "q95": z["q95"].astype(np.float64), "betaN": z["betaN"].astype(np.float64)}
    return ls.score_shot(s, ref, pred, R, Z, MC, MF, sign, means)


def tot(sum_, sumsq, n):
    return sumsq - sum_ * sum_ / n if n else 0.0


if __name__ == "__main__":
    args = sys.argv[1:]
    pairs = [args[i + 1].split(",") for i, a in enumerate(args) if a == "--pair"]
    args = [a for i, a in enumerate(args) if a != "--pair" and (i == 0 or args[i - 1] != "--pair")]
    out_json, label_dir, sets = Path(args[0]), Path(args[1]), dict(a.split("=", 1) for a in args[2:])
    files = sorted(label_dir.glob("*.parquet"))
    stems = [f.stem for f in files]
    workers = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "4"))
    with Pool(workers) as pool:
        refs = pool.map(ref_job, files)
    ps, pss, pn = 0.0, 0.0, 0.0
    ss_, sss, sn = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cs, css, cn = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for s, ref in refs:
        g = s["psi"].astype(np.float64); fin = np.isfinite(g)
        ps += g[fin].sum(); pss += (g[fin] ** 2).sum(); pn += fin.sum()
        for j, nm in enumerate(SCALARS):
            v = s[nm][np.isfinite(s[nm])]; ss_[j] += v.sum(); sss[j] += (v ** 2).sum(); sn[j] += v.size
        for j in range(N_CONS):
            v = ref[1][ref[2][:, j], j]; cs[j] += v.sum(); css[j] += (v ** 2).sum(); cn[j] += v.size
    means = (ps / pn, np.where(sn > 0, ss_ / np.maximum(sn, 1), 0.0), np.where(cn > 0, cs / np.maximum(cn, 1), 0.0))
    res = {}
    for name, d in sets.items():
        acc = Accum(); acc.psi_sign = 1.0
        with Pool(workers) as pool:
            for part in pool.map(score_job, [(s, ref, Path(d, st + ".npz"), 1.0, means) for (s, ref), st in zip(refs, stems)]):
                acc.add(part)
        r2_psi = 1.0 - acc.ss_res_psi[1.0] / tot(ps, pss, pn)
        r2_s = {nm: 1.0 - acc.ss_res_scal[j] / tot(ss_[j], sss[j], sn[j]) for j, nm in enumerate(SCALARS)}
        r2_c = {nm: 1.0 - acc.ss_res_cons[j] / tot(cs[j], css[j], cn[j]) for j, nm in enumerate(CONS_SCALARS)}
        cons = float(np.mean([max(0.0, v) for v in r2_c.values()]))
        dl = float(np.mean(acc.shot_dlcfs))
        S = W_PSI * max(0.0, r2_psi) + W_QB * max(0.0, float(np.mean(list(r2_s.values())))) + W_LCFS * (1 - min(1.0, dl)) + W_CONS * cons
        res[name] = {"S": S, "r2_psi": r2_psi, **r2_s, "D_lcfs": dl, "consistency": cons, **r2_c}
        print(f"[{name}] S {S:.8f}  " + "  ".join(f"{k} {v:.7f}" for k, v in res[name].items() if k != "S"), flush=True)
    res["_pairs"] = {}
    for a, b in pairs:
        dd = {k: res[a][k] - res[b][k] for k in res[a]}
        res["_pairs"][f"{a}-{b}"] = dd
        print(f"[{a} - {b}] dS {dd['S']:+.2e}  " + "  ".join(f"{k} {v:+.1e}" for k, v in dd.items() if k != "S"), flush=True)
    res["_shots"] = len(stems); res["_frames"] = int(sum(s["psi"].shape[0] for s, _ in refs))
    out_json.write_text(json.dumps(res, indent=1))
