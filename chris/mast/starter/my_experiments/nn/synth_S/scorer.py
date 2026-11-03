#!/usr/bin/env python3
"""Ground truth and scoring for synthetic held-out MAST, with the official scorer's own functions.

Ground truth comes from the synthetic held-out parquet files in challenge format (efit_psirz, efit_q95, efit_beta_n;
one file = one "shot"), exactly as local_score.load_shots reads DIII-D training shots. The reference targets
(per-frame LCFS, the seven derived scalars, rgeo) are built by local_score.build_reference, ψ residuals and per-shot
parts by local_score.psi_residuals / score_shot, the machine row by metrics.finalize_machine: all with
local_score.MACHINE = "MAST" and the MAST envelope mask (fusion_scoring/masks/mast_envelope.npz), as the platform
scores MAST and as synth_study/score_folder.py does. The flux sign is chosen once per fold from the summed ψ
residuals under both signs (PSI_SIGNS), then imposed on every orientation-sensitive term, as in local_score.main.

    python scorer.py truth g4_5000            # build and cache the reference (SCR/truth/g4_5000.pkl)
    python scorer.py perfect g4_5000          # sanity: true maps and true scalars through the same path
    python scorer.py score g4_5000 PRED.npz   # PRED: psi (N,4225) in set-row order, q95 (N,), betaN (N,)

This process never imports model code or mast_region, so the envelope is used only inside the scorer's derivations.
"""
from __future__ import annotations

import json
import os
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CACHE, SCR, SETS, STARTER, V1_TAG  # noqa: E402

sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
import local_score as ls  # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

ls.MACHINE = "MAST"          # local_score's functions read this module global (as score_folder.py sets it)
_M = np.load(STARTER / "fusion_scoring" / "masks" / "mast_envelope.npz")
MASK_COARSE = _M["mask_coarse"].astype(bool)
MASK_F = MASK_COARSE.astype(np.float64)
WORKERS = int(os.environ.get("SYNTHS_WORKERS", "28"))
# SYNTHS_FAST=1 (train_mm.py --synthS-every jobs): the full fold only, no q95-capped rescore, and the flux-sign probe on
# the first 64 shots as fusion_scoring's local_score.main does; S of the full fold is the selection metric.
FAST = os.environ.get("SYNTHS_FAST", "0") == "1"
PROBE = 64
_G: dict = {}


def _load_file(f: Path) -> dict:
    row = pq.read_table(f, columns=["efit_times", "efit_psirz", "efit_q95", "efit_beta_n", "efit_grid_R",
                                    "efit_grid_Z", "synth_meta"]).to_pylist()[0]
    psi = np.stack([np.asarray(fr, np.float32) for fr in row["efit_psirz"]])
    if psi.ndim == 2:
        psi = psi.reshape(-1, 65, 65)
    meta = json.loads(row["synth_meta"])
    return {"stem": f.stem, "case": f"heldout_{f.stem}", "psi": psi,
            "q95": np.asarray(row["efit_q95"], float), "betaN": np.asarray(row["efit_beta_n"], float),
            "R": np.asarray(row["efit_grid_R"], float).ravel(), "Z": np.asarray(row["efit_grid_Z"], float).ravel(),
            "origin": str(meta["case"]["file"]).replace(".parquet", "")}


def _ref(s):
    return ls.build_reference(s["psi"], _G["R"], _G["Z"], MASK_COARSE, MASK_F)


def build_truth(name: str) -> dict:
    files = sorted(SETS[name]["parquet"].glob("mast_synth*.parquet"))
    with mp.get_context("fork").Pool(WORKERS) as pool:
        shots = pool.map(_load_file, files, chunksize=4)
    R, Z = shots[0]["R"], shots[0]["Z"]
    for s in shots:
        if not (np.allclose(s["R"], R) and np.allclose(s["Z"], Z)):
            raise SystemExit(f"{s['stem']}: grid differs")
    if not (np.allclose(R, _M["grid_R"], atol=1e-5) and np.allclose(Z, _M["grid_Z"], atol=1e-5)):
        raise SystemExit("synthetic grid differs from the scorer's MAST grid")
    _G["R"], _G["Z"] = R, Z
    t0 = time.time()
    with mp.get_context("fork").Pool(WORKERS) as pool:
        refs = pool.map(_ref, shots, chunksize=2)
    print(f"{name}: {len(shots)} shots, {sum(len(s['psi']) for s in shots)} frames, references in "
          f"{time.time() - t0:.0f} s", flush=True)
    v1 = np.load(CACHE / "synth" / V1_TAG / "arrays.npz")
    v1_train = set(v1["origin"][v1["split"] == "train"].tolist())
    for s in shots:
        s.pop("R"); s.pop("Z")
        s["v1_origin"] = s["origin"] in v1_train
    truth = {"name": name, "R": R, "Z": Z, "shots": shots, "refs": refs}
    out = SCR / "truth" / f"{name}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(truth, fh)
    print(f"  {sum(s['v1_origin'] for s in shots)} shots from origins in {V1_TAG} training; wrote {out}", flush=True)
    return truth


def load_truth(name: str) -> dict:
    p = SCR / "truth" / f"{name}.pkl"
    if not p.exists():
        return build_truth(name)
    with open(p, "rb") as fh:
        return pickle.load(fh)


def ref_stats_means(shots, refs):
    """As local_score.main: the pooled sums for SS_tot and the means used for missing / failed values."""
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for s, ref in zip(shots, refs):
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))
    return ref_stats, means


def _score_one(i):
    s, ref, p = _G["shots"][i], _G["refs"][i], _G["preds"][i]
    return ls.score_shot(s, ref, p, _G["R"], _G["Z"], MASK_COARSE, MASK_F, _G["psi_sign"], _G["means"])


def score_fold(truth: dict, preds: list[dict], keep: np.ndarray | None = None, workers: int = WORKERS) -> dict:
    """One fold (all shots, or the shots where `keep`): local_score.main's sign probe, score_shot, finalize_machine."""
    idx = np.arange(len(truth["shots"])) if keep is None else np.nonzero(keep)[0]
    shots = [truth["shots"][i] for i in idx]
    refs = [truth["refs"][i] for i in idx]
    preds = [preds[i] for i in idx]
    ref_stats, means = ref_stats_means(shots, refs)
    totals = {sg: 0.0 for sg in PSI_SIGNS}
    for s_, p in list(zip(shots, preds))[:PROBE if FAST else None]:
        rr, _, _ = ls.psi_residuals(s_["psi"], p["psirz"], means[0])
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    _G.update(shots=shots, refs=refs, preds=preds, R=truth["R"], Z=truth["Z"], psi_sign=psi_sign, means=means)
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_score_one, range(len(shots)), chunksize=1)
    acc = Accum()
    acc.psi_sign = psi_sign
    for part in parts:
        acc.add(part)
    res = finalize_machine(acc, ref_stats)
    res["n_frames"] = int(sum(len(s["psi"]) for s in shots))
    return res


def preds_from_rows(truth: dict, z: dict, case: np.ndarray, frame: np.ndarray, q95=None) -> list[dict]:
    """Set-row arrays -> per-shot prediction dicts in the truth's shot order (local_score.predict's layout)."""
    pos = {(c, int(f)): i for i, (c, f) in enumerate(zip(case.tolist(), frame.tolist()))}
    out = []
    for s in truth["shots"]:
        rows = np.array([pos[(s["case"], k)] for k in range(len(s["psi"]))])
        out.append({"psirz": np.asarray(z["psi"][rows], np.float64).reshape(-1, 65, 65),
                    "q95": np.asarray((z["q95"] if q95 is None else q95)[rows], np.float64),
                    "betaN": np.asarray(z["betaN"][rows], np.float64)})
    return out


def flat(res: dict, prefix: str = "") -> dict:
    out = {f"{prefix}S": res["S"], f"{prefix}r2_psi": res["r2_psi"], f"{prefix}r2_q95": res["r2_each"]["q95"],
           f"{prefix}r2_betaN": res["r2_each"]["betaN"], f"{prefix}r2_qb": res["r2_qb"], f"{prefix}dlcfs": res["dlcfs"],
           f"{prefix}consistency": res["consistency"]}
    for k, v in res["r2_cons_each"].items():
        out[f"{prefix}r2c_{k}"] = v
    out.update({f"{prefix}psi_sign": res["psi_sign"], f"{prefix}lcfs_fail_frac": res["lcfs_fail_frac"],
                f"{prefix}cons_fail_frac": res["cons_fail_frac"], f"{prefix}n_shots": res["n_shots"],
                f"{prefix}n_frames": res["n_frames"]})
    return out


def score_all(truth: dict, preds: list[dict]) -> dict:
    """Full fold and the fold without shots whose origin is in zara_v1_mast's training split."""
    v1 = np.array([s["v1_origin"] for s in truth["shots"]])
    full = score_fold(truth, preds)
    if FAST:
        return {"all": full, "noV1": full}
    return {"all": full, "noV1": score_fold(truth, preds, keep=~v1)}


def main() -> int:
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "truth":
        build_truth(name)
        return 0
    truth = load_truth(name)
    t0 = time.time()
    if mode == "perfect":
        preds = [{"psirz": s["psi"].astype(np.float64), "q95": s["q95"], "betaN": s["betaN"]} for s in truth["shots"]]
        # the same path with our sign convention (ψ × −1), as model predictions arrive
        preds_neg = [{"psirz": -s["psi"].astype(np.float64), "q95": s["q95"], "betaN": s["betaN"]}
                     for s in truth["shots"]]
        out = {"file_sign": score_all(truth, preds), "our_sign": score_all(truth, preds_neg)}
    elif mode == "score":
        z = np.load(sys.argv[3])
        case, frame = z["case"], z["frame"]
        preds = preds_from_rows(truth, z, case, frame)
        out = {"as_is": score_all(truth, preds)}
        if len(sys.argv) > 4 and sys.argv[4] == "--q95cap" and not FAST:
            cap = float(sys.argv[5])
            out["q95cap"] = score_all(truth, preds_from_rows(truth, z, case, frame, q95=np.minimum(z["q95"], cap)))
    else:
        raise SystemExit(f"unknown mode {mode}")
    out["seconds"] = time.time() - t0
    js = json.dumps(out, default=float)
    print("RESULT " + js, flush=True)
    for k, v in out.items():
        if isinstance(v, dict):
            for kk, r in v.items():
                print(f"{k}/{kk}: S {r['S']:.4f} R2psi {r['r2_psi']:.4f} R2qb {r['r2_qb']:.4f} "
                      f"D_LCFS {r['dlcfs']:.4f} cons {r['consistency']:.4f} sign {r['psi_sign']:+d} "
                      f"{r['r2_cons_each']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
