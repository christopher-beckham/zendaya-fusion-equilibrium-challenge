"""Frozen copy of xg/score_generic.py (2026-10-05 21:55 UTC).

Composite score S on any labelled equilibrium set (synthetic MAST, SPARC, ...) with the official scorer modules.

    python score_generic.py --truth $R1_ROOT/cache/mast/mast_synthetic_truth.npz --preds runs/xg_synth/preds

Each equilibrium is one single-frame shot. The scorer's functions (local_score.build_reference / score_shot,
metrics.finalize_machine) run unchanged in the "MAST" orientation (axis = flux maximum, as these sets are stored).
The envelope mask is built the way the scorer builds its own: the union of the set's ground-truth LCFS polygons on
the grid, dilated by 2 cells (no machine's EFIT data enters). Prints S and its breakdown and writes a JSON next to
the predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fusion_scoring"))
import local_score as LS                                 # noqa: E402
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from metrics import Accum, finalize_machine              # noqa: E402

LS.MACHINE = "MAST"
# ch2/syn: record the scorer's own derived values of every predicted frame (derive_frame is called once per frame that needs
# Consistency terms) -> per-term bias / rms of pred - truth, in the scorer's units (metres for R_axis / Z_axis)
_REC = []
_derive = LS.derive_frame


def _derive_rec(*a, **k):
    v = _derive(*a, **k)
    _REC.append(v)
    return v


LS.derive_frame = _derive_rec


def envelope(lcfs_r, lcfs_z, R, Z, dilate=2):
    RR, ZZ = np.meshgrid(R, Z); pts = np.stack([RR.ravel(), ZZ.ravel()], 1)
    m = np.zeros(RR.size, bool)
    for r, z in zip(lcfs_r, lcfs_z):
        ok = np.isfinite(r) & np.isfinite(z)
        if ok.sum() >= 10:
            m |= MPath(np.stack([r[ok], z[ok]], 1)).contains_points(pts)
    return binary_dilation(m.reshape(RR.shape), iterations=dilate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--select", type=Path, default=None, help="optional text file of stems to score")
    ap.add_argument("--json-dir", type=Path, default=None, help="where the score json goes (default: next to the predictions)")
    args = ap.parse_args()
    T = np.load(args.truth, allow_pickle=True)
    stems = [str(s) for s in T["stems"]]
    R, Z = T["grid_R"].astype(np.float64), T["grid_Z"].astype(np.float64)
    keep = np.arange(len(stems))
    if args.select is not None:
        sel = set(args.select.read_text().split())
        keep = np.array([i for i, s in enumerate(stems) if s in sel])
    mask = envelope(T["lcfs_r"][keep], T["lcfs_z"][keep], R, Z)
    mask_f = mask.astype(np.float64)
    shots, preds = [], []
    gmean = float(np.nanmean(T["psi"][keep]))          # cells with no truth: equal constant in truth and prediction,
    n_missing = 0                                       # so they add nothing to either sum of squares
    for i in keep:
        g = T["psi"][i].astype(np.float64)
        miss = ~np.isfinite(g); n_missing += int(miss.sum())
        shots.append({"psi": np.where(miss, gmean, g)[None], "q95": np.array([T["q95"][i]]), "betaN": np.array([T["betaN"][i]])})
        p = np.load(args.preds / f"{stems[i]}.npz")
        pp = p["psi"].astype(np.float64)[:1]
        pp[:, miss] = gmean
        preds.append({"psirz": pp, "q95": p["q95"][:1].astype(np.float64), "betaN": p["betaN"][:1].astype(np.float64)})
    if n_missing:
        print(f"note: {n_missing} truth cells are undefined (outside the source grid) and excluded from R2_psi", flush=True)
    refs = [LS.build_reference(s["psi"], R, Z, mask, mask_f) for s in shots]
    psi_all = np.concatenate([s["psi"].ravel() for s in shots])
    ref_stats = {"psi_sum": float(psi_all.sum()), "psi_sumsq": float((psi_all ** 2).sum()), "psi_n": float(psi_all.size)}
    sc = np.zeros(N_SCALARS); sq = np.zeros(N_SCALARS); sn = np.zeros(N_SCALARS)
    for j, nme in enumerate(SCALARS):
        v = np.concatenate([s[nme] for s in shots]).astype(np.float64); v = v[np.isfinite(v)]
        sc[j], sq[j], sn[j] = v.sum(), (v ** 2).sum(), v.size
    cs = np.zeros(N_CONS); cq = np.zeros(N_CONS); cn = np.zeros(N_CONS)
    for ref in refs:
        cons, cm = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons[cm[:, j], j]; cs[j] += v.sum(); cq[j] += (v ** 2).sum(); cn[j] += v.size
    ref_stats.update(scal_sum=sc, scal_sumsq=sq, scal_n=sn, cons_sum=cs, cons_sumsq=cq, cons_n=cn)
    means = (ref_stats["psi_sum"] / ref_stats["psi_n"], np.where(sn > 0, sc / np.maximum(sn, 1), 0.0),
             np.where(cn > 0, cs / np.maximum(cn, 1), 0.0))
    totals = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = LS.psi_residuals(s_["psi"], p["psirz"], means[0])
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    acc = Accum(); acc.psi_sign = psi_sign
    err = []                                    # (n_frames, N_CONS) pred - truth (NaN where undefined)
    for s_, ref, p in zip(shots, refs, preds):
        n0 = len(_REC)
        acc.add(LS.score_shot(s_, ref, p, R, Z, mask, mask_f, psi_sign, means))
        if len(_REC) > n0:
            vals, cg, cm = _REC[-1], ref[1][0], ref[2][0]
            err.append([(float(vals[c]) - float(cg[j])) if (cm[j] and np.isfinite(vals[c])) else np.nan for j, c in enumerate(CONS_SCALARS)])
    res = finalize_machine(acc, ref_stats)
    name = args.name or args.preds.parent.name
    print("=" * 62)
    print(f"  {name}:  COMPOSITE S = {res['S']:.4f}   ({len(shots)} equilibria, own-LCFS envelope mask)")
    print("=" * 62)
    for label, key, w in [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15), ("1 - D_LCFS", "dlcfs", 0.10),
                          ("Consistency", "consistency", 0.20)]:
        v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    each = res["r2_cons_each"]
    print("  Consistency: " + "  ".join(f"{k}={'n/a' if v is None else f'{v:.3f}'}" for k, v in each.items()))
    for k in ("r2_scal_each", "r2_scalars_each", "r2_each"):
        if k in res:
            print(f"  q95 / betaN: {res[k]}")
    if err:
        e = np.array(err, float)
        print("  pred-truth bias / rms: " + "  ".join(f"{c} {np.nanmean(e[:, j]):+.4f}/{np.sqrt(np.nanmean(e[:, j] ** 2)):.4f}" for j, c in enumerate(CONS_SCALARS)))
        res["cons_bias"] = {c: float(np.nanmean(e[:, j])) for j, c in enumerate(CONS_SCALARS)}
        res["cons_rms"] = {c: float(np.sqrt(np.nanmean(e[:, j] ** 2))) for j, c in enumerate(CONS_SCALARS)}
    print(f"  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames; derivations on {res['cons_fail_frac']:.1%}; psi_sign {int(psi_sign):+d}")
    out = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in res.items()}
    (args.json_dir or args.preds.parent).mkdir(parents=True, exist_ok=True)
    ((args.json_dir or args.preds.parent) / f"score_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
