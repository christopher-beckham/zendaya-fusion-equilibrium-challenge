"""Label-free pairwise statistics between MAST submission files (blending board-scored candidates).

For every file: per-frame q95, betaN and the scorer's seven derived scalars (local_score.build_reference = extract_lcfs +
derive_frame, MAST axis sign, hand-drawn limiter mask, grid from the input basis), and the pooled psi sum / sum of squares.
For every pair: pooled sum over all grid points of (psi_a - psi_b)^2 (the scorer pools R2psi over the whole grid) and the
mean symmetric Hausdorff distance between the two LCFS / rgeo (D_LCFS-like).

Why: for any metric that is a pooled R^2 of a submitted quantity, the board R^2 of the blend w a + (1 - w) b is exactly
    w R2_a + (1 - w) R2_b + w (1 - w) D_ab / SS_y,
with D_ab = sum (a - b)^2 computed here and SS_y the label sum of squares about its mean (approximated by the files' own
spread). psi, q95 and betaN follow this exactly; the derived scalars of a blended map follow it to first order.
No MAST labels are read.

usage: python diag/board_pairs.py --out <dir> [--workers 64] [--shots N] name=path.npz [name=path.npz ...]
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

STARTER = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter"
sys.path.insert(0, STARTER)
sys.path.insert(0, os.path.join(STARTER, "fusion_scoring"))
import local_score as ls  # noqa: E402
from common import CONS_SCALARS  # noqa: E402

ls.MACHINE = "MAST"
FILES = []
SIGN = {}
_Z = {}
_CTX = {}


def limiter_mask(R, Z):
    sys.path.insert(0, "/fsx/zara.ercan/more_data/generated/code")
    import mast_machine as mm
    from matplotlib.path import Path as MPath
    RR, ZZ = np.meshgrid(R, Z)
    return MPath(mm.limiter_polygon()).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)


def load(name, path, key):
    if name not in _Z:
        _Z[name] = np.load(path)
    return _Z[name][key]


def one(i):
    c = _CTX
    k = f"shot_{i:04d}"
    psi, q, b, refs = {}, {}, {}, {}
    for name, path in FILES:
        psi[name] = SIGN.get(name, 1.0) * load(name, path, k + "_psirz").astype(np.float64)
        q[name] = load(name, path, k + "_q95").astype(np.float64)
        b[name] = load(name, path, k + "_betaN").astype(np.float64)
        refs[name] = ls.build_reference(psi[name], c["R"], c["Z"], c["mask"], c["mask_f"])
    out = {"T": psi[FILES[0][0]].shape[0], "file": {}, "pair": {}}
    for name, _ in FILES:
        p = psi[name]
        out["file"][name] = dict(psi_sum=float(p.sum()), psi_sumsq=float((p * p).sum()), psi_n=int(p.size),
                                 q95=q[name], betaN=b[name], cons=refs[name][1], rgeo=refs[name][3])
    for (na, _), (nb, _) in itertools.combinations(FILES, 2):
        d = psi[na] - psi[nb]
        ca, cb = refs[na][0], refs[nb][0]
        rg = np.nanmean([refs[na][3], refs[nb][3]])
        hs = []
        for ea, eb in zip(ca, cb):
            if ea is None or eb is None:
                hs.append(np.nan)
            else:
                hs.append(min(1.0, ls.symmetric_hausdorff(np.asarray(ea, float), np.asarray(eb, float)) / rg))
        out["pair"][f"{na}|{nb}"] = dict(psi_d=float((d * d).sum()), haus=np.asarray(hs))
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--shots", type=int, default=1206)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for x in a.files:
        n, p = x.split("=", 1)
        if p.endswith("@-1"):
            p = p[:-3]
            SIGN[n] = -1.0
        FILES.append((n, p))
    bas = np.load("/fsx/zara.ercan/cache/mast/mast_public_test/basis.npz")
    R, Z = bas["grid_R"].astype(float), bas["grid_Z"].astype(float)
    m = limiter_mask(R, Z)
    _CTX.update(R=R, Z=Z, mask=m, mask_f=m.astype(np.float64))
    os.makedirs(a.out, exist_ok=True)
    res = {}
    with get_context("fork").Pool(a.workers) as pool:
        for n, (i, o) in enumerate(pool.imap_unordered(one, range(a.shots), chunksize=2)):
            res[i] = o
            if n % 100 == 0:
                print(f"{n}/{a.shots} shots", flush=True)
    order = sorted(res)
    names = [n for n, _ in FILES]
    arr = {"shot": np.concatenate([np.full(res[i]["T"], i) for i in order])}
    tot = {}
    for nm in names:
        for key in ("q95", "betaN"):
            arr[f"{nm}__{key}"] = np.concatenate([res[i]["file"][nm][key] for i in order])
        arr[f"{nm}__cons"] = np.concatenate([res[i]["file"][nm]["cons"] for i in order])
        tot[nm] = {k: float(sum(res[i]["file"][nm][k] for i in order)) for k in ("psi_sum", "psi_sumsq", "psi_n")}
    pairs = {}
    for pk in res[order[0]]["pair"]:
        arr[f"{pk.replace('|', '__vs__')}__haus"] = np.concatenate([res[i]["pair"][pk]["haus"] for i in order])
        pairs[pk] = float(sum(res[i]["pair"][pk]["psi_d"] for i in order))
    np.savez_compressed(os.path.join(a.out, "frames.npz"), **arr)
    json.dump({"files": dict(FILES), "cons_names": list(CONS_SCALARS), "psi": tot, "psi_pair_d": pairs},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
