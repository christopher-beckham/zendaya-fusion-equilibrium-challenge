"""Pair statistics of NEW board-scored MAST files against the existing pool (fpb.FILES + b2), label-free.

Same quantities as team_blend/pairs2.py, but only for pairs that involve a new file, so a new member can join the blend
search without recomputing the whole pool: per-frame q95, betaN, the seven derived scalars of each new file; pooled
sum over grid points of (psi_a - psi_b)^2 and per-frame LCFS Hausdorff / rgeo for (new, old) and (new, new) pairs.

usage: python add_pairs.py --out <dir> [--workers 92] NAME=path.npz[@-1] [NAME=path.npz[@-1] ...]
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

import fpb

_C = {}


def _init(new, old, R, Z, mask):
    _C.update(new=new, old=old, R=R, Z=Z, mask=mask, maskf=mask.astype(np.float64), npz={})
    sys.path.insert(0, fpb.STARTER)
    sys.path.insert(0, os.path.join(fpb.STARTER, "fusion_scoring"))
    import local_score as ls
    ls.MACHINE = "MAST"
    _C["ls"] = ls


def _load(name, path, sign, key):
    if name not in _C["npz"]:
        _C["npz"][name] = np.load(path)
    return _C["npz"][name][key]


def _one(i):
    ls = _C["ls"]
    k = f"shot_{i:04d}"
    psi, refs, out = {}, {}, {"file": {}, "pair": {}}
    for name, (path, sign) in list(_C["new"].items()) + list(_C["old"].items()):
        psi[name] = sign * _load(name, path, sign, k + "_psirz").astype(np.float64)
        refs[name] = ls.build_reference(psi[name], _C["R"], _C["Z"], _C["mask"], _C["maskf"])
    for name, (path, sign) in _C["new"].items():
        out["file"][name] = dict(q95=_load(name, path, sign, k + "_q95").astype(np.float64),
                                 betaN=_load(name, path, sign, k + "_betaN").astype(np.float64), cons=refs[name][1])
    pairs = [(a, b) for a in _C["new"] for b in _C["old"]] + list(itertools.combinations(list(_C["new"]), 2))
    for a, b in pairs:
        d = psi[a] - psi[b]
        rg = np.nanmean([refs[a][3], refs[b][3]])
        hs = []
        for ea, eb in zip(refs[a][0], refs[b][0]):
            hs.append(np.nan if ea is None or eb is None else
                      min(1.0, ls.symmetric_hausdorff(np.asarray(ea, float), np.asarray(eb, float)) / rg))
        out["pair"][f"{a}|{b}"] = dict(psi_d=float((d * d).sum()), haus=np.asarray(hs))
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--shots", type=int, default=1206)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    new = {}
    for x in a.files:
        n, p = x.split("=", 1)
        s = 1.0
        if p.endswith("@-1"):
            p, s = p[:-3], -1.0
        new[n] = (p, s)
    old = {n: v for n, v in fpb.FILES.items() if n not in new}
    old["b2"] = (fpb.BLENDS["b2"], 1.0)
    bas = np.load("/fsx/zara.ercan/cache/mast/mast_public_test/basis.npz")
    R, Z = bas["grid_R"].astype(float), bas["grid_Z"].astype(float)
    mask = fpb.limiter_mask(R, Z)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(new, old, R, Z, mask)) as pool:
        for n, (i, o) in enumerate(pool.imap_unordered(_one, range(a.shots), chunksize=1)):
            res[i] = o
            if n % 200 == 0:
                print(f"{n}/{a.shots} shots", flush=True)
    order = sorted(res)
    arr = {}
    for nm in new:
        for key in ("q95", "betaN", "cons"):
            arr[f"{nm}__{key}"] = np.concatenate([res[i]["file"][nm][key] for i in order])
    pd = {}
    for pk in res[order[0]]["pair"]:
        arr[f"{pk.replace('|', '__vs__')}__haus"] = np.concatenate([res[i]["pair"][pk]["haus"] for i in order])
        pd[pk] = float(sum(res[i]["pair"][pk]["psi_d"] for i in order))
    np.savez_compressed(os.path.join(a.out, "frames.npz"), **arr)
    json.dump({"files": {n: f"{p}@{s:+.0f}" for n, (p, s) in new.items()}, "psi_pair_d": pd},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
