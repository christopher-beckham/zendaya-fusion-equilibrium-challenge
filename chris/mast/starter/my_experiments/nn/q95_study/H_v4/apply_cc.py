"""Apply the centre-column q95 rule to a built MAST submission. Label-free; no MAST label is read.

    CUDA_VISIBLE_DEVICES= FEC_MAST_MASK=limiter python apply_cc.py SUB_DIR OUT_DIR --factor 0.932 [--workers 8]

Rule (fitted on synthetic held-out sets only, see REPORT_Q95_V4.md): on frames whose map boundary (the scorer's
procedure inside the limiter region) lies within one grid cell of the region edge AND reaches the centre column
(innermost R < 0.205 m), the submitted map q95 is multiplied by FACTOR. Applied only where the submitted q95 equals
the map q95 (not where postproc_qb kept the head). ψ and βN arrays are copied unchanged; the DIII-D file is
hard-linked. Writes OUT_DIR/mast_public_test.npz and OUT_DIR/cc_flags.npz (per-frame flag, old and new q95).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qlib  # noqa: E402,F401
C = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
SSFX = "" if SPLIT == "mast_public_test" else f"_{SPLIT}"   # file-name suffix for caches without the split in their name
MAST = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data" / SPLIT
EDGE_CELLS, R_IN_MAX = 1.0, 0.205
_S = {}


def _init():
    import features_v5 as f5
    ps, m, dist = qlib.setup()
    _S.update(ps=ps, m=m, dist=dist, feat=np.load(C / f"features_v5_{SPLIT}_sol0.5_limiter.npy", mmap_mode="r"),
              fq=f5.FEATURES.index("q_ITER"), ip=np.load(C / f"mast_inputs{SSFX}.npz")["ip"])


def _shot(job):
    i, start, psi, qsub = job
    ps, m = _S["ps"], _S["m"]
    rows = start + np.arange(len(psi))
    bt = ps.bt_from_qiter(np.asarray(_S["feat"][rows])[:, _S["fq"]], _S["ip"][rows], m.r0, m.a)
    flag = np.zeros(len(psi), bool)
    for k in range(len(psi)):
        o = qlib.frame(ps, m, _S["dist"], psi[k], bt[k]) or {}
        qm = o.get("q95", np.nan)
        flag[k] = (o.get("edge_cells", 99) < EDGE_CELLS and o.get("r_in", 9) < R_IN_MAX
                   and np.isfinite(qm) and abs(qm - float(qsub[k])) < 1e-3 * max(1.0, abs(qm)))
    return i, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--factor", type=float, required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import mast_region
    assert mast_region.name() == "limiter", "run with FEC_MAST_MASK=limiter"
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.sub / f"{SPLIT}.npz")
    files = sorted(MAST.glob("*.parquet"))
    index = pd.read_parquet(C / f"features_v5_{SPLIT}_sol0.5.parquet").set_index("filename")
    jobs = [(i, int(index.loc[f.name, "start"]), z[f"shot_{i:04d}_psirz"], z[f"shot_{i:04d}_q95"]) for i, f in enumerate(files)]
    arrays = {k: z[k] for k in z.files}
    flags = {}
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for i, fl in ex.map(_shot, jobs, chunksize=4):
            flags[i] = fl
            q = arrays[f"shot_{i:04d}_q95"]
            arrays[f"shot_{i:04d}_q95"] = np.where(fl, q * a.factor, q).astype(q.dtype)
    np.savez(a.out / f"{SPLIT}.npz", **arrays)
    src, dst = a.sub / "diii_d_public_test.npz", a.out / "diii_d_public_test.npz"
    if src.exists() and not dst.exists():
        os.link(src, dst)
    fl = np.concatenate([flags[i] for i in range(len(files))])
    old = np.concatenate([z[f"shot_{i:04d}_q95"] for i in range(len(files))])
    new = np.concatenate([arrays[f"shot_{i:04d}_q95"] for i in range(len(files))])
    np.savez(a.out / "cc_flags.npz", flag=fl, q_old=old, q_new=new)
    print(f"wrote {a.out}: factor {a.factor} on {fl.mean():.2%} of {len(fl)} frames", flush=True)


if __name__ == "__main__":
    main()
