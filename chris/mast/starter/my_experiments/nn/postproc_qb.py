"""Replace the q95 and βN heads in a MAST submission with physics estimates from the submitted flux map. Label-free.

usage: python postproc_qb.py SUB_DIR OUT_DIR [--workers 8] [--variant _sol0.5]
       python postproc_qb.py --calibrate CKPT          (DIII-D val: R² of each estimate against the labels)

Per MAST frame, from the submitted ψ and MAST inputs only (mast_diag/physics_suite.frame_checks):
  q95  = F / (2π) ∮ dl / (R |∇ψ|) on the ψ_N = 0.95 surface of the map, F = R0·B0 with B0 from the q_ITER feature
         (measured TF current; MAST TF turns 24). On DIII-D true maps this matches the label to ±1% (10–90%).
  βN   = BN_FACTOR · βN_e, where βN_e uses the Thomson electron pressure mapped onto the map's flux surfaces and
         averaged over the plasma volume. BN_FACTOR = 1 / 0.52 accounts for the pressure Thomson does not see (ions,
         fast particles); 0.52 is the median βN_e / βN on DIII-D val true maps (correlation 0.97).
Frames where an estimate cannot be computed (no boundary, too few Thomson channels) keep the head's value. ψ is
copied unchanged; the DIII-D file is hard-linked. No MAST label is read.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mast_diag"))
import numpy as np
import pandas as pd

BN_FACTOR = 1 / 0.52
Q95_RANGE, BN_RANGE = (1.5, 40.0), (0.0, 10.0)
C = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
SSFX = "" if SPLIT == "mast_public_test" else f"_{SPLIT}"   # file-name suffix for caches without the split in their name
MAST = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data" / SPLIT
_STATE = {}


def _init(variant):
    import physics_suite as ps
    import features_v5 as f5
    _STATE["ps"] = ps
    _STATE["m"] = ps.Machine("MAST")
    import mast_region
    _STATE["feat"] = np.load(C / f"features_v5_{SPLIT}{variant}{mast_region.suffix()}.npy", mmap_mode="r")
    _STATE["ip"] = np.load(C / f"mast_inputs{SSFX}.npz")["ip"]
    _STATE["fq"] = f5.FEATURES.index("q_ITER")


def _shot(job):
    """(i, start, psi (T,65,65), q_head, b_head) -> (i, q95, βN, n_q, n_b)."""
    i, start, psi, qh, bh = job
    ps, m = _STATE["ps"], _STATE["m"]
    rows = start + np.arange(len(psi))
    ip = _STATE["ip"][rows]
    bt = ps.bt_from_qiter(np.asarray(_STATE["feat"][rows])[:, _STATE["fq"]], ip, m.r0, m.a)
    q, b = qh.astype(np.float32).copy(), bh.astype(np.float32).copy()
    nq = nb = 0
    for k in range(len(psi)):
        o = ps.frame_checks(m, psi[k], int(rows[k]), ip[k], bt[k]) or {}
        qv, bv = o.get("q95_map", np.nan), o.get("betaN_th", np.nan)
        if np.isfinite(qv) and Q95_RANGE[0] <= qv <= Q95_RANGE[1]:
            q[k] = qv; nq += 1
        if np.isfinite(bv) and BN_RANGE[0] <= BN_FACTOR * bv <= BN_RANGE[1]:
            b[k] = BN_FACTOR * bv; nb += 1
    return i, q, b, nq, nb


def postprocess(sub: Path, out: Path, workers: int, variant: str):
    out.mkdir(parents=True, exist_ok=True)
    src = sub / "diii_d_public_test.npz"
    dst = out / "diii_d_public_test.npz"
    if src.exists() and not dst.exists():
        os.link(src, dst)
    z = np.load(sub / f"{SPLIT}.npz")
    files = sorted(MAST.glob("*.parquet"))
    index = pd.read_parquet(C / f"features_v5_{SPLIT}{variant}.parquet").set_index("filename")
    jobs = []
    for i, f in enumerate(files):
        jobs.append((i, int(index.loc[f.name, "start"]), z[f"shot_{i:04d}_psirz"], z[f"shot_{i:04d}_q95"],
                     z[f"shot_{i:04d}_betaN"]))
    arrays = {k: z[k] for k in z.files}
    tq = tb = tot = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(variant,)) as ex:
        for done, (i, q, b, nq, nb) in enumerate(ex.map(_shot, jobs, chunksize=4), start=1):
            arrays[f"shot_{i:04d}_q95"], arrays[f"shot_{i:04d}_betaN"] = q, b
            tq += nq; tb += nb; tot += len(q)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} shots", flush=True)
    np.savez(out / f"{SPLIT}.npz", **arrays)
    print(f"wrote {out / f'{SPLIT}.npz'}: physics q95 on {tq / tot:.1%} of {tot} frames, "
          f"physics βN on {tb / tot:.1%}", flush=True)


def calibrate(ckpt: Path):
    """DIII-D val: R² against the labels of the heads and of the physics estimates from the model's own maps."""
    import torch
    import physics_suite as ps
    from model import load_checkpoint
    net, ck = load_checkpoint(ckpt, torch.device("cpu"))
    sh = pd.read_parquet(C / "shots.parquet"); val = sh[sh.split == "val"].sample(20, random_state=7)
    rows = pd.DataFrame(ps.d3d_rows(ps.Machine("DIII-D"), net, ck, val, model=True))
    r2 = lambda y, p: 1 - np.nansum((y - p) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
    ok = rows.q95_map.notna() & rows.betaN_th.notna()
    rows = rows[ok]
    print(f"DIII-D val, {len(rows)} frames, model maps: q95 from map R² {r2(rows.q95_ref, rows.q95_map):.4f}, "
          f"βN = {BN_FACTOR:.2f}·βN_e R² {r2(rows.betaN_ref, BN_FACTOR * rows.betaN_th):.4f}; "
          f"median βN_e / βN {np.median(rows.betaN_th / rows.betaN_ref):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variant", default="_sol0.5")
    ap.add_argument("--calibrate", type=Path)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.calibrate)
    else:
        postprocess(args.sub, args.out, args.workers, args.variant)
