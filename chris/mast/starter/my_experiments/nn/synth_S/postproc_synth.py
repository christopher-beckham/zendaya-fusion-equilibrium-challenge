#!/usr/bin/env python3
"""Submitted q95 / βN for synthetic held-out predictions, made exactly as for our real MAST submissions.

    CUDA_VISIBLE_DEVICES= [FEC_MAST_MASK=limiter] python postproc_synth.py SET PRED.npz OUT.npz [--workers 16]

The real pipeline (build_candidate.sh): postproc_qb.py replaces the q95 / βN heads by physics estimates from the
submitted map (mast_diag/physics_suite.frame_checks: q95 from the ψ_N = 0.95 surface with B0 from the q_ITER
feature; βN = BN_FACTOR · βN_e from Thomson electron pressure), keeping the head where an estimate is missing or out
of range; then βN × 1.2 on every frame. This wrapper runs the same frame_checks and the same acceptance rules
(postproc_qb's BN_FACTOR, Q95_RANGE, BN_RANGE, imported from the unmodified main-repo file) with the MAST inputs
taken from the synthetic frame instead of a public-test row:
  Thomson Te, ne and chord R, Z    the set's te, ne, tr, tz (raw Thomson at the labelled time, as built by
                                   build_thomson_raw.shot, the same function as thomson_raw_mast.npz)
  q_ITER                           feats[:, mid] column q_ITER (features_v5 at the labelled time, the same column
                                   postproc_qb reads from features_v5_mast_public_test_sol0.5[_limiter].npy)
  Ip (kA)                          ip[:, mid]
The MAST region (mask, R0, a) is the process's FEC_MAST_MASK selection, as for the real post-processing.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MAIN_NN, SCR  # noqa: E402

sys.path.insert(0, str(MAIN_NN / "mast_diag"))
sys.path.insert(0, str(MAIN_NN))
BETAN_MULT = 1.2           # build_candidate.sh: z[k] * 1.2 for every *_betaN after postproc_qb
_S: dict = {}


def _init(set_name):
    import physics_suite as ps
    import features_v5 as f5
    import mast_region
    m = ps.Machine("MAST")
    z = np.load(SCR / "sets" / f"{set_name}_ho{mast_region.suffix()}" / "arrays.npz")
    m.te, m.ne, m.cr, m.cz = z["te"], z["ne"], z["tr"], z["tz"]     # row = set row
    mid = int(z["mid"])
    _S.update(ps=ps, m=m, ip=z["ip"][:, mid].astype(np.float64),
              qiter=z["feats"][:, mid, f5.FEATURES.index("q_ITER")].astype(np.float64))


def _chunk(job):
    """As postproc_qb._shot, over a list of set rows."""
    import postproc_qb as pq_
    rows, psi, qh, bh = job
    ps, m = _S["ps"], _S["m"]
    ip = _S["ip"][rows]
    bt = ps.bt_from_qiter(_S["qiter"][rows], ip, m.r0, m.a)
    q, b = qh.astype(np.float32).copy(), bh.astype(np.float32).copy()
    nq = nb = 0
    for k in range(len(rows)):
        o = ps.frame_checks(m, psi[k], int(rows[k]), ip[k], bt[k]) or {}
        qv, bv = o.get("q95_map", np.nan), o.get("betaN_th", np.nan)
        if np.isfinite(qv) and pq_.Q95_RANGE[0] <= qv <= pq_.Q95_RANGE[1]:
            q[k] = qv; nq += 1
        if np.isfinite(bv) and pq_.BN_RANGE[0] <= pq_.BN_FACTOR * bv <= pq_.BN_RANGE[1]:
            b[k] = pq_.BN_FACTOR * bv; nb += 1
    return rows, q, b, nq, nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("pred", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--true-maps", action="store_true", help="sanity: use the set's true ψ (our sign) as the map")
    a = ap.parse_args()
    t0 = time.time()
    import mast_region
    z = dict(np.load(a.pred))
    if a.true_maps:
        zz = np.load(SCR / "sets" / f"{a.set}_ho{mast_region.suffix()}" / "arrays.npz")
        z["psi"] = zz["psi"]
    n = len(z["psi"])
    jobs = [(r, z["psi"][r], z["q95_head"][r], z["betaN_head"][r]) for r in np.array_split(np.arange(n), 64)]
    q, b = np.empty(n, np.float32), np.empty(n, np.float32)
    nq = nb = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.set,)) as ex:
        for rows, qq, bb, i, j in ex.map(_chunk, jobs):
            q[rows], b[rows] = qq, bb
            nq += i; nb += j
    b = np.asarray(b * BETAN_MULT, np.float32)
    np.savez(a.out, psi=z["psi"], q95=q, betaN=b, case=z["case"], frame=z["frame"])
    print(f"post-processed {n} frames in {time.time() - t0:.0f} s (region {mast_region.name()}): physics q95 on "
          f"{nq / n:.1%}, physics betaN on {nb / n:.1%}; q95 10/50/90 {np.percentile(q, [10, 50, 90]).round(2)}, "
          f"betaN 10/50/90 {np.percentile(b, [10, 50, 90]).round(2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
