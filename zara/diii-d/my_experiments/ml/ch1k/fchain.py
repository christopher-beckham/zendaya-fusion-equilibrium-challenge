"""Fused calibration chain: per frame, kapcal stretch -> axcal axis warp -> lical li transform (+ li guard), all toward the
absolute targets of a targets.py dir, in memory (one output dir, no intermediate maps on disk). Same per-frame code as
kapcal.py / axcal.py / lical.py; q95 / betaN optionally replaced by those of --scal.

    python fchain.py --preds MAP --targets TG --out OUT [--steps k,a,l] [--retry] [--max-shift 0.12] [--scal DIR]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LI_MAX, LI_GUARD = 5.0, 4.9


def one_shot(args):
    import axcal
    import kapcal
    import lical
    src, tgf, dst, steps, kw, scal = args
    p = dict(np.load(src))
    z = np.load(tgf)
    psi = p["psi"].astype(np.float64)
    T = len(psi)
    get = lambda k: z[k].astype(np.float64) if k in z.files else np.full(T, np.nan)  # noqa: E731
    tk = np.c_[get("kappa"), get("tri_top"), get("tri_bot")]
    tR, tZ, tl = get("R"), get("Z"), get("li")
    cnt = {s: [0, 0] for s in steps}
    for k in range(T):
        cur = psi[k]
        for s in steps:
            if s == "k":
                if not np.isfinite(tk[k]).any():
                    continue
                new, info = kapcal.cal_frame(cur, tk[k], retry=kw["retry"], anchor=kw["anchor"], max_axis_mm=kw["max_axis_mm"])
            elif s == "a":
                if not (np.isfinite(tR[k]) or np.isfinite(tZ[k])):
                    continue
                new, info = axcal.cal_frame(cur, (tR[k], tZ[k]), max_shift=kw["max_shift"])
            elif s == "l":
                t = tl[k]

                def target_fn(li0, t=t):
                    if not np.isfinite(t):
                        return LI_GUARD if li0 > LI_MAX else None
                    return min(t, LI_GUARD) if t > LI_MAX else t
                new, info = lical.calib_frame(cur, target_fn)
                if new is None and kw.get("li_retry") and info is not None and "target" in info and "li0" in info:
                    # (opt-in) the full li step was rejected: try 1/2 and 1/4 of the correction
                    for f in (0.5, 0.25):
                        tt = info["li0"] + f * (info["target"] - info["li0"])
                        new, info2 = lical.calib_frame(cur, lambda li0, tt=tt: tt)
                        if new is not None:
                            break
            else:
                raise ValueError(s)
            if info is not None:
                cnt[s][0] += 1
            if new is not None:
                cnt[s][1] += 1
                cur = new.astype(np.float64)
        p["psi"][k] = cur.astype(np.float32)
    if scal:
        sc = np.load(Path(scal) / Path(src).name)
        p["q95"], p["betaN"] = sc["q95"].astype(np.float32), sc["betaN"].astype(np.float32)
    assert np.isfinite(p["psi"]).all()
    np.savez(dst, **p)
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", default="k,a,l")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="kapcal stretch about the X-point (single X-point end)")
    ap.add_argument("--max-axis-mm", type=float, default=1.0, help="kapcal: reject a stretch that moves the axis more")
    ap.add_argument("--li-retry", action="store_true", help="li step: retry rejected frames with 1/2 and 1/4 of the correction")
    ap.add_argument("--max-shift", type=float, default=0.12)
    ap.add_argument("--scal", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.preds.glob("*.npz"))
    steps = a.steps.split(",")
    kw = {"retry": a.retry, "max_shift": a.max_shift, "anchor": a.anchor, "max_axis_mm": a.max_axis_mm, "li_retry": a.li_retry}
    jobs = [(str(f), str(a.targets / f.name), str(a.out / f.name), steps, kw, str(a.scal) if a.scal else None) for f in files]
    t0 = time.time()
    tot = {s: [0, 0] for s in steps}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for c in pool.map(one_shot, jobs, chunksize=1):
            for s in steps:
                tot[s][0] += c[s][0]
                tot[s][1] += c[s][1]
    msg = ", ".join(f"{s}: {v[0]} targeted / {v[1]} modified" for s, v in tot.items())
    (a.out.parent / "fchain_report.json").write_text(json.dumps(tot))
    print(f"{len(files)} shots, {msg}, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
