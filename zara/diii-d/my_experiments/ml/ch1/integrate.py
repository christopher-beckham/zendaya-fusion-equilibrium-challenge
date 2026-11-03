"""Integrate member predictions (any workstream, same per-shot layout: psi (T,65,65), q95 (T,), betaN (T,) [, t]).

    python integrate.py --out runs/<name>/preds --split 0 \
        --psi DIR [DIR ...] [--psi-w w1 w2 ...] [--psi-mode mean|median|trim] [--trim 0.2] \
        [--scal DIR ...] [--scal-w ...] [--scal-mode mean|median] [--li-src DIR ...]

psi is combined per pixel over the --psi members (weighted mean, median, or trimmed mean that drops the --trim fraction of
members at each end per pixel); q95 / betaN over the --scal members (default: the psi members). Writes <out>/<stem>.npz and
<out>/../split_seed. --li-src: per-frame li targets (files carrying an "li" array, e.g. ml/ch1c scalar-head predictions) are
averaged over the given dirs and stored as "li" in the integrated files, so the integrated dir can be passed to
`lical.py --li-src <out>` (snapshot of ml/ch1c/lical.py: li calibration + li range guard in one step). Without li targets the
li guard is `liguard.py`. Everything runs inside Slurm jobs (slurm/pipe.sbatch chains integrate -> li step -> scoring).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def combine(arrs, w, mode, trim):
    A = np.stack(arrs).astype(np.float64)              # (M, ...)
    if mode == "mean":
        w = np.asarray(w, np.float64)
        return np.tensordot(w / w.sum(), A, axes=1)
    if mode == "median":
        return np.median(A, 0)
    k = int(np.floor(trim * len(arrs)))
    if k == 0:
        return A.mean(0)
    S = np.sort(A, 0)
    return S[k:len(arrs) - k].mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--psi", nargs="+", type=Path, required=True)
    ap.add_argument("--psi-w", nargs="*", type=float, default=None)
    ap.add_argument("--psi-mode", choices=["mean", "median", "trim"], default="mean")
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--scal", nargs="*", type=Path, default=None)
    ap.add_argument("--scal-w", nargs="*", type=float, default=None)
    ap.add_argument("--scal-mode", choices=["mean", "median", "trim"], default="mean")
    ap.add_argument("--li-src", nargs="*", type=Path, default=None, help="dirs whose files carry a per-frame 'li' target")
    a = ap.parse_args()
    scal = a.scal or a.psi
    pw = a.psi_w or [1.0] * len(a.psi)
    sw = a.scal_w or [1.0] * len(scal)
    assert len(pw) == len(a.psi) and len(sw) == len(scal)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out.parent / "split_seed").write_text(f"{a.split}\n")
    (a.out.parent / "members.txt").write_text("psi " + " ".join(map(str, a.psi)) + f" w {pw} mode {a.psi_mode}\nscal "
                                              + " ".join(map(str, scal)) + f" w {sw} mode {a.scal_mode}\n"
                                              + ("li_src " + " ".join(map(str, a.li_src)) + "\n" if a.li_src else ""))
    stems = sorted(p.stem for p in a.psi[0].glob("*.npz"))
    for s in stems:
        P = [np.load(d / f"{s}.npz") for d in a.psi]
        Q = [np.load(d / f"{s}.npz") for d in scal]
        out = {"psi": combine([p["psi"] for p in P], pw, a.psi_mode, a.trim).astype(np.float32)}
        for k in ("q95", "betaN"):
            out[k] = combine([q[k] for q in Q], sw, a.scal_mode, a.trim).astype(np.float32)
        if "t" in P[0].files:
            out["t"] = P[0]["t"]
        if a.li_src:
            li = np.mean([np.load(d / f"{s}.npz")["li"].astype(np.float64) for d in a.li_src], 0)
            assert li.shape == out["q95"].shape, (s, li.shape)
            out["li"] = li.astype(np.float32)
        assert np.isfinite(out["psi"]).all() and np.isfinite(out["q95"]).all() and np.isfinite(out["betaN"]).all(), s
        np.savez(a.out / f"{s}.npz", **out)
    print(f"{len(stems)} shots -> {a.out}")


if __name__ == "__main__":
    main()
