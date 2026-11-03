"""Per-frame flux at the magnetic axis and at the boundary for a gen4 synthetic cache (synth_cache_seq.py), for the
shape losses in shape_loss.py. Read from the synthetic parquet labels (efit_psi_axis, efit_psi_boundary), sign-flipped
like the cached maps, and written next to the cache as shape_targets.npz (rows aligned with arrays.npz).

    FEC_MAST_MASK=limiter python synth_shape_targets.py zara_g4_5000 [--workers 16]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import CACHE
import mast_region


def _read(job):
    path = job
    t = pq.read_table(path, columns=["efit_psi_axis", "efit_psi_boundary"]).to_pylist()[0]
    return np.asarray(t["efit_psi_axis"], np.float64).ravel(), np.asarray(t["efit_psi_boundary"], np.float64).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    folder = CACHE / "synth" / a.tag
    if not (folder / "manifest.json").exists():
        folder = folder.with_name(folder.name + mast_region.suffix())
    meta = json.loads((folder / "manifest.json").read_text())
    z = np.load(folder / "arrays.npz", mmap_mode="r")
    case, frame, split = np.asarray(z["case"]), np.asarray(z["frame"]), np.asarray(z["split"])
    src = {"train": Path(meta["source"]), "heldout": Path(meta["heldout_source"])}
    cases = sorted(set(case.tolist()))
    paths = [src[c.split("_", 1)[0]] / (c.split("_", 1)[1] + ".parquet") for c in cases]
    with ProcessPoolExecutor(a.workers) as ex:
        got = dict(zip(cases, ex.map(_read, paths, chunksize=16)))
    ax = np.array([-got[c][0][f] for c, f in zip(case, frame)], np.float32)
    bd = np.array([-got[c][1][f] for c, f in zip(case, frame)], np.float32)
    # check: the cached map at the pixel nearest the labelled axis should be close to the axis flux
    import features_v2
    g = features_v2.geometry("mast")
    sc = np.asarray(z["scal"])
    ir = np.abs(g["r"][None, :] - sc[:, 3:4]).argmin(1); iz = np.abs(g["z"][None, :] - sc[:, 4:5]).argmin(1)
    psi = np.asarray(z["psi"]).reshape(len(case), len(g["z"]), len(g["r"]))
    at_ax = psi[np.arange(len(case)), iz, ir]
    rel = np.abs(at_ax - ax) / np.abs(ax - bd)
    print(f"{folder.name}: {len(case)} frames; |map at axis pixel − axis flux| / |axis − boundary| median "
          f"{np.median(rel):.3f}, 95% {np.percentile(rel, 95):.3f}; axis−boundary sign {np.sign(np.median(ax - bd)):+.0f}",
          flush=True)
    np.savez(folder / "shape_targets.npz", psi_axis=ax, psi_bnd=bd)
    print(f"wrote {folder / 'shape_targets.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
