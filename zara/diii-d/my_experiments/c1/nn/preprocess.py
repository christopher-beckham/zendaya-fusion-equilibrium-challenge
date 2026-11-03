"""Build a per-shot cache for the GPU residual model (my_experiments/c1/nn).

    python preprocess.py --data-dir .../diii_d_train --out $FEQ_CACHE/nn/diii_d_train --workers 20

For every DIII-D parquet shot, one UNCOMPRESSED .npz with
    X        (T, F)  float32  C1 features incl. physics options field,shaf,eddy (raw; NaN = missing)
    I        (T, 19) float32  coil currents, A-turn, in the shot's Ip > 0 frame
    ip, bt   (T,)    float32  plasma current (A), |B_T| coil current (A)
    sign     ()      float32  +1 / -1 (reversed-Ip shot)
    t        (T,)    float32  efit_times (ms)
  train only:
    psi      (T, 65, 65) float32  target flux map (NOT sign-flipped, as the model's target)
    q95, betaN, li, r_axis, z_axis   (T,) float32   EFIT labels (NaN where missing)
    lcfs_r, lcfs_z   (T, NMAX) float32  EFIT LCFS polygon, NaN-padded
and <out>/meta.json with the feature names plus <out>/basis.npz with the coil Green's
functions (identical geometry on every shot is asserted via the per-shot geometry key).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                    # my_experiments/c1
from model import GRID_R, GRID_Z, PLASMA_FILAMENT, C1Model, target_psi  # noqa: E402
from greens import coil_basis, filament_psi                               # noqa: E402

PHYSICS = ("field", "shaf", "eddy")
NMAX = 256
LABELS = {"q95": "efit_q95", "betaN": "efit_beta_n", "li": "efit_li",
          "r_axis": "efit_r_axis", "z_axis": "efit_z_axis"}


def _geom_key(row) -> str:
    return ",".join(f"{v:.5f}" for v in np.concatenate(
        [np.asarray(row[k], dtype=float) for k in
         ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")]))


def _lcfs(row, T):
    r = np.full((T, NMAX), np.nan, np.float32)
    z = np.full((T, NMAX), np.nan, np.float32)
    try:
        n = np.asarray(row["efit_lcfs_n"], dtype=float)
        lr, lz = row["efit_lcfs_r"], row["efit_lcfs_z"]
        for k in range(T):
            a = np.asarray(lr[k], dtype=np.float32)
            b = np.asarray(lz[k], dtype=np.float32)
            m = int(n[k]) if np.isfinite(n[k]) else min(len(a), len(b))
            m = max(0, min(m, len(a), len(b), NMAX))
            r[k, :m], z[k, :m] = a[:m], b[:m]
    except Exception as e:  # noqa: BLE001
        print("  lcfs failed:", e, flush=True)
    return r, z


def _one(args):
    f, out_dir, with_targets = args
    out = Path(out_dir) / (Path(f).stem + ".npz")
    if out.exists():
        return Path(f).stem, None, "skip"
    row = pd.read_parquet(f).iloc[0].to_dict()
    if row.get("source", "DIII-D") != "DIII-D":
        return Path(f).stem, None, "not-d3d"
    m = C1Model(physics=PHYSICS)
    inp = m._inputs(row)
    T = len(inp["ip"])
    data = {"X": inp["X"].astype(np.float32), "I": inp["I"].astype(np.float32),
            "ip": inp["ip"].astype(np.float32), "bt": inp["bt"].astype(np.float32),
            "sign": np.float32(inp["sign"]),
            "t": np.asarray(row["efit_times"], dtype=np.float32)}
    if with_targets:
        data["psi"] = target_psi(row).astype(np.float32)
        assert data["psi"].shape == (T, 65, 65), data["psi"].shape
        for k, col in LABELS.items():
            data[k] = np.asarray(row[col], dtype=np.float32)
        data["lcfs_r"], data["lcfs_z"] = _lcfs(row, T)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, out)
    return Path(f).stem, (list(map(str, inp["names"])), _geom_key(row), list(map(str, inp["cols"]))), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--no-targets", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in sorted(args.data_dir.glob("*.parquet"))[: args.max_shots or None]]
    with_targets = not args.no_targets
    print(f"{len(files)} shots from {args.data_dir} -> {args.out}  ({args.workers} workers, "
          f"targets={with_targets})", flush=True)
    t0 = time.time()
    names = key = cols = None
    keys = {}
    n_ok = n_skip = 0
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (stem, info, status) in enumerate(pool.map(_one, [(f, str(args.out), with_targets) for f in files],
                                                           chunksize=4), 1):
            if status == "ok":
                n_ok += 1
                names, k, cols = info
                keys[k] = keys.get(k, 0) + 1
                key = k
            elif status == "skip":
                n_skip += 1
            if i % 250 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  ok {n_ok} skipped {n_skip}  {time.time() - t0:.0f}s", flush=True)
    if names is not None:
        meta = {"names": names, "cols": cols, "n_features": len(names), "physics": list(PHYSICS),
                "geometry_keys": {k: v for k, v in keys.items()}, "data_dir": str(args.data_dir)}
        (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
        # Green's functions for the (single) coil geometry, from the last shot read
        row = pd.read_parquet(files[-1], columns=["coil_R", "coil_Z", "coil_width", "coil_height",
                                                   "coil_angle1", "coil_angle2", "coil_input_column"]
                              ).iloc[0].to_dict()
        gcols, G = coil_basis(row, GRID_R, GRID_Z)
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        Gp = filament_psi(*PLASMA_FILAMENT, RR, ZZ)
        np.savez(args.out / "basis.npz", G=G, Gp=Gp, cols=np.asarray(gcols), grid_R=GRID_R, grid_Z=GRID_Z,
                 geom_key=np.asarray(_geom_key(row)))
        print(f"geometry keys seen: {len(keys)}  -> basis.npz from {Path(files[-1]).stem}", flush=True)
        if len(keys) > 1:
            print("WARNING: more than one coil geometry across shots:", {k[:40]: v for k, v in keys.items()})
    print(f"done: {n_ok} written, {n_skip} already present, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
