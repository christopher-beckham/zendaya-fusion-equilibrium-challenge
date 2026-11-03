"""Per-shot cache for MAST (Challenge 2) plus the machine's vacuum Green's functions.

    python preprocess_mast.py --data-dir .../mast_public_test --out $R1_ROOT/cache/mast/mast_public_test --workers 16

Per shot <out>/<stem>.npz:
    t        (T,)      efit_times (ms)
    I        (T, 13)   current columns at efit_times (window mean +-1 ms), in A (or A-turn for the P coils)
    ip       (T,)      plasma current, A (sign as shipped; MAST Ip is positive in the shipped data)
    itf      (T,)      TF current, A (per turn as shipped)
    ts_R     (130,)    core Thomson radii (midplane, Z = 0)
    ts_Te    (T, 130)  core Te (eV) interpolated in time onto efit_times (nearest profile within +-ts_window ms)
    ts_ne    (T, 130)  core ne (m^-3)
    ts_age   (T,)      |t - t_ts| of the profile used (ms), NaN if none
    te_R     (16,)     edge Thomson radii; te_Te, te_ne (T, 16)
<out>/basis.npz: per current column the flux image (65,65) per AMPERE PER ELEMENT summed over that column's
elements (G_col = sum_e G_e), the element counts, grid, and the geometry key; <out>/meta.json.
Units: p{2..6}{u,l} are kA-turn for the whole coil -> element current = 1e3 * I_col / n_elements;
sol is kA per turn (656 single-turn elements) with the x2 convention caveat; tf and efps carry no poloidal flux.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "c1"))
from data_fixes import align_mast_thomson_core          # noqa: E402
from features import window_mean                        # noqa: E402
from greens import coil_filaments, filament_psi          # noqa: E402

COLS = ["magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current", "magnetics_p3u_current",
        "magnetics_p4l_current", "magnetics_p4u_current", "magnetics_p5l_current", "magnetics_p5u_current",
        "magnetics_p6l_current", "magnetics_p6u_current", "magnetics_sol_current", "magnetics_tf_current",
        "magnetics_efps_current"]
TS_WINDOW_MS = 6.0


def _geom_key(row) -> str:
    return ",".join(f"{v:.5f}" for v in np.concatenate(
        [np.asarray(row[k], dtype=float) for k in ("coil_R", "coil_Z", "coil_width", "coil_height")]))


def _ts_at(times, prof, tq, half):
    """Nearest profile in time within +-half ms (NaN rows where none)."""
    times = np.asarray(times, dtype=np.float64)
    order = np.argsort(times)
    times, prof = times[order], prof[order]
    out = np.full((len(tq), prof.shape[1]), np.nan)
    age = np.full(len(tq), np.nan)
    if len(times) == 0:
        return out, age
    k = np.searchsorted(times, tq)
    k0 = np.clip(k - 1, 0, len(times) - 1); k1 = np.clip(k, 0, len(times) - 1)
    pick = np.where(np.abs(times[k0] - tq) <= np.abs(times[k1] - tq), k0, k1)
    d = np.abs(times[pick] - tq)
    ok = d <= half
    out[ok] = prof[pick[ok]]
    age[ok] = d[ok]
    return out, age


def _one(args):
    f, out_dir = args
    out = Path(out_dir) / (Path(f).stem + ".npz")
    if out.exists():
        return Path(f).stem, None, "skip"
    row = pd.read_parquet(f).iloc[0].to_dict()
    if row.get("source") != "MAST":
        return Path(f).stem, None, "not-mast"
    te = np.asarray(row["efit_times"], dtype=np.float64)
    tm = np.asarray(row["magnetics_time"], dtype=np.float64)
    I = np.column_stack([window_mean(tm, np.asarray(row[c], dtype=np.float64), te, 1.0) for c in COLS]) * 1e3
    ip = window_mean(tm, np.asarray(row["magnetics_plasma_current"], dtype=np.float64), te, 1.0) * 1e3
    itf = I[:, COLS.index("magnetics_tf_current")]
    R, Te, ne = align_mast_thomson_core(row)
    Te = np.where(Te > 0, Te, np.nan); ne = np.where(ne > 0, ne, np.nan)
    ts_Te, age = _ts_at(row["thomson_core_times"], Te, te, TS_WINDOW_MS)
    ts_ne, _ = _ts_at(row["thomson_core_times"], ne, te, TS_WINDOW_MS)
    eR = np.asarray(row["thomson_edge_spatial"], dtype=np.float64)
    try:
        eTe = np.stack([np.asarray(p, dtype=np.float64) for p in row["thomson_edge_Te"]])
        ene = np.stack([np.asarray(p, dtype=np.float64) for p in row["thomson_edge_ne"]])
        eTe = np.where(eTe > 0, eTe, np.nan); ene = np.where(ene > 0, ene, np.nan)
        te_Te, _ = _ts_at(row["thomson_edge_times"], eTe, te, TS_WINDOW_MS)
        te_ne, _ = _ts_at(row["thomson_edge_times"], ene, te, TS_WINDOW_MS)
    except Exception:  # noqa: BLE001
        te_Te = te_ne = np.full((len(te), len(eR)), np.nan)
    data = {"t": te.astype(np.float32), "I": I.astype(np.float32), "ip": ip.astype(np.float32),
            "itf": itf.astype(np.float32), "ts_R": R.astype(np.float32), "ts_Te": ts_Te.astype(np.float32),
            "ts_ne": ts_ne.astype(np.float32), "ts_age": age.astype(np.float32), "te_R": eR.astype(np.float32),
            "te_Te": te_Te.astype(np.float32), "te_ne": te_ne.astype(np.float32),
            "grid_R": np.asarray(row["efit_grid_R"], dtype=np.float32), "grid_Z": np.asarray(row["efit_grid_Z"], dtype=np.float32)}
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, out)
    stats = {"T": int(len(te)), "t0": float(te[0]), "t1": float(te[-1]), "ip_med": float(np.nanmedian(ip)),
             "ip_min": float(np.nanmin(ip)), "ip_max": float(np.nanmax(ip)),
             "ts_cover": float(np.isfinite(age).mean()), "grid_key": (float(row["efit_grid_R"][0]), float(row["efit_grid_R"][-1]),
                                                                       float(row["efit_grid_Z"][0]), float(row["efit_grid_Z"][-1])),
             "I_med": [float(v) for v in np.nanmedian(I, 0)]}
    return Path(f).stem, (_geom_key(row), stats), "ok"


def build_basis(row, grid_R, grid_Z, n_sub=3):
    cols = np.asarray(row["coil_input_column"]).astype(str)
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    G = np.zeros((len(COLS), len(grid_Z), len(grid_R)))
    counts = np.zeros(len(COLS), int)
    for i, c in enumerate(COLS):
        sel = np.nonzero(cols == c)[0]
        counts[i] = len(sel)
        for e in sel:
            fr, fz = coil_filaments(float(row["coil_R"][e]), float(row["coil_Z"][e]), float(row["coil_width"][e]),
                                    float(row["coil_height"][e]), float(row["coil_angle1"][e]), float(row["coil_angle2"][e]), n_sub)
            G[i] += filament_psi(fr[:, None, None], fz[:, None, None], RR[None], ZZ[None]).mean(0)
    return G, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--max-shots", type=int, default=0)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in sorted(args.data_dir.glob("*.parquet"))[: args.max_shots or None]]
    print(f"{len(files)} shots from {args.data_dir} -> {args.out} ({args.workers} workers)", flush=True)
    t0 = time.time()
    keys, stats = {}, {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (stem, info, status) in enumerate(pool.map(_one, [(f, str(args.out)) for f in files], chunksize=4), 1):
            if status == "ok":
                k, st = info
                keys[k] = keys.get(k, 0) + 1
                stats[stem] = st
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    (args.out / "shot_stats.json").write_text(json.dumps(stats, indent=0))
    row = pd.read_parquet(files[0]).iloc[0].to_dict()
    grid_R, grid_Z = np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)
    G, counts = build_basis(row, grid_R, grid_Z)
    np.savez(args.out / "basis.npz", G=G.astype(np.float32), cols=np.asarray(COLS), counts=counts, grid_R=grid_R, grid_Z=grid_Z,
             geom_key=np.asarray(_geom_key(row)))
    (args.out / "meta.json").write_text(json.dumps({"cols": COLS, "counts": counts.tolist(), "geometry_keys": len(keys),
                                                     "n_shots": len(stats), "ts_window_ms": TS_WINDOW_MS}, indent=1))
    ips = np.array([s["ip_med"] for s in stats.values()])
    print(f"geometry keys: {len(keys)}; grids: {set(tuple(s['grid_key']) for s in stats.values())}")
    print(f"Ip median per shot: min {ips.min():.0f} max {ips.max():.0f} A; negative-Ip shots: {(ips < 0).sum()}")
    print(f"frames total: {sum(s['T'] for s in stats.values())}; TS coverage mean {np.mean([s['ts_cover'] for s in stats.values()]):.3f}")
    Imed = np.array([s["I_med"] for s in stats.values()])
    for j, c in enumerate(COLS):
        print(f"  {c:28s} n_elem {counts[j]:4d}  median over shots {np.median(Imed[:, j]):12.0f}  min {Imed[:, j].min():12.0f} max {Imed[:, j].max():12.0f}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
