"""One resumable pass over the DIII-D train shots. Writes memmaps under the cache dir.

Run this on the GPU allocation, not the login node. It reads every train parquet (about an hour)
and writes ~26 GB of ψ.

    python my_experiments/nn/build_cache.py

Shards land in `<cache>/shards/<shot>.npz` and are skipped if present. The concat step rewrites
the memmaps whenever the READY marker is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE, CODE_TO_PHASE, DATA_ROOT, N_PSI, PHASE_TO_CODE, SCALAR_COLS, SCALAR_NAMES,
    assign_splits, featurize, mirror, score_subset, train_files,
)


def _read_psi(path: Path) -> np.ndarray:
    col = pq.read_table(path, columns=["efit_psirz"]).column(0).combine_chunks()[0].values
    while hasattr(col, "flatten"):
        col = col.flatten()
    return col.to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, N_PSI)


def _read_scalars(path: Path, t_len: int) -> np.ndarray:
    table = pq.read_table(path, columns=SCALAR_COLS)
    cols = []
    for name in SCALAR_COLS:
        a = table.column(name).combine_chunks()[0].values.to_numpy(zero_copy_only=False)
        a = np.asarray(a, dtype=np.float32)
        if len(a) < t_len:
            a = np.concatenate([a, np.full(t_len - len(a), np.nan, np.float32)])
        cols.append(a[:t_len])
    return np.stack(cols, axis=1)


def _write_shard(path: Path, split: str, shard: Path) -> dict:
    """Featurise one shot and write its shard. Returns the shot header for the concat step."""
    feat = featurize(path)
    coils = mirror(feat["coils"], feat["s"])
    psi = _read_psi(path)
    n = min(len(coils), len(psi), len(feat["t"]))
    if n == 0:
        raise RuntimeError(f"{path.name}: no frames")
    scal = _read_scalars(path, n)
    phase = np.array([PHASE_TO_CODE[p] for p in feat["phase"][:n]], dtype=np.int8)
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.parent / f".{shard.stem}.tmp.npz"
    np.savez(
        tmp,
        coils=np.ascontiguousarray(coils[:n]),
        psi=np.ascontiguousarray(psi[:n]),
        scal=np.ascontiguousarray(scal),
        t=np.ascontiguousarray(feat["t"][:n].astype(np.float64)),
        phase=phase,
        s=np.float32(feat["s"]),
    )
    os.replace(tmp, shard)
    return {
        "shot": path.stem,
        "filename": path.name,
        "split": split,
        "s": float(feat["s"]),
        "n": int(n),
    }


def _worker(job):
    path, split, shard = job
    path, shard = Path(path), Path(shard)
    if shard.exists():
        with np.load(shard) as z:
            n = int(z["coils"].shape[0])
            s = float(z["s"])
        return {"shot": path.stem, "filename": path.name, "split": split, "s": s, "n": n, "cached": True}
    header = _write_shard(path, split, shard)
    header["cached"] = False
    return header


def _concat(headers: list[dict], cache: Path) -> None:
    n_frames = sum(h["n"] for h in headers)
    coils_mm = np.lib.format.open_memmap(cache / "coils.npy", mode="w+", dtype=np.float32, shape=(n_frames, 21))
    psi_mm = np.lib.format.open_memmap(cache / "psi.npy", mode="w+", dtype=np.float32, shape=(n_frames, N_PSI))
    scal_mm = np.lib.format.open_memmap(cache / "scal.npy", mode="w+", dtype=np.float32, shape=(n_frames, len(SCALAR_NAMES)))
    phase_mm = np.lib.format.open_memmap(cache / "phase.npy", mode="w+", dtype=np.int8, shape=(n_frames,))

    frame_shot, frame_index, frame_file, frame_t, frame_phase, frame_s, frame_split = [], [], [], [], [], [], []
    shot_rows = []
    cursor = 0
    for shot_index, h in enumerate(headers):
        shard = cache / "shards" / f"{h['shot']}.npz"
        with np.load(shard) as z:
            n = int(z["coils"].shape[0])
            sl = slice(cursor, cursor + n)
            coils_mm[sl] = z["coils"]
            psi_mm[sl] = z["psi"]
            scal_mm[sl] = z["scal"]
            phase_mm[sl] = z["phase"]
            t = np.asarray(z["t"])
            phase = np.asarray(z["phase"])
        frame_shot.append(np.full(n, h["shot"]))
        frame_index.append(np.full(n, shot_index, dtype=np.int32))
        frame_file.append(np.full(n, h["filename"]))
        frame_t.append(t)
        frame_phase.append(np.array([CODE_TO_PHASE[int(c)] for c in phase]))
        frame_s.append(np.full(n, h["s"], dtype=np.float32))
        frame_split.append(np.full(n, h["split"]))
        shot_rows.append({**h, "shot_index": shot_index, "start": cursor})
        cursor += n
        if (shot_index + 1) % 500 == 0:
            print(f"  concatenated {shot_index + 1}/{len(headers)}", flush=True)

    for mm in (coils_mm, psi_mm, scal_mm, phase_mm):
        mm.flush()
    del coils_mm, psi_mm, scal_mm, phase_mm

    frames = pa.table({
        "shot": pa.array(np.concatenate(frame_shot)),
        "shot_index": pa.array(np.concatenate(frame_index)),
        "filename": pa.array(np.concatenate(frame_file)),
        "t": pa.array(np.concatenate(frame_t)),
        "phase": pa.array(np.concatenate(frame_phase)),
        "s": pa.array(np.concatenate(frame_s)),
        "split": pa.array(np.concatenate(frame_split)),
    })
    pq.write_table(frames, cache / "frames.parquet")
    shots = pa.table({
        "shot": [r["shot"] for r in shot_rows],
        "shot_index": [r["shot_index"] for r in shot_rows],
        "filename": [r["filename"] for r in shot_rows],
        "split": [r["split"] for r in shot_rows],
        "s": [r["s"] for r in shot_rows],
        "start": [r["start"] for r in shot_rows],
        "n": [r["n"] for r in shot_rows],
    })
    pq.write_table(shots, cache / "shots.parquet")

    import pandas as pd
    subset = score_subset(pd.read_parquet(cache / "shots.parquet"))
    subset.to_parquet(cache / "score_subset.parquet")
    meta = {
        "n_frames": int(n_frames),
        "n_shots": len(headers),
        "scalar_names": SCALAR_NAMES,
        "coils": "mirrored (ECOILA, F-coils, Ip) * s; bcoil unchanged",
        "score_subset": int(len(subset)),
        "score_subset_reversed": int((subset["s"] < 0).sum()),
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    (cache / "READY").write_text(json.dumps(meta))
    print(f"cache ready: {len(headers)} shots, {n_frames} frames -> {cache}", flush=True)
    print(f"  score subset: {meta['score_subset']} val shots, {meta['score_subset_reversed']} reversed", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    files = train_files(args.data_root)
    splits = assign_splits(len(files))
    shard_dir = args.cache / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(path), str(splits[i]), str(shard_dir / f"{path.stem}.npz")) for i, path in enumerate(files)]
    print(f"{len(files)} train files, {args.workers} workers -> {args.cache}", flush=True)

    headers = [None] * len(jobs)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_worker, job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            headers[i] = fut.result()
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"  shards {done}/{len(jobs)}", flush=True)

    # Filename order, which is the file-list order, so shot_index matches assign_splits.
    _concat(headers, args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
