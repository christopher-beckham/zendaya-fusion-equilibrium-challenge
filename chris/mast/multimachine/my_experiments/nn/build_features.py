"""Precompute a feature set (features_v2 or features_v3) once, for DIII-D train/val/test and both public test sets. CPU only.

    python my_experiments/nn/build_features.py --workers 3 [--version v3]

Shards land in `<cache>/features_<version>/shards/<config>/<shot>.npy` and are skipped if present, so a
stopped build resumes. The join step writes
    <cache>/features_<version>.npy                 (n_frames, F), row-aligned with coils.npy and psi.npy
    <cache>/features_<version>_<config>.npy, .parquet  per public test set, with each shot's start and n
    <cache>/features_<version>.json                column names, geometry constants, picture checksums
and then the coverage report runs/shift_report/report_<version>.md: DIII-D train frames set the range,
DIII-D public test and MAST public test are compared against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATA_ROOT  # noqa: E402
import importlib  # noqa: E402

import features_v2 as fv  # noqa: E402
from shift_report import MAST_TF_TURNS, coverage  # noqa: E402

HERE = Path(__file__).resolve().parent
TEST_CONFIGS = ("diii_d_public_test", "mast_public_test")
PASS_SHARE = 0.90
MAX_SHIFT_SD = 2.0


def _worker(job):
    path, pictures_file, shard = Path(job[0]), job[1], Path(job[2])
    fs = importlib.import_module(f"features_{job[3]}")
    if shard.exists():
        return shard.name, None
    try:
        feats = fs.shot_features(path, fv.load_pictures(Path(pictures_file)))
    except Exception as exc:  # noqa: BLE001 - reported per shot, the build carries on
        return shard.name, f"{type(exc).__name__}: {exc}"
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.parent / f".{shard.stem}.tmp.npy"
    np.save(tmp, feats)
    tmp.replace(shard)
    return shard.name, None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _join(paths: list[Path], counts: list[int] | None, shard_dir: Path, out: Path) -> tuple[np.ndarray, list[int]]:
    """Concatenate shards in `paths` order. With `counts`, each shot keeps its first n rows, as the cache does."""
    blocks, ns = [], []
    for i, p in enumerate(paths):
        block = np.load(shard_dir / f"{p.stem}.npy")
        if counts is not None:
            n = counts[i]
            if len(block) < n:
                raise RuntimeError(f"{p.name}: {len(block)} feature rows, the cache has {n}")
            block = block[:n]
        blocks.append(block)
        ns.append(len(block))
    feats = np.concatenate(blocks).astype(np.float32)
    np.save(out, feats)
    return feats, ns


def _report(fs, train: np.ndarray, d3d_test: np.ndarray, mast: np.ndarray, out: Path) -> None:
    lines = [
        f"# MAST input coverage, features_{fs.VERSION}",
        "",
        f"DIII-D train split: {len(train)} frames. DIII-D public test: {len(d3d_test)}. MAST public test: {len(mast)}.",
        f"Pass: at least {PASS_SHARE:.0%} of MAST frames inside DIII-D train's 0.5–99.5% range, "
        f"and MAST's median within {MAX_SHIFT_SD:g} DIII-D sd of DIII-D's.",
        "",
        "| feature | DIII-D range | DIII-D test inside | MAST inside | MAST median shift (sd) | "
        "DIII-D missing | MAST missing | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for j, name in enumerate(fs.FEATURES):
        if name in fs.FLAGS:
            lines.append(f"| {name} | flag | | | | | D {train[:, j].mean():.1%} / M {mast[:, j].mean():.1%} set | |")
            continue
        ct = coverage(train[:, j], d3d_test[:, j])
        cm = coverage(train[:, j], mast[:, j])
        miss = float(1.0 - np.isfinite(train[:, j]).mean())
        ok = cm["inside"] >= PASS_SHARE and abs(cm["shift_sd"]) <= MAX_SHIFT_SD
        lines.append(
            f"| {name} | {cm['lo']:.3g} … {cm['hi']:.3g} | {ct['inside']:.1%} | {cm['inside']:.1%} | "
            f"{cm['shift_sd']:+.2f} | {miss:.1%} | {cm['missing']:.1%} | {'pass' if ok else 'fail'} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--version", choices=["v2", "v3", "v4", "v5", "v6"], default="v2")
    ap.add_argument("--limit", type=int, default=0, help="shots per set, for a quick check; skips the join")
    ap.add_argument("--only-config", default=None,
                    help="build and join only this test set (e.g. mast_private_test) with the existing pictures; "
                         "DIII-D train, the .json and the report are left as they are")
    args = ap.parse_args()

    fs = importlib.import_module(f"features_{args.version}")
    name = f"features_{args.version}"
    root = args.cache / name
    # Coil pictures depend only on the geometry, so every version shares the v2 pair.
    pic_root = args.cache / "features_v2"
    pic_root.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    data = args.data_root / "data"
    if args.only_config:
        shots = None
        sets = {args.only_config: sorted((data / args.only_config).glob("*.parquet"))}
        if not sets[args.only_config]:
            raise SystemExit(f"no parquet files in {data / args.only_config}")
    else:
        shots = pd.read_parquet(args.cache / "shots.parquet").sort_values("start").reset_index(drop=True)
        sets = {
            "diii_d_train": [data / "diii_d_train" / f for f in shots["filename"]],
            **{c: sorted((data / c).glob("*.parquet")) for c in TEST_CONFIGS},
        }
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}
    pictures = {
        "d3d": pic_root / "pictures_d3d.npz",
        "mast": pic_root / "pictures_mast.npz",
    }
    for machine, path in pictures.items():
        if not path.exists() and args.only_config:
            raise SystemExit(f"{path} is missing: build the pictures with a full run first")
        if not path.exists():
            src = sets["diii_d_train"][0] if machine == "d3d" else sets["mast_public_test"][0]
            fv.build_pictures(src, path)
            print(f"built {machine} pictures from {src.name}", flush=True)

    jobs = []
    for config, paths in sets.items():
        pic = pictures["mast" if config.startswith("mast") else "d3d"]
        jobs += [(str(p), str(pic), str(root / "shards" / config / f"{p.stem}.npy"), args.version) for p in paths]
    print(f"{len(jobs)} shots, {args.workers} workers -> {root}", flush=True)
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (shard_name, err) in enumerate(ex.map(_worker, jobs, chunksize=4)):
            if err:
                failed.append((shard_name, err))
                print(f"  {shard_name}: {err}", flush=True)
            if (i + 1) % 250 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} shots, {len(failed)} failed", flush=True)
    if failed:
        print(f"{len(failed)} shots failed; rerun to retry them. No join.", flush=True)
        return 1
    if args.limit:
        print("--limit set: shards written, join skipped", flush=True)
        return 0
    if args.only_config:
        config = args.only_config
        feats, ns = _join(sets[config], None, root / "shards" / config, args.cache / f"{name}_{config}.npy")
        starts = np.concatenate([[0], np.cumsum(ns)[:-1]])
        pd.DataFrame({"shot": [p.stem for p in sets[config]], "filename": [p.name for p in sets[config]],
                      "start": starts, "n": ns}).to_parquet(args.cache / f"{name}_{config}.parquet")
        print(f"wrote {name}_{config}: {len(feats)} rows from {len(ns)} shots", flush=True)
        return 0

    train_all, _ = _join(sets["diii_d_train"], shots["n"].astype(int).tolist(),
                         root / "shards" / "diii_d_train", args.cache / f"{name}.npy")
    if len(train_all) != int(shots["n"].sum()):
        raise RuntimeError(f"{name}.npy has {len(train_all)} rows, the cache {int(shots['n'].sum())}")
    tests = {}
    for config in TEST_CONFIGS:
        feats, ns = _join(sets[config], None, root / "shards" / config, args.cache / f"{name}_{config}.npy")
        starts = np.concatenate([[0], np.cumsum(ns)[:-1]])
        pd.DataFrame({"shot": [p.stem for p in sets[config]], "filename": [p.name for p in sets[config]],
                      "start": starts, "n": ns}).to_parquet(args.cache / f"{name}_{config}.parquet")
        tests[config] = feats
    meta = {
        "version": fs.VERSION,
        "features": fs.FEATURES,
        "flags": fs.FLAGS,
        "geometry": fv.GEOMETRY,
        "kappa": fv.KAPPA,
        "mast_tf_turns": MAST_TF_TURNS,
        "d3d_ecoil_scale": fv.D3D_ECOIL_SCALE,
        "pictures_sha256": {m: _sha(p) for m, p in pictures.items()},
        "n_frames": int(len(train_all)),
    }
    (args.cache / f"{name}.json").write_text(json.dumps(meta, indent=2))
    split = np.repeat(shots["split"].to_numpy(), shots["n"].astype(int).to_numpy())
    _report(fs, train_all[split == "train"], tests["diii_d_public_test"], tests["mast_public_test"],
            HERE / "runs" / "shift_report" / f"report_{args.version}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
