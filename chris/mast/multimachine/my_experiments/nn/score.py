"""Full composite S on held-out train shots, using the scorer's own code.

Run this on the GPU allocation, not the login node. Building a reference is about 10 ms per frame
(contour plus seven derived scalars) and is cached under the cache dir. Scoring a prediction
repeats that work on the predicted flux map. Two or three processes.

    python my_experiments/nn/score.py --mode perfect
    python my_experiments/nn/score.py --mode zeros
    python my_experiments/nn/score.py --ckpt runs/v1/best.pt --split test
    python my_experiments/nn/score.py --pred runs/v1/score_ep040 --split score
    python my_experiments/nn/score.py --ckpt runs/v1/best.pt --split train_score

`--split score` is the 100 fixed val shots. `--split train_score` is 100 fixed train shots, picked the
same way. Its metrics line has source "score_train", so readers of "score" lines still see val only.

`--mode perfect` and `--mode zeros` are the harness check (S = 1 and S = 0) and are not quantised.
Checkpoint and `--pred` outputs are cast to float16 ψ and float32 scalars before scoring, matching
the submission skeleton. Predictions are a directory of `<shot>.npz` with arrays psirz, q95, betaN.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

# Our common.py and fusion_scoring/common.py share a module name. Bind ours first, drop it from
# the import cache, then let local_score load the scorer's common.
import common as nn_common  # noqa: E402

_OURS = nn_common
sys.modules.pop("common", None)
import local_score  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402
import common as fusion_common  # noqa: E402

_MASK = {}


def _mask():
    if not _MASK:
        z = np.load(REPO / "fusion_scoring" / "masks" / "d3d_envelope.npz")
        coarse = z["mask_coarse"].astype(bool)
        _MASK["R"] = z["grid_R"]
        _MASK["Z"] = z["grid_Z"]
        _MASK["coarse"] = coarse
        _MASK["f"] = coarse.astype(np.float64)
    return _MASK


def load_gt(path: Path) -> dict:
    row = local_score._read_row(path)
    psi = np.asarray(row["efit_psirz"], dtype=np.float32)
    if psi.ndim != 3:
        psi = np.stack([np.asarray(frame, dtype=np.float32) for frame in row["efit_psirz"]])
    return {
        "psi": psi,
        "q95": np.asarray(row["efit_q95"], dtype=np.float64),
        "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
    }


def cached_reference(path: Path, psi: np.ndarray):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    m = _mask()
    ref = local_score.build_reference(psi, m["R"], m["Z"], m["coarse"], m["f"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(ref, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return ref


def _prediction(mode: str, gt: dict, pred_root: str, stem: str, quantize: bool) -> dict:
    if mode == "perfect":
        return {
            "psirz": np.asarray(gt["psi"], dtype=np.float64),
            "q95": np.asarray(gt["q95"], dtype=np.float64),
            "betaN": np.asarray(gt["betaN"], dtype=np.float64),
        }
    if mode == "zeros":
        return {
            "psirz": np.zeros_like(gt["psi"], dtype=np.float64),
            "q95": np.zeros(gt["q95"].shape, dtype=np.float64),
            "betaN": np.zeros(gt["betaN"].shape, dtype=np.float64),
        }
    with np.load(Path(pred_root) / f"{stem}.npz") as z:
        psi = np.asarray(z["psirz"])
        q95 = np.asarray(z["q95"])
        beta = np.asarray(z["betaN"])
    if quantize:
        psi = np.asarray(psi, dtype=np.float32).astype(np.float16).astype(np.float64)
        q95 = np.asarray(q95, dtype=np.float32).astype(np.float64)
        beta = np.asarray(beta, dtype=np.float32).astype(np.float64)
    else:
        psi = np.asarray(psi, dtype=np.float64)
        q95 = np.asarray(q95, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
    if psi.ndim == 2:
        psi = psi.reshape(-1, 65, 65)
    return {"psirz": psi, "q95": q95, "betaN": beta}


def _sums(gt: dict, ref) -> dict:
    psi = gt["psi"].astype(np.float64)
    fin = np.isfinite(psi)
    out = {
        "psi_sum": float(psi[fin].sum()),
        "psi_sumsq": float(np.square(psi[fin]).sum()),
        "psi_n": int(fin.sum()),
        "scal_sum": np.zeros(fusion_common.N_SCALARS),
        "scal_sumsq": np.zeros(fusion_common.N_SCALARS),
        "scal_n": np.zeros(fusion_common.N_SCALARS),
        "cons_sum": np.zeros(fusion_common.N_CONS),
        "cons_sumsq": np.zeros(fusion_common.N_CONS),
        "cons_n": np.zeros(fusion_common.N_CONS),
    }
    for j, name in enumerate(fusion_common.SCALARS):
        v = np.asarray(gt[name], dtype=np.float64)
        v = v[np.isfinite(v)]
        out["scal_sum"][j] = v.sum()
        out["scal_sumsq"][j] = np.square(v).sum()
        out["scal_n"][j] = v.size
    cons, cmask = ref[1], ref[2]
    for j in range(fusion_common.N_CONS):
        v = cons[cmask[:, j], j]
        out["cons_sum"][j] = v.sum()
        out["cons_sumsq"][j] = np.square(v).sum()
        out["cons_n"][j] = v.size
    return out


def prepare(job):
    parquet, ref_path = job
    gt = load_gt(Path(parquet))
    ref = cached_reference(Path(ref_path), gt["psi"])
    return _sums(gt, ref)


def residuals(job):
    parquet, pred_root, stem, mode, quantize, mean_psi = job
    gt = load_gt(Path(parquet))
    pred = _prediction(mode, gt, pred_root, stem, quantize)
    rr, _, _ = local_score.psi_residuals(gt["psi"], pred["psirz"], mean_psi)
    return rr


def score_one(job):
    parquet, ref_path, pred_root, stem, mode, quantize, psi_sign, means = job
    gt = load_gt(Path(parquet))
    ref = cached_reference(Path(ref_path), gt["psi"])
    pred = _prediction(mode, gt, pred_root, stem, quantize)
    m = _mask()
    return local_score.score_shot(
        gt, ref, pred, m["R"], m["Z"], m["coarse"], m["f"], psi_sign, means,
    )


def _import_model():
    """Load model.py against our common, without disturbing the scorer's common."""
    import importlib.util
    saved = sys.modules.get("common")
    sys.modules["common"] = _OURS
    spec = importlib.util.spec_from_file_location("nn_model", HERE / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if saved is not None:
        sys.modules["common"] = saved
    return mod


def write_ckpt_predictions(ckpt: Path, rows: pd.DataFrame, cache: Path, out_dir: Path, device: str,
                           data_root: Path) -> None:
    import torch
    nn_model = _import_model()
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but is not available")
    net, checkpoint = nn_model.load_checkpoint(ckpt, dev)
    offsets = nn_common.parse_window_ms(str(checkpoint["args"].get("window_ms") or ""))
    safe = nn_common.as_bool(checkpoint["args"].get("extra_safe_features", False))
    robust = nn_common.as_bool(checkpoint["args"].get("robust_inputs", False))
    coils_mm = np.load(cache / "coils.npy", mmap_mode="r")
    pictures = nn_common.checkpoint_pictures(checkpoint["args"])
    feature_set = checkpoint["norm"].get("feature_set")
    feat_mm = np.load(cache / f"features_{feature_set}.npy", mmap_mode="r") if feature_set else None
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows.itertuples(index=False)):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        cached = np.asarray(coils_mm[sl])
        path = nn_common.shot_parquet(data_root, row)
        if feature_set:
            # The cached features are the whole input. The cached currents at t drive the decoder picture.
            coils = np.array(feat_mm[sl], dtype=np.float32, copy=True)
            cols = checkpoint["norm"].get("feature_columns")
            if cols is not None:
                coils = coils[:, list(cols)]
            coils = nn_common.stack_history(coils, int(checkpoint["norm"].get("history_frames", 0)))
            picture = nn_common.external_flux(cached, pictures["green"], pictures["ecoil"])
        else:
            if len(offsets) == 0:
                coils = cached
            else:
                coils = nn_common.inputs_for_shot(path, offsets, cached)
            coils, picture = nn_common.append_image_features(
                nn_common.append_safe_features(coils, path, safe=safe, robust=robust), path, pictures, robust,
            )
        psi, q95, beta = nn_model.predict_raw(net, coils, picture, checkpoint["norm"], dev,
                                              ip_ka=cached[:, nn_common.IP_COL])
        np.savez(
            out_dir / f"{row.shot}.npz",
            psirz=psi.astype(np.float16),
            q95=q95.astype(np.float32),
            betaN=beta.astype(np.float32),
        )
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  predicted {i + 1}/{len(rows)}", flush=True)


def _add_sums(total, part):
    for key in ("psi_sum", "psi_sumsq", "psi_n"):
        total[key] += part[key]
    for key in ("scal_sum", "scal_sumsq", "scal_n", "cons_sum", "cons_sumsq", "cons_n"):
        total[key] = total[key] + part[key]


def _blank_sums():
    return {
        "psi_sum": 0.0, "psi_sumsq": 0.0, "psi_n": 0,
        "scal_sum": np.zeros(fusion_common.N_SCALARS),
        "scal_sumsq": np.zeros(fusion_common.N_SCALARS),
        "scal_n": np.zeros(fusion_common.N_SCALARS),
        "cons_sum": np.zeros(fusion_common.N_CONS),
        "cons_sumsq": np.zeros(fusion_common.N_CONS),
        "cons_n": np.zeros(fusion_common.N_CONS),
    }


def _means(stats):
    mean_psi = stats["psi_sum"] / stats["psi_n"] if stats["psi_n"] else 0.0
    scal_n = np.where(stats["scal_n"] > 0, stats["scal_n"], 1.0)
    cons_n = np.where(stats["cons_n"] > 0, stats["cons_n"], 1.0)
    return (
        mean_psi,
        np.where(stats["scal_n"] > 0, stats["scal_sum"] / scal_n, 0.0),
        np.where(stats["cons_n"] > 0, stats["cons_sum"] / cons_n, 0.0),
    )


def _pool(fn, jobs, workers: int, label: str):
    import multiprocessing as mp
    out = [None] * len(jobs)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(fn, job): i for i, job in enumerate(jobs)}
        done = 0
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {label} {done}/{len(jobs)}", flush=True)
    return out


def _report(title: str, res: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}   S = {res['S']:.4f}   ({res['n_shots']} shots)")
    print("=" * 62)
    for label, key, weight in (
        ("R2_psi", "r2_psi", 0.55),
        ("R2_{q95,betaN}", "r2_qb", 0.15),
        ("1 - D_LCFS", "dlcfs", 0.10),
        ("Consistency", "consistency", 0.20),
    ):
        value = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {label:>16s}  {value:8.4f}   x {weight:.2f}  =  {weight * max(0.0, value):.4f}")
    print("  per-derived-scalar R2 (Consistency):")
    for name, value in res["r2_cons_each"].items():
        text = "n/a" if value is None else f"{value:7.4f}"
        print(f"      {name:>8s}  {text}")


def select_rows(cache: Path, split: str) -> pd.DataFrame:
    if split == "score":
        path = cache / "score_subset.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}. Run build_cache.py first.")
        return pd.read_parquet(path)
    shots = pd.read_parquet(cache / "shots.parquet")
    if split == "train_score":
        return nn_common.score_subset(shots, split="train")
    if split not in ("train", "val", "test"):
        raise SystemExit(f"unknown split {split}")
    return shots[shots["split"] == split].sort_values("filename").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=nn_common.CACHE)
    ap.add_argument("--data-root", type=Path, default=nn_common.DATA_ROOT)
    ap.add_argument("--split", default="score", choices=["score", "train_score", "train", "val", "test"])
    ap.add_argument("--mode", default="model", choices=["model", "perfect", "zeros"])
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--pred", type=Path, help="directory of <shot>.npz predictions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--append-metrics", type=Path)
    ap.add_argument("--epoch", type=int)
    args = ap.parse_args()

    rows = select_rows(args.cache, args.split)
    quantize = args.mode == "model"
    pred_root = args.pred
    if args.mode == "model" and args.ckpt is not None:
        pred_root = args.pred or (args.ckpt.parent / f"{args.split}_pred")
        print(f"writing predictions from {args.ckpt} -> {pred_root}", flush=True)
        write_ckpt_predictions(args.ckpt, rows, args.cache, Path(pred_root), args.device, args.data_root)
    elif args.mode == "model" and (pred_root is None or not Path(pred_root).is_dir()):
        raise SystemExit("model mode needs --ckpt or a --pred directory of <shot>.npz files")

    data = args.data_root / "data" / nn_common.TRAIN_CONFIG
    refs = args.cache / "refs"
    print(f"scoring {len(rows)} shots ({args.split}, mode={args.mode}) on {args.workers} processes", flush=True)

    prep_jobs = [(str(data / f"{row.shot}.parquet"), str(refs / f"{row.shot}.pkl")) for row in rows.itertuples(index=False)]
    prepared = _pool(prepare, prep_jobs, args.workers, "references")

    groups = {
        "all": _blank_sums(),
        "normal": _blank_sums(),
        "reversed": _blank_sums(),
    }
    for row, part in zip(rows.itertuples(index=False), prepared):
        _add_sums(groups["all"], part)
        _add_sums(groups["reversed"] if row.s < 0 else groups["normal"], part)
    mean_psi, mean_scal, mean_cons = _means(groups["all"])

    res_jobs = [
        (str(data / f"{row.shot}.parquet"), str(pred_root or ""), row.shot, args.mode, quantize, mean_psi)
        for row in rows.itertuples(index=False)
    ]
    per_shot_ss = _pool(residuals, res_jobs, args.workers, "sign probe")
    totals = {sign: 0.0 for sign in fusion_common.PSI_SIGNS}
    for rr in per_shot_ss:
        for sign in fusion_common.PSI_SIGNS:
            totals[sign] += rr[sign]
    psi_sign = min(fusion_common.PSI_SIGNS, key=lambda sign: totals[sign])
    print(f"  psi_sign = {psi_sign:+.0f}", flush=True)

    # Each group is finalised with its own totals, under the one sign chosen for the whole set.
    means = {
        name: _means(stats) for name, stats in groups.items()
    }
    score_jobs = []
    owners = []
    for row in rows.itertuples(index=False):
        group = "reversed" if row.s < 0 else "normal"
        score_jobs.append((
            str(data / f"{row.shot}.parquet"), str(refs / f"{row.shot}.pkl"),
            str(pred_root or ""), row.shot, args.mode, quantize, psi_sign, means["all"],
        ))
        owners.append(group)
    parts = _pool(score_one, score_jobs, args.workers, "scoring")

    acc = {name: Accum() for name in groups}
    for name in acc:
        acc[name].psi_sign = psi_sign
    for group, part in zip(owners, parts):
        acc["all"].add(part)
        acc[group].add(part)

    results = {name: finalize_machine(acc[name], groups[name]) for name in groups}
    _report(f"{args.split} / {args.mode}", results["all"])
    print(f"\n  S normal   {results['normal']['S']:.4f}   ({results['normal']['n_shots']} shots)")
    print(f"  S reversed {results['reversed']['S']:.4f}   ({results['reversed']['n_shots']} shots)")

    if args.append_metrics is not None:
        res = results["all"]
        nn_common.append_jsonl(args.append_metrics, {
            "epoch": args.epoch,
            "source": "score_train" if args.split == "train_score" else "score",
            "split": args.split,
            "S": res["S"],
            "r2_psi": res["r2_psi"],
            "r2_qb": res["r2_qb"],
            "dlcfs": res["dlcfs"],
            "consistency": res["consistency"],
            "S_normal": results["normal"]["S"],
            "S_reversed": results["reversed"]["S"],
            "r2_cons_each": res["r2_cons_each"],
            "n_shots": res["n_shots"],
        })
        print(f"appended S to {args.append_metrics}", flush=True)

    if args.mode == "perfect" and abs(results["all"]["S"] - 1.0) > 1e-6:
        print(f"HARNESS FAIL: perfect scored S={results['all']['S']}, expected 1", flush=True)
        return 1
    if args.mode == "zeros" and abs(results["all"]["S"] - 0.0) > 1e-6:
        print(f"HARNESS FAIL: zeros scored S={results['all']['S']}, expected 0", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
