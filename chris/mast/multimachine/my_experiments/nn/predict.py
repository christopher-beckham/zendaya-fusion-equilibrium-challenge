"""Run a checkpoint on the public-test parquets and write a submission.

    python my_experiments/nn/predict.py runs/v1/best.pt --parity-check 20
    python my_experiments/nn/predict.py runs/v1/best.pt --out submission --repo USER/repo --read-token hf_...

Shot order is sorted filenames, checked against the Hub stream with infer_mirror_test.check_order.
MAST is predicted by a --feature-set model trained with --decoder-norm physical (predict_mast); any other
checkpoint writes zeros for MAST. ψ is float32 (float16 rounding alone costs about 0.0006 S on DIII-D).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "my_experiments"))

from common import (  # noqa: E402
    CACHE, DATA_ROOT, append_image_features, append_safe_features, as_bool, broadcast_coil_stats,
    IP_COL, check_inputs, checkpoint_pictures, coil_window, stack_history, external_flux, featurize, image_width, inputs_for_shot,
    load_coil_pictures, parse_window_ms, safe_width,
)
from model import load_checkpoint, predict_raw  # noqa: E402
from infer_mirror_test import check_order  # noqa: E402
from validate_submission import validate  # noqa: E402


def _times(path: Path) -> int:
    column = pq.read_table(path, columns=["efit_times"]).column(0).combine_chunks()[0].values
    return len(column)


def _offsets(ckpt: dict) -> np.ndarray:
    return parse_window_ms(str(ckpt.get("args", {}).get("window_ms") or ""))


def _safe(ckpt: dict) -> bool:
    return as_bool(ckpt.get("args", {}).get("extra_safe_features", False))


def _robust(ckpt: dict) -> bool:
    return as_bool(ckpt.get("args", {}).get("robust_inputs", False))


def feature_parity_check(ckpt: dict, cache: Path, data_root: Path, n: int) -> int:
    """Recompute the cached feature columns from the parquet for `n` val shots. Max |Δ| must be < 1e-4."""
    import importlib
    import features_v2
    feature_set = ckpt["norm"]["feature_set"]
    fs = importlib.import_module(f"features_{feature_set}")
    shots = pd.read_parquet(cache / "shots.parquet")
    val = shots[shots["split"] == "val"].sort_values("filename").head(n)
    feat_mm = np.load(cache / f"features_{feature_set}.npy", mmap_mode="r")
    # Every feature version is built from the shared v2 picture pair (build_features.py).
    pics = features_v2.load_pictures(cache / "features_v2" / "pictures_d3d.npz")
    worst = 0.0
    for row in val.itertuples(index=False):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        fresh = fs.shot_features(data_root / "data" / "diii_d_train" / f"{row.shot}.parquet", pics)[:int(row.n)]
        cached = np.asarray(feat_mm[sl])
        same_nan = np.isnan(fresh) == np.isnan(cached)
        if not same_nan.all():
            print(f"  {row.shot}: missing values differ", flush=True)
            return 1
        diff = np.abs(np.nan_to_num(fresh.astype(np.float64)) - np.nan_to_num(cached.astype(np.float64)))
        worst = max(worst, float(diff.max()))
        print(f"  {row.shot}  feature {worst:.3g}", flush=True)
    print(f"parity max |Δ| features {worst:.3g}", flush=True)
    if worst >= 1e-4:
        print("PARITY FAIL: max absolute difference is at least 1e-4", flush=True)
        return 1
    print("parity ok", flush=True)
    return 0


def parity_check(net, norm, ckpt: dict, cache: Path, data_root: Path, device, n: int) -> int:
    """Max |cache − parquet| on the stamp at t, over `n` val shots, must be < 1e-4.

    Without a window, predictions from the cache and from the parquet are compared too.
    With a window, the cache has no neighboring times, so the check is the stamp at t.
    """
    offsets = _offsets(ckpt)
    safe = _safe(ckpt)
    robust = _robust(ckpt)
    shots = pd.read_parquet(cache / "shots.parquet")
    val = shots[shots["split"] == "val"].sort_values("filename").head(n)
    coils_mm = np.load(cache / "coils.npy", mmap_mode="r")
    pictures = load_coil_pictures()
    data = data_root / "data" / "diii_d_train"
    worst = 0.0
    worst_feat = 0.0
    for row in val.itertuples(index=False):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        cached = np.asarray(coils_mm[sl])
        path = data / f"{row.shot}.parquet"
        fresh = inputs_for_shot(path, offsets, cached)
        stamp = fresh[:, :cached.shape[1]]
        diff = np.abs(cached.astype(np.float64) - stamp.astype(np.float64))
        finite = diff[np.isfinite(diff)]
        if finite.size:
            worst_feat = max(worst_feat, float(finite.max()))
        if len(offsets) == 0:
            cached_in, cached_pic = append_image_features(
                append_safe_features(cached, path, safe=safe, robust=robust), path, pictures, robust,
            )
            fresh_in, fresh_pic = append_image_features(
                append_safe_features(fresh, path, safe=safe, robust=robust), path, pictures, robust,
            )
            psi_c, q_c, b_c = predict_raw(net, cached_in, cached_pic, norm, device)
            psi_f, q_f, b_f = predict_raw(net, fresh_in, fresh_pic, norm, device)
            worst = max(
                worst,
                float(np.max(np.abs(psi_c - psi_f))),
                float(np.max(np.abs(q_c - q_f))),
                float(np.max(np.abs(b_c - b_f))),
            )
        pred = f"{worst:.3g}" if len(offsets) == 0 else "n/a"
        print(f"  {row.shot}  feature {worst_feat:.3g}  prediction {pred}", flush=True)
    pred = f"{worst:.3g}" if len(offsets) == 0 else "n/a"
    print(f"parity max |Δ| features {worst_feat:.3g}  predictions {pred}", flush=True)
    if worst >= 1e-4 or worst_feat >= 1e-4:
        print("PARITY FAIL: max absolute difference is at least 1e-4", flush=True)
        return 1
    print("parity ok", flush=True)
    return 0


def _rax_column(feature_set: str) -> int:
    """Column of the unscaled R_ax_th feature in a features cache row, which places the --output-prior prior."""
    import importlib
    return importlib.import_module(f"features_{feature_set}").FEATURES.index("R_ax_th")


def predict_mast(net, norm: dict, paths: list[Path], cache: Path, device, step: int,
                 keep_inputs: bool = False, variant: str = "", split: str = "mast_public_test") -> dict:
    """Predict the MAST public-test shots in `paths` with a --feature-set, --decoder-norm physical model.

    The decoder is pointed at MAST's grid and plasma size (Net.set_machine). Inputs per shot: the cached MAST
    features (same columns as training), stacked with the training history K at `step` frames per offset, the
    MAST coil picture from the mirrored currents, and the mirrored Ip. Returns the submission arrays, keyed as
    in the DIII-D file. With `keep_inputs`, also `<shot>_inputs`: the picture and Ip, for checks.
    """
    import features_v2
    import mast_region
    feature_set = norm["feature_set"]
    # variant "_sol0.5": MAST inputs with the solenoid current halved (build_mast_sol_half.py in the main repo).
    # The MAST region's suffix (mast_region.py): the features depend on its Z0, as does the geometry below.
    stem = f"features_{feature_set}_{split}{variant}{mast_region.suffix()}"
    index = pd.read_parquet(cache / f"{stem}.parquet").set_index("filename")
    feat_mm = np.load(cache / f"{stem}.npy", mmap_mode="r")
    pics = features_v2.load_pictures(cache / "features_v2" / f"pictures_mast{variant}.npz")
    geo = features_v2.geometry("mast")
    net.set_machine(geo["r"], geo["z"], (geo["R0"], geo["Z0"], geo["a"]))
    k_hist = int(norm.get("history_frames", 0))
    cols = norm.get("feature_columns")
    preds = {}
    all_mast = sorted(paths[0].parent.glob("*.parquet"))
    for done, path in enumerate(paths, start=1):
        i = all_mast.index(path)
        row = index.loc[path.name]
        feats = np.array(feat_mm[int(row.start):int(row.start) + int(row.n)], dtype=np.float32, copy=True)
        rax_th = feats[:, _rax_column(feature_set)]
        if cols is not None:
            feats = feats[:, list(cols)]
        feats = stack_history(feats, k_hist, step)
        _, extra = features_v2.shot_features(path, pics, return_inputs=True)
        picture = np.ascontiguousarray(extra["cur"] @ pics["green"], dtype=np.float32)
        ip_ka = np.asarray(extra["ip"], dtype=np.float32)
        t = _times(path)
        if not (len(feats) == len(picture) == len(ip_ka) == t):
            raise SystemExit(f"{path.name}: {len(feats)} feature rows, {len(picture)} pictures, {t} EFIT times")
        if not np.isfinite(picture).all():
            raise SystemExit(f"{path.name}: non-finite MAST coil picture")
        psi, q95, beta = predict_raw(net, feats, picture, norm, device, ip_ka=ip_ka, rax_th=rax_th)
        preds[f"shot_{i:04d}_psirz"] = psi.astype(np.float32)
        preds[f"shot_{i:04d}_q95"] = q95.astype(np.float32)
        preds[f"shot_{i:04d}_betaN"] = beta.astype(np.float32)
        if keep_inputs:
            preds[f"shot_{i:04d}_inputs"] = {"picture": picture, "ip_ka": ip_ka, "shot": path.stem}
        if done % 100 == 0 or done == len(paths):
            print(f"  MAST {done}/{len(paths)}", flush=True)
    return preds


def write_submission(net, norm, ckpt: dict, data_root: Path, out: Path, device, check: int,
                     cache: Path = CACHE, mast_step: int = 4, mast_variant: str | None = None) -> int:
    data = data_root / "data"
    if check:
        print("checking shot order against the Hub stream...", flush=True)
        for config in ("diii_d_public_test", "mast_public_test"):
            check_order(config, check, data_dir=data)

    # MAST inputs: as trained (the checkpoint's --mast-variant) unless overridden.
    variant = ckpt.get("args", {}).get("mast_variant", "") if mast_variant is None else mast_variant
    variant = variant or ""
    print(f"MAST inputs variant: {variant or 'as shipped'}", flush=True)
    import mast_region
    trained = str(ckpt.get("args", {}).get("mast_mask", "envelope"))
    if mast_region.name() != trained:
        raise SystemExit(f"checkpoint trained with MAST region {trained!r}, {mast_region.ENV_VAR} selects "
                         f"{mast_region.name()!r}; unset it or set it to {trained!r}")
    print(f"MAST region: {trained}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((data / "diii_d_public_test").glob("*.parquet"))
    preds = {}
    n_rev = 0
    outside = 0
    n_values = 0
    offsets = _offsets(ckpt)
    safe = _safe(ckpt)
    robust = _robust(ckpt)
    n_tail = image_width(robust) + (safe_width(robust) if safe else 0)
    pictures = checkpoint_pictures(ckpt.get("args", {}))
    feature_set = norm.get("feature_set")
    if feature_set:
        index = pd.read_parquet(cache / f"features_{feature_set}_diii_d_public_test.parquet")
        if list(index["filename"]) != [p.name for p in files]:
            raise SystemExit(f"features_{feature_set}_diii_d_public_test.parquet does not list the test files in order")
        feat_mm = np.load(cache / f"features_{feature_set}_diii_d_public_test.npy", mmap_mode="r")
    coil_lo = np.asarray(ckpt["coil_lo"], dtype=np.float32)
    coil_hi = np.asarray(ckpt["coil_hi"], dtype=np.float32)
    for i, path in enumerate(files):
        feat = featurize(path, offsets if len(offsets) else None)
        coils = coil_window(feat, offsets)
        ip_ka = np.array(coils[:, IP_COL], dtype=np.float32, copy=True)
        if feature_set:
            row = index.iloc[i]
            picture = external_flux(coils, pictures["green"], pictures["ecoil"])
            coils = np.array(feat_mm[int(row.start):int(row.start) + int(row.n)], dtype=np.float32, copy=True)
            rax_th = coils[:, _rax_column(feature_set)]
            if norm.get("feature_columns") is not None:
                coils = coils[:, list(norm["feature_columns"])]
            coils = stack_history(coils, int(norm.get("history_frames", 0)))
            if np.isinf(coils).any():
                raise SystemExit(f"{path.name}: features contain infinities")
        else:
            coils, picture = append_image_features(
                append_safe_features(coils, path, safe=safe, robust=robust), path, pictures, robust,
            )
            try:
                check_inputs(coils, n_tail=n_tail, robust=robust, label=path.name)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        if not np.isfinite(picture).all():
            raise SystemExit(f"{path.name}: non-finite coil picture")
        if coil_lo.shape != (coils.shape[1],) or coil_hi.shape != (coils.shape[1],):
            raise SystemExit(
                f"checkpoint coil range is {coil_lo.shape}, the window is {coils.shape[1]} wide"
            )
        if feature_set:
            mean, std_scale = norm["feat_mean"], norm["feat_std"]
        else:
            mean, std_scale = broadcast_coil_stats(
                norm["coil_mean"], norm["coil_std"], coils.shape[1],
                norm["image_mean"], norm["image_std"],
                norm.get("safe_mean"), norm.get("safe_std"),
            )
        std = np.nan_to_num((coils - mean) / std_scale, nan=0.0)
        outside += int(np.any((std < coil_lo) | (std > coil_hi), axis=1).sum())
        n_values += len(std)
        psi, q95, beta = predict_raw(net, coils, picture, norm, device, ip_ka=ip_ka,
                                     rax_th=rax_th if feature_set else None)
        t = _times(path)
        if len(psi) != t:
            raise SystemExit(f"{path.name}: predicted T={len(psi)} but efit_times has {t}")
        preds[f"shot_{i:04d}_psirz"] = psi.astype(np.float32)
        preds[f"shot_{i:04d}_q95"] = q95.astype(np.float32)
        preds[f"shot_{i:04d}_betaN"] = beta.astype(np.float32)
        n_rev += feat["s"] < 0
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    d3d_path = out / "diii_d_public_test.npz"
    np.savez_compressed(d3d_path, **preds)
    print(f"DIII-D: {len(files)} shots, {n_rev} mirrored -> {d3d_path}", flush=True)
    if n_rev != 76:
        raise SystemExit(f"expected 76 mirrored public-test shots, found {n_rev}")
    print(f"  standardised coils outside the train range: {outside}/{n_values} frames", flush=True)

    mast = sorted((data / "mast_public_test").glob("*.parquet"))
    if net.decoder_norm == "physical" and feature_set:
        preds = predict_mast(net, norm, mast, cache, device, mast_step, variant=variant)
        how = f"model, history step {mast_step}" if norm.get("history_frames") else "model"
    else:
        preds = {}
        for i, path in enumerate(mast):
            t = _times(path)
            preds[f"shot_{i:04d}_psirz"] = np.zeros((t, 65, 65), dtype=np.float32)
            preds[f"shot_{i:04d}_q95"] = np.zeros(t, dtype=np.float32)
            preds[f"shot_{i:04d}_betaN"] = np.zeros(t, dtype=np.float32)
        how = "zeros: only a --feature-set model with --decoder-norm physical predicts MAST"
    mast_path = out / "mast_public_test.npz"
    np.savez_compressed(mast_path, **preds)
    print(f"MAST: {len(mast)} shots ({how}) -> {mast_path}", flush=True)

    rc = validate(d3d_path, "diii_d_public_test", 0)
    rc = validate(mast_path, "mast_public_test", 0) or rc
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, help="directory for the two submission npz files")
    ap.add_argument("--parity-check", type=int, default=0, help="compare cache vs parquet on this many val shots")
    ap.add_argument("--check-order", type=int, default=None, help="Hub rows to compare (default 20 when writing --out)")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repo", help="if set, push the npz files and write a pointer zip")
    ap.add_argument("--read-token")
    ap.add_argument("--pointer", type=Path, default=REPO / "submission_pointer.zip")
    ap.add_argument(
        "--mast-history-step", type=int, default=4,
        help="frames per history offset on MAST. DIII-D trains at 1 frame (20 ms); 4 MAST frames are 20 ms",
    )
    ap.add_argument(
        "--mast-variant", default=None, choices=["", "_sol0.5"],
        help="MAST input files; default: the checkpoint's --mast-variant ('' for older checkpoints)",
    )
    args = ap.parse_args()
    if args.check_order is None:
        args.check_order = 20 if args.out else 0
    if not args.parity_check and args.out is None:
        raise SystemExit("pass --parity-check N and/or --out DIR")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    net, ckpt = load_checkpoint(args.ckpt, device)
    rc = 0
    if args.parity_check and ckpt["norm"].get("feature_set"):
        rc = feature_parity_check(ckpt, args.cache, args.data_root, args.parity_check)
        if rc:
            return rc
    elif args.parity_check:
        rc = parity_check(net, ckpt["norm"], ckpt, args.cache, args.data_root, device, args.parity_check)
        if rc:
            return rc
    if args.out is not None:
        rc = write_submission(net, ckpt["norm"], ckpt, args.data_root, args.out, device, args.check_order,
                              args.cache, args.mast_history_step, args.mast_variant)
        if rc:
            return rc
        if args.repo:
            from push_predictions import push_and_write_pointer
            rc = push_and_write_pointer(args.repo, args.read_token, args.out, args.pointer)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
