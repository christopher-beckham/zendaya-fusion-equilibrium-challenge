"""Training loop for the conv-decoder. Tensors for one split live on the GPU; there is no DataLoader.

Run this on the GPU allocation, not the login node. A full run loads ~20 GB of ψ onto the GPU.

    python my_experiments/nn/train.py --name overfit --overfit-shots 2
    python my_experiments/nn/train.py --name v1 --epochs 40 --device cuda
    python my_experiments/nn/train.py --name v1b-grad0.1-window --epochs 300 --lambda-grad 0.1 \
        --window-ms=-40,-20,-10,-5,0,5,10,20,40
    python my_experiments/nn/train.py --name v1b-grad0.1-window-safe --epochs 300 --lambda-grad 0.1 \
        --window-ms=-10,-5,0,5,10 --extra-safe-features
    python my_experiments/nn/train.py --name robust-win-10-safe --epochs 300 --lambda-grad 0.3 \
        --window-ms=-10,-5,0,5,10 --extra-safe-features --robust-inputs

The decoder always sees the summed coil picture as an input. It is not added to the output. The trunk always
receives the columns read off that picture (18, or 14 with --robust-inputs). A window does not add a
picture per offset.

--robust-inputs scales every input column by its train median and IQR/1.349 instead of mean and std,
drops constant, duplicate and grid-snapped columns, and marks values that cannot be computed (no usable
Thomson in the shot, a single-frame shot) with a 0/1 column. A missing value becomes 0 after scaling.

Every K epochs, predictions for the 100-shot score subset are written and `score.py` is started
in a subprocess. Training does not wait for it. Its line is appended to metrics.jsonl when it
finishes. `best.pt` is chosen by the val proxy 0.55·R²ψ + 0.15·R²qb, which does not need S.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    CACHE, DATA_ROOT, ECOIL_COL, F_SLICE, INPUT_CLIP, IP_COL, LI_FILES, N_COIL, ROBUST_IMAGE_FLAGS, ROBUST_SAFE_FLAGS,
    append_jsonl, broadcast_coil_stats, check_inputs, feature_width, format_window_ms, image_columns, image_width,
    history_stats, inputs_for_shot, parse_window_ms, robust_center_scale, safe_columns, safe_width, score_subset,
    stack_history,
    shot_parquet, tail_widths,
)
import importlib  # noqa: E402

import features_v2  # noqa: E402

from greens import DEFAULT_BASELINE, DEFAULT_FRAMES, load_or_build_baseline  # noqa: E402
from model import (  # noqa: E402
    Net, boundary_error, flux_gradients, flux_scale, grad_mse, li_error, precision_context, unpack_inside,
)

RUNS = HERE / "runs"

# Weights of the boundary terms inside the score-aligned loss, in the units of li_geo["weight"]: per-frame
# loss of S times the number of scored frames. Fitted by non-negative least squares of the scorer's
# per-frame S-loss (kappa, tri_top, tri_bot through (0.20/7)/Var, plus D_LCFS) on the terms, over 7800
# val frames of the 100-shot score subset predicted by img-noadd-robust-win-20-safe-grad0.3-fp32head,
# with the volume term held at its exact weight. The fit is weak: band correlates 0.49 with that loss,
# and refits on half the shots gave W_BAND 10 to 45 and W_XPOINT 0 to 92. Kappa and the triangularities
# are dominated by the boundary's lowest point jumping a grid cell, which no smooth term follows.
W_BAND = 25.7      # per m² of mean squared boundary shift
W_XPOINT = 8.6     # per m² of the X-point term, fitted jointly with W_BAND; it carries 0.13 of that fit


def _rows(shots: pd.DataFrame, split: str) -> pd.DataFrame:
    return shots[shots["split"] == split].sort_values("filename").reset_index(drop=True)


def assert_disjoint(shots: pd.DataFrame) -> None:
    groups = {name: set(shots.loc[shots["split"] == name, "shot"]) for name in ("train", "val", "test")}
    if not groups["train"].isdisjoint(groups["val"]):
        raise AssertionError("train and val shots overlap")
    if not groups["train"].isdisjoint(groups["test"]):
        raise AssertionError("train and test shots overlap")
    if not groups["val"].isdisjoint(groups["test"]):
        raise AssertionError("val and test shots overlap")
    if sum(len(v) for v in groups.values()) != len(shots):
        raise AssertionError("a shot is missing from the train/val/test partition")


def _blocks(mmap, rows: pd.DataFrame):
    for row in rows.itertuples(index=False):
        yield row, np.asarray(mmap[int(row.start):int(row.start) + int(row.n)])


def fit_norm(coils_mm, psi_mm, scal_mm, rows: pd.DataFrame, robust: bool = False) -> dict:
    """Per-pixel ψ mean, one global ψ std, coil StandardScaler, clipped-and-standardised scalars.

    With `robust`, the coil pair is the median and IQR/1.349 instead of mean and std.
    """
    n = int(rows["n"].sum())
    sum_pix = np.zeros(psi_mm.shape[1], dtype=np.float64)
    sum_coil = np.zeros(coils_mm.shape[1], dtype=np.float64)
    scal = np.empty((n, 2), dtype=np.float64)
    cursor = 0
    for row, block in _blocks(psi_mm, rows):
        sum_pix += block.sum(axis=0, dtype=np.float64)
        coils = np.asarray(coils_mm[int(row.start):int(row.start) + int(row.n)], dtype=np.float64)
        sum_coil += coils.sum(axis=0)
        scal[cursor:cursor + int(row.n)] = np.asarray(
            scal_mm[int(row.start):int(row.start) + int(row.n), :2], dtype=np.float64
        )
        cursor += int(row.n)
    pixel_mean = sum_pix / n
    coil_mean = sum_coil / n
    psi_acc = 0.0
    coil_acc = np.zeros_like(coil_mean)
    for row, block in _blocks(psi_mm, rows):
        psi_acc += np.square(block.astype(np.float64) - pixel_mean).sum()
        coils = np.asarray(coils_mm[int(row.start):int(row.start) + int(row.n)], dtype=np.float64)
        coil_acc += np.square(coils - coil_mean).sum(axis=0)
    psi_std = float(np.sqrt(psi_acc / (n * psi_mm.shape[1])))
    coil_std = np.sqrt(coil_acc / n)
    coil_std = np.where(coil_std < 1e-8, 1.0, coil_std)
    if robust:
        coil_mean, coil_std = robust_center_scale(_stack_stamp(rows, coils_mm))

    lo = np.zeros(2, dtype=np.float64)
    hi = np.zeros(2, dtype=np.float64)
    clipped = np.empty_like(scal)
    for j in range(2):
        col = scal[:, j]
        ok = np.isfinite(col)
        lo[j], hi[j] = np.percentile(col[ok], [0.1, 99.9])
        clipped[:, j] = np.clip(col, lo[j], hi[j])
        clipped[~ok, j] = np.nan
    scal_mean = np.nanmean(clipped, axis=0)
    scal_std = np.nanstd(clipped, axis=0)
    scal_std = np.where(scal_std < 1e-8, 1.0, scal_std)
    print(f"  scalar clip q95 [{lo[0]:.3g}, {hi[0]:.3g}]  βN [{lo[1]:.3g}, {hi[1]:.3g}]", flush=True)
    print(f"  ψ global std {psi_std:.4g}", flush=True)
    return {
        "pixel_mean": pixel_mean.astype(np.float32),
        "psi_std": np.float32(psi_std),
        "coil_mean": coil_mean.astype(np.float32),
        "coil_std": coil_std.astype(np.float32),
        "scal_lo": lo.astype(np.float32),
        "scal_hi": hi.astype(np.float32),
        "scal_mean": scal_mean.astype(np.float32),
        "scal_std": scal_std.astype(np.float32),
    }


def split_key(rows: pd.DataFrame) -> str:
    """Identifies a set of shots and their frame counts. A cached statistic is reused only under the same key."""
    text = "\n".join(f"{r.shot}:{int(r.start)}:{int(r.n)}" for r in rows.itertuples(index=False))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cached_stats(path: Path, key: str, fit, save: bool) -> dict:
    """Load the arrays in `path` if they were fitted on the shots with this key, else call `fit` and,
    with `save`, write them. 0-d arrays come back as numpy scalars."""
    if path.exists():
        with np.load(path) as z:
            if str(z["key"]) == key:
                print(f"  loaded {path.name}", flush=True)
                return {k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files if k != "key"}
        print(f"  {path.name} was fitted on other shots; refitting", flush=True)
    stats = fit()
    if save:
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez(tmp, key=np.array(key), **stats)
        tmp.replace(path)
        print(f"  saved {path.name}", flush=True)
    return stats


def _stack_stamp(rows: pd.DataFrame, coils_mm) -> np.ndarray:
    """Cached currents at time t, in `rows` order. Shape (frames, 21)."""
    n = int(rows["n"].sum())
    out = np.empty((n, N_COIL), dtype=np.float32)
    cursor = 0
    for row in rows.itertuples(index=False):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        block = np.asarray(coils_mm[sl, :N_COIL], dtype=np.float32)
        out[cursor:cursor + len(block)] = block
        cursor += len(block)
    if cursor != n:
        raise RuntimeError(f"stamp stack has {cursor} frames, the split has {n}")
    return out


def _fit_column_norm(features: np.ndarray, n_cols: int, label: str, n_flags: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Training mean and std of one appended block.

    With `n_flags` set (the robust layout), median and IQR/1.349 ignoring NaN, and the last
    `n_flags` columns are 0/1 flags left unscaled.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.shape[1] != n_cols:
        raise ValueError(f"expected {n_cols} {label} columns, got {values.shape[1]}")
    if n_flags is not None:
        return robust_center_scale(values, flag_cols=range(n_cols - n_flags, n_cols))
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} features are not finite")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def assemble_safe(rows: pd.DataFrame, data_root: Path, robust: bool = False) -> np.ndarray:
    """The shared columns in `rows` order."""
    n = int(rows["n"].sum())
    out = np.empty((n, safe_width(robust)), dtype=np.float32)
    cursor = 0
    for i, row in enumerate(rows.itertuples(index=False)):
        block = safe_columns(shot_parquet(data_root, row), robust)
        if len(block) != int(row.n):
            raise RuntimeError(
                f"{row.shot}: safe block has {len(block)} frames, the cache has {int(row.n)}"
            )
        out[cursor:cursor + len(block)] = block
        cursor += len(block)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  safe {i + 1}/{len(rows)} shots", flush=True)
    if cursor != n:
        raise RuntimeError(f"safe block has {cursor} frames, the split has {n}")
    return out


def assemble_image(rows: pd.DataFrame, coils_mm, data_root: Path, pictures: dict,
                   robust: bool = False) -> np.ndarray:
    """The coil-picture columns in `rows` order."""
    n = int(rows["n"].sum())
    out = np.empty((n, image_width(robust)), dtype=np.float32)
    cursor = 0
    for i, row in enumerate(rows.itertuples(index=False)):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        stamp = np.asarray(coils_mm[sl, :N_COIL])
        block = image_columns(shot_parquet(data_root, row), stamp, pictures["green"], pictures["ecoil"], robust)
        if len(block) != int(row.n):
            raise RuntimeError(
                f"{row.shot}: image block has {len(block)} frames, the cache has {int(row.n)}"
            )
        out[cursor:cursor + len(block)] = block
        cursor += len(block)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  image {i + 1}/{len(rows)} shots", flush=True)
    if cursor != n:
        raise RuntimeError(f"image block has {cursor} frames, the split has {n}")
    return out


def assemble_window(rows: pd.DataFrame, coils_mm, data_root: Path, offsets: np.ndarray) -> np.ndarray:
    """Unstandardised window, one row per picture, in `rows` order. The stamp at t must match the cache."""
    n = int(rows["n"].sum())
    out = np.empty((n, N_COIL * len(offsets)), dtype=np.float32)
    cursor = 0
    for i, row in enumerate(rows.itertuples(index=False)):
        sl = slice(int(row.start), int(row.start) + int(row.n))
        block = inputs_for_shot(shot_parquet(data_root, row), offsets, np.asarray(coils_mm[sl]))
        out[cursor:cursor + len(block)] = block
        cursor += len(block)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  window {i + 1}/{len(rows)} shots", flush=True)
    if cursor != n:
        raise RuntimeError(f"window has {cursor} frames, the split has {n}")
    return out


def _polygon_volume(contour: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """The scorer's volume, 2π |first moment about Z|, of (N, K, 2) (R, Z) polygons. As derive._shape_from_contour."""
    out = np.empty(len(contour), dtype=np.float64)
    for start in range(0, len(contour), chunk):
        c = np.asarray(contour[start:start + chunk], dtype=np.float64)
        r, z = c[..., 0], c[..., 1]
        r1, z1 = np.roll(r, -1, axis=1), np.roll(z, -1, axis=1)
        moment = ((r + r1) * (r * z1 - r1 * z)).sum(axis=1) / 6.0
        out[start:start + chunk] = 2.0 * np.pi * np.abs(moment)
    return out


def load_li_geometry(cache: Path, rows: pd.DataFrame, device) -> dict:
    """True boundary, inside-region and li for the frames of `rows`, in bundle order, from li_geometry.py.

    Frames the precompute skipped (its --stride) or could not use get valid = False and no li, band,
    X-point or volume loss.
    The weights are the score's: li and volume are two of seven consistency terms worth 0.20 of S, and
    each R² divides the squared error by the variance of the true value. The volume variance is taken
    from the 128-point boundaries; on val it differs from the scorer's 512-point value by under 1%.
    """
    paths = {key: cache / name for key, name in LI_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"the boundary losses need the li geometry: {missing}. Run my_experiments/nn/li_geometry.py")
    index = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n)) for r in rows.itertuples(index=False)])
    status = np.asarray(np.load(paths["status"], mmap_mode="r")[index])
    valid = status == 1
    if not valid.any():
        raise SystemExit("no train frame has li geometry. Run my_experiments/nn/li_geometry.py --split train")
    li = np.asarray(np.load(paths["li"], mmap_mode="r")[index], dtype=np.float64)
    var = float(li[valid].var())
    contour = np.asarray(np.load(paths["contour"], mmap_mode="r")[index], dtype=np.float32)
    inside = np.asarray(np.load(paths["inside"], mmap_mode="r")[index], dtype=np.uint8)
    vol_var = float(_polygon_volume(contour[valid]).var())
    print(f"  li geometry: {int(valid.sum())} of {len(index)} train frames, li std {np.sqrt(var):.4f}, "
          f"volume std {np.sqrt(vol_var):.3f} m³", flush=True)
    return {
        "valid": torch.from_numpy(valid).to(device),
        "contour": torch.from_numpy(contour).to(device),
        "inside": torch.from_numpy(inside).to(device),
        "weight": (0.20 / 7.0) / var,
        "var": var,
        "vol_weight": (0.20 / 7.0) / vol_var,
        "vol_var": vol_var,
    }



def _with_history(inputs: np.ndarray, rows: pd.DataFrame, k: int) -> np.ndarray:
    """Apply common.stack_history shot by shot. `inputs` is in `rows` order, `n` frames per shot."""
    if k <= 0:
        return inputs
    out, cursor = [], 0
    for row in rows.itertuples(index=False):
        n = int(row.n)
        out.append(stack_history(inputs[cursor:cursor + n], k))
        cursor += n
    if cursor != len(inputs):
        raise RuntimeError(f"history: {cursor} frames in rows, {len(inputs)} inputs")
    return np.concatenate(out)


def surrogate_scale(epoch: int, alpha: float, start: int, ramp: int) -> float:
    """alpha from epoch `start`, reached linearly over `ramp` epochs. 0 before `start`."""
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return alpha
    return alpha * min(1.0, (epoch - start + 1) / ramp)


def load_split(coils_mm, psi_mm, scal_mm, phase_mm, rows, norm, device, keep_raw_scalars: bool,
               coil_rows: np.ndarray | None = None):
    """Copy one split onto `device`. Coils are standardised. ψ is normalised. Scalars too, if not raw.

    `coil_rows`, when set, is the unstandardised input for this split in row order. It replaces
    the cached coil columns. The 21 per-coil mean/std are repeated once per stamp.
    """
    n = int(rows["n"].sum())
    n_in = coils_mm.shape[1] if coil_rows is None else coil_rows.shape[1]
    if coil_rows is not None and coil_rows.shape[0] != n:
        raise ValueError(f"coil window has {coil_rows.shape[0]} rows, the split has {n}")
    coils = torch.empty((n, n_in), dtype=torch.float32, device=device)
    stamp = torch.empty((n, N_COIL), dtype=torch.float32, device=device)
    psi = torch.empty((n, psi_mm.shape[1]), dtype=torch.float32, device=device)
    phase = torch.empty(n, dtype=torch.int8, device=device)
    sign = torch.empty(n, dtype=torch.float32, device=device)
    # The unscaled R_ax_th feature per frame, which places the plasma prior of --output-prior. NaN when unused.
    rax = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    rax_col = norm.get("rax_col")
    raw = torch.empty((n, 2), dtype=torch.float32, device=device) if keep_raw_scalars else None
    scal = torch.empty((n, 2), dtype=torch.float32, device=device)
    feature_set = bool(norm.get("feature_set"))
    if feature_set:
        c_mean, c_std = norm["feat_mean"], norm["feat_std"]
        n_safe, n_image = 0, 0
    else:
        c_mean, c_std = broadcast_coil_stats(
            norm["coil_mean"], norm["coil_std"], n_in,
            norm["image_mean"], norm["image_std"],
            norm.get("safe_mean"), norm.get("safe_std"),
        )
        n_safe, n_image = tail_widths(norm)
    p_mean = norm["pixel_mean"]
    p_std = float(norm["psi_std"])
    s_mean = norm["scal_mean"]
    s_std = norm["scal_std"]
    lo, hi = norm["scal_lo"], norm["scal_hi"]
    robust = bool(norm.get("robust_inputs", False))
    cursor = 0
    spans = []
    coil_lo = np.full(n_in, np.inf, dtype=np.float64)
    coil_hi = np.full(n_in, -np.inf, dtype=np.float64)
    for row, psi_block in _blocks(psi_mm, rows):
        k = int(row.n)
        sl = slice(int(row.start), int(row.start) + k)
        if coil_rows is None:
            raw_coils = np.asarray(coils_mm[sl], dtype=np.float32)
        else:
            raw_coils = coil_rows[cursor:cursor + k]
        coil_block = (raw_coils - c_mean) / c_std
        if feature_set:
            # NaN marks a value features_v2 could not compute; its flag column says so. 0 is the train median.
            if np.isinf(raw_coils).any():
                raise RuntimeError(f"{row.shot}: features contain infinities")
            coil_block = np.clip(np.nan_to_num(coil_block, nan=0.0), -INPUT_CLIP, INPUT_CLIP)
        elif robust:
            # NaN is allowed only in the flagged safe and image columns, as in predict.py.
            # A missing coil current has no flag, so it stops training instead of becoming 0.
            check_inputs(raw_coils, n_tail=n_safe + n_image, robust=True, label=str(row.shot))
            # A missing value is NaN until here. 0 is the train median.
            coil_block = np.nan_to_num(coil_block, nan=0.0)
        coil_lo = np.minimum(coil_lo, coil_block.min(axis=0))
        coil_hi = np.maximum(coil_hi, coil_block.max(axis=0))
        if n_image:
            coil_block[:, -n_image:] = np.clip(coil_block[:, -n_image:], -INPUT_CLIP, INPUT_CLIP)
        if n_safe:
            safe_cols = slice(-(n_safe + n_image), -n_image)
            coil_block[:, safe_cols] = np.clip(coil_block[:, safe_cols], -INPUT_CLIP, INPUT_CLIP)
        coils[cursor:cursor + k] = torch.from_numpy(np.array(coil_block, dtype=np.float32, copy=True))
        # The cached currents at t, which are also the first stamp of a window, drive the coil picture.
        stamp[cursor:cursor + k] = torch.from_numpy(np.array(coils_mm[sl, :N_COIL], dtype=np.float32, copy=True))
        psi[cursor:cursor + k] = torch.from_numpy(np.array(
            (psi_block.astype(np.float32) - p_mean) / np.float32(p_std), dtype=np.float32, copy=True
        ))
        # The phase memmap is read-only. A view of it is not writable, and torch refuses that.
        phase[cursor:cursor + k] = torch.from_numpy(np.array(phase_mm[sl], dtype=np.int8, copy=True))
        sign[cursor:cursor + k] = float(row.s)
        if rax_col is not None:
            rax[cursor:cursor + k] = torch.from_numpy(np.array(raw_coils[:, rax_col], dtype=np.float32, copy=True))
        raw_block = np.array(scal_mm[sl, :2], dtype=np.float32, copy=True)
        if raw is not None:
            raw[cursor:cursor + k] = torch.from_numpy(raw_block)
        clipped = np.clip(raw_block, lo, hi)
        clipped = (clipped - s_mean) / s_std
        clipped[~np.isfinite(raw_block)] = 0.0
        scal[cursor:cursor + k] = torch.from_numpy(np.array(clipped, dtype=np.float32, copy=True))
        spans.append({"shot": row.shot, "start": cursor, "n": k, "s": float(row.s)})
        cursor += k
    bundle = {
        "coils": coils, "stamp": stamp, "psi": psi, "scal": scal, "raw": raw,
        "phase": phase, "sign": sign, "spans": spans, "rax": rax,
    }
    return bundle, coil_lo.astype(np.float32), coil_hi.astype(np.float32)


class _Sum:
    def __init__(self, device):
        z = torch.zeros((), dtype=torch.float64, device=device)
        self.ss, self.sum, self.sumsq = z.clone(), z.clone(), z.clone()
        self.n = 0

    def add(self, err2, values):
        if values.numel() == 0:
            return
        self.ss = self.ss + err2.sum(dtype=torch.float64)
        self.sum = self.sum + values.sum(dtype=torch.float64)
        self.sumsq = self.sumsq + values.square().sum(dtype=torch.float64)
        self.n += int(values.numel())


def _r2(acc: _Sum) -> float:
    if acc.n <= 0:
        return float("nan")
    tot = acc.sumsq - acc.sum * acc.sum / acc.n
    if float(tot) <= 0:
        return float("nan")
    return float(1.0 - acc.ss / tot)


def scheduled_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _external(stamp: torch.Tensor, green: torch.Tensor, ecoil: torch.Tensor) -> torch.Tensor:
    """Physical vacuum flux (B, 4225) from the stamp at time t."""
    return stamp[:, F_SLICE].float() @ green + stamp[:, ECOIL_COL:ECOIL_COL + 1].float() * ecoil


def evaluate(net, batch, norm, bundle, device, green, ecoil) -> dict:
    """Pooled R² on denormalised ψ and on the raw scalar labels, plus RMSE of ∇ψ."""
    pixel_mean = torch.as_tensor(norm["pixel_mean"], device=device)
    psi_std = float(norm["psi_std"])
    scal_mean = torch.as_tensor(norm["scal_mean"], device=device)
    scal_std = torch.as_tensor(norm["scal_std"], device=device)
    groups = {name: _Sum(device) for name in ("all", "normal", "reversed", "rampup", "flattop", "rampdown")}
    scalars = {name: _Sum(device) for name in ("q95", "betaN")}
    grad_ss = torch.zeros((), dtype=torch.float64, device=device)
    grad_n = 0
    use_amp = device.type == "cuda" and bool(norm.get("amp", True))
    n = bundle["coils"].shape[0]
    net.eval()
    with torch.no_grad():
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            picture = _external(bundle["stamp"][start:stop], green, ecoil)
            with precision_context(device.type, use_amp):
                psi_n, scal_n = net(bundle["coils"][start:stop], picture, bundle["stamp"][start:stop, IP_COL],
                                    bundle["rax"][start:stop])
            pred = psi_n.float() * psi_std + pixel_mean
            true = bundle["psi"][start:stop] * psi_std + pixel_mean
            err2 = (pred - true).square()
            sign = bundle["sign"][start:stop]
            phase = bundle["phase"][start:stop]
            masks = {
                "all": None,
                "normal": sign > 0,
                "reversed": sign < 0,
                "rampup": phase == 0,
                "flattop": phase == 1,
                "rampdown": phase == 2,
            }
            for name, mask in masks.items():
                if mask is None:
                    groups[name].add(err2, true)
                else:
                    groups[name].add(err2[mask], true[mask])
            pred_s = scal_n.float() * scal_std + scal_mean
            raw = bundle["raw"][start:stop]
            for j, name in enumerate(("q95", "betaN")):
                ok = torch.isfinite(raw[:, j])
                diff = (raw[ok, j] - pred_s[ok, j]).square()
                scalars[name].add(diff, raw[ok, j])
            dpr, dpz = flux_gradients(pred, net.z_coord, net.r_coord)
            dtr, dtz = flux_gradients(true, net.z_coord, net.r_coord)
            grad_ss = grad_ss + (dpr - dtr).square().sum(dtype=torch.float64)
            grad_ss = grad_ss + (dpz - dtz).square().sum(dtype=torch.float64)
            grad_n += int(dpr.numel() + dpz.numel())
    r2_psi = {name: _r2(acc) for name, acc in groups.items()}
    r2_q = _r2(scalars["q95"])
    r2_b = _r2(scalars["betaN"])
    r2_qb = float(np.nanmean([r2_q, r2_b]))
    rmse = float(torch.sqrt(grad_ss / max(grad_n, 1)))
    proxy = 0.55 * max(0.0, r2_psi["all"]) + 0.15 * max(0.0, r2_qb)
    return {
        "r2_psi": r2_psi["all"],
        "r2_psi_normal": r2_psi["normal"],
        "r2_psi_reversed": r2_psi["reversed"],
        "r2_psi_rampup": r2_psi["rampup"],
        "r2_psi_flattop": r2_psi["flattop"],
        "r2_psi_rampdown": r2_psi["rampdown"],
        "r2_q95": r2_q,
        "r2_betaN": r2_b,
        "r2_qb": r2_qb,
        "rmse_grad": rmse,
        "proxy": proxy,
    }


def _save(path, net, norm, args, epoch, proxy, coil_lo, coil_hi):
    torch.save({
        "model": net.state_dict(),
        "norm": norm,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "epoch": epoch,
        "proxy": proxy,
        "coil_lo": coil_lo,
        "coil_hi": coil_hi,
        "picture_added": False,
    }, path)


def _dump_subset(net, bundle, norm, subset_shots, out_dir: Path, device, batch: int,
                 green, ecoil) -> None:
    """Write per-shot predictions for the score subset. ψ is float16, as in a submission."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_mean = torch.as_tensor(norm["pixel_mean"], device=device)
    psi_std = float(norm["psi_std"])
    scal_mean = torch.as_tensor(norm["scal_mean"], device=device)
    scal_std = torch.as_tensor(norm["scal_std"], device=device)
    want = set(subset_shots)
    use_amp = device.type == "cuda" and bool(norm.get("amp", True))
    net.eval()
    with torch.no_grad():
        for span in bundle["spans"]:
            if span["shot"] not in want:
                continue
            sl = slice(span["start"], span["start"] + span["n"])
            psi_parts, q_parts, b_parts = [], [], []
            for start in range(0, span["n"], batch):
                stop = min(start + batch, span["n"])
                picture = _external(bundle["stamp"][sl][start:stop], green, ecoil)
                with precision_context(device.type, use_amp):
                    psi_n, scal_n = net(bundle["coils"][sl][start:stop], picture, bundle["stamp"][sl][start:stop, IP_COL],
                                        bundle["rax"][sl][start:stop])
                psi = (psi_n.float() * psi_std + pixel_mean).view(-1, 65, 65).cpu().numpy()
                scal = (scal_n.float() * scal_std + scal_mean).cpu().numpy()
                psi_parts.append(psi)
                q_parts.append(scal[:, 0])
                b_parts.append(scal[:, 1])
            np.savez(
                out_dir / f"{span['shot']}.npz",
                psirz=np.concatenate(psi_parts).astype(np.float16),
                q95=np.concatenate(q_parts).astype(np.float32),
                betaN=np.concatenate(b_parts).astype(np.float32),
            )


def _argv_with_window(argv: list[str]) -> list[str]:
    """Attach a leading-minus offset list to --window-ms.

    Argparse treats a separate token that starts with '-' as a new flag, so
    `--window-ms -40,-20,0` is read as a missing value. `--window-ms=-40,-20,0`
    is already one token and is left alone.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--window-ms" and i + 1 < len(argv):
            out.append("--window-ms=" + argv[i + 1])
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="default 40, or 400 with --overfit-shots")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument(
        "--stop-epoch", type=int, default=0,
        help="stop after this epoch; the LR schedule still spans --epochs. 0 runs all --epochs",
    )
    ap.add_argument("--lambda-s", type=float, default=1.0)
    ap.add_argument("--lambda-grad", type=float, default=0.0)
    ap.add_argument(
        "--extra-safe-features",
        action="store_true",
        help="append 11 columns shared by DIII-D and MAST: log10 peak electron pressure, "
             "the outboard pressure shape, up-down shaping current over Ip, and total "
             "shaping current over Ip. The flux output is unchanged.",
    )
    ap.add_argument(
        "--robust-inputs",
        action="store_true",
        help="scale every input column by train median and IQR/1.349, drop constant, duplicate and "
             "grid-snapped columns, use |q_cyl| and a 100 kA Ip floor, and add 0/1 columns for "
             "values that cannot be computed. Missing values become 0 after scaling.",
    )
    ap.add_argument(
        "--window-ms",
        default="",
        help="comma-separated millisecond offsets around each picture, including 0. "
             "A list that starts with a minus must be attached with '=', "
             "as in --window-ms=-40,-20,0,20,40. "
             "Empty uses the 21 coil readings at that picture only.",
    )
    ap.add_argument(
        "--feature-set",
        choices=["", "v2", "v3", "v4", "v5", "v6"],
        default="",
        help="v2 or v3: the trunk sees only the cached features_v2 or features_v3 columns, which mean the same on DIII-D and MAST "
             "(build_features.py), scaled by train median and IQR/1.349, NaN to 0, clipped to ±5. No coil "
             "current enters the trunk. The decoder still sees the coil picture. Excludes --window-ms, "
             "--extra-safe-features and --robust-inputs",
    )
    ap.add_argument(
        "--zoom-aug", action="store_true",
        help="with --decoder-norm physical: draw each training example on its own grid, zoomed and shifted about "
             "DIII-D's (zoom.py), with the coil picture, prior and coordinates of that grid and the loss masked to "
             "pixels inside DIII-D's grid. Validation stays on the native grid",
    )
    ap.add_argument("--zoom-r", default="0.8,1.5", help="range of the R zoom factor")
    ap.add_argument("--zoom-z", default="0.8,1.6", help="range of the Z zoom factor")
    ap.add_argument("--zoom-shift", default="0,0.35", help="range of the outward R shift, in units of a")
    ap.add_argument("--zoom-native-frac", type=float, default=0.25, help="share of examples kept on the native grid")
    ap.add_argument(
        "--lambda-current", type=float, default=0.0,
        help="with --decoder-norm physical: weight of the label-free current-consistency loss (physics_loss.py) on "
             "DIII-D training frames (native-grid examples only under --zoom-aug)")
    ap.add_argument(
        "--mast-current", type=float, default=0.0,
        help="with --decoder-norm physical and --output-prior: weight of the same loss on MAST public-test inputs, "
             "drawn each step (inputs only, no MAST label; needs <cache>/mast_inputs.npz from build_mast_inputs.py)")
    ap.add_argument("--mast-batch-size", type=int, default=512, help="MAST input frames per step for --mast-current")
    ap.add_argument(
        "--lambda-axis", type=float, default=0.0,
        help="with --decoder-norm physical: weight of a one-sided pull of the map's axis towards the Thomson pressure "
             "peak (physics_loss.AxisPull), on DIII-D frames and, with --mast-current, on the MAST input batch")
    ap.add_argument(
        "--lambda-shape", type=float, default=0.0,
        help="weight of a flux-span-normalised shape loss (S_SURROGATES.md, proposal 2, Ip version): per frame, the "
             "ψ error with its frame mean removed, mean-squared, times (median |Ip| / max(|Ip|, 5th pct |Ip|))². "
             "Low-current frames (ramp-up, termination) count for the derived scalars as much as full-current ones; "
             "the plain ψ MSE gives them little weight. Native-grid examples only under --zoom-aug")
    ap.add_argument(
        "--lambda-te", type=float, default=0.0,
        help="with --decoder-norm physical: weight of the Te-order loss (physics_loss.TeOrder: hotter Thomson points on "
             "flux surfaces nearer the axis), on DIII-D frames and, with --mast-current, on the MAST input batch. "
             "Needs <cache>/thomson_raw_{d3d,mast}.npz from build_thomson_raw.py")
    ap.add_argument("--mast-axis-hi", type=float, default=0.08,
                    help="with --lambda-axis and --mast-current: outboard slack of the MAST axis pull, (R_axis − R_th) / a "
                         "(DIII-D keeps 0.08). With --lambda-fluxfn the MAST axis drifts to the slack")
    ap.add_argument("--axis-z-slack", type=float, default=0.0,
                    help="with --lambda-axis: also penalise |Z_axis − Z0| / a beyond this (0 = off; 0.25 suggested)")
    ap.add_argument("--order-scale-free", action="store_true",
                    help="divide each frame's ordering gaps by its φ range along the Thomson channels")
    ap.add_argument("--order-quantities", default="te",
                    help="comma list of Thomson quantities for the --lambda-te ordering loss: te, ne, p (= ne·Te). "
                         "Each gets the full --lambda-te weight; ne and p need build_thomson_raw.py --quantity ne")
    ap.add_argument("--mast-history-step", type=int, default=4,
                    help="MAST frames per history offset in the --mast-current batch (5 ms frames; 4 = 20 ms, as on "
                         "DIII-D) with --history-frames > 0")
    ap.add_argument(
        "--synth", action="append", default=[], metavar="TAG[:WEIGHT]",
        help="labelled synthetic set from synth_cache.py (<cache>/synth/TAG, or a folder path), weight 1 by default; "
             "repeat for several sets. Each step adds WEIGHT × the scaled flux error on a homogeneous batch of that set "
             "(synth_data.py). Needs --decoder-norm physical. The held-out part is scored every epoch")
    ap.add_argument("--no-mast-monitor", action="store_true",
                    help="skip the per-epoch label-free MAST monitor (mast_monitor.py; on by default with "
                         "--decoder-norm physical)")
    ap.add_argument("--mast-checks-every", type=int, default=0,
                    help="at scoring epochs divisible by this, run the scorer-based MAST checks (shape scalars, "
                         "Thomson edges and ordering, li) on that checkpoint in the background -> mast_checks_epXXX.log")
    ap.add_argument("--synth-batch", type=int, default=256, help="frames per synthetic batch, per set and step")
    ap.add_argument(
        "--mast-variant", default="", choices=["", "_sol0.5"],
        help="MAST input files for --mast-current: '' as shipped, '_sol0.5' with the solenoid current halved "
             "(organisers' note: magnetics_sol_current reads 2x the FAIR-MAST level-2 value; build_mast_sol_half.py)")
    ap.add_argument(
        "--zoom-default-weight", type=float, default=0.0,
        help="with --zoom-aug and --output-prior: weight of a loss pulling the network's correction "
             "(ψ − coil picture − prior) / (μ0 R0 Ip / 2π) towards 0 on pixels with no EFIT target",
    )
    ap.add_argument(
        "--output-prior", action="store_true",
        help="with --decoder-norm physical: ψ = coil picture + plasma prior + network output × μ0 R0 Ip / 2π. The prior "
             "is Ip spread over an ellipse at the Thomson axis (model.plasma_prior_table); it is not an input",
    )
    ap.add_argument(
        "--decoder-norm", choices=["target", "physical"], default="target",
        help="physical: the decoder's picture is ψ_ext / (μ0 R0 Ip / 2π), its coordinates are (R − R0)/a, "
             "(Z − Z0)/a and R/R0, and its output is ψ in units of μ0 R0 Ip / 2π. No training statistic enters "
             "what the decoder sees or emits. target: the earlier per-pixel ψ mean and std (model.Net)",
    )
    ap.add_argument(
        "--decoder", choices=["conv", "linear"], default="conv",
        help="linear: the conv decoder emits a current map J on the envelope cells (non-negative, Σ J = −Ip) and "
             "ψ = coil picture + G · J + one offset per frame (lindec.py). Needs --decoder-norm physical; excludes "
             "--output-prior and --zoom-aug. conv: the decoder emits ψ (model.Net)")
    ap.add_argument("--lindec-erode", type=int, default=0,
                    help="with --decoder linear: keep J off envelope cells within this many cells of its edge")
    ap.add_argument("--lindec-act", choices=["softplus", "exp"], default="softplus",
                    help="with --decoder linear: map from the decoder output to current weights (exp = softmax)")
    ap.add_argument("--lindec-exact-ip", action="store_true",
                    help="with --decoder linear: force Σ J = −Ip exactly instead of learning a per-frame total-current "
                         "gain (exact Ip caps the decoder at Consistency 0.875 on the true current, the gain at 0.955)")
    ap.add_argument("--lindec-exact-ip-mast", action="store_true",
                    help="with --decoder linear: Σ J = −Ip exactly on MAST (the total-current gain is used on DIII-D only)")
    ap.add_argument("--lindec-ip-floor", type=float, default=0.0,
                    help="with --decoder linear: below this |Ip| (kA) a learned current of up to ± this much is added "
                         "to the measured Ip, fading to zero at the floor (lindec.py); 0 = off")
    ap.add_argument(
        "--lambda-fluxfn", type=float, default=0.0,
        help="with --decoder linear: weight of the label-free flux-function loss (fluxfn_loss.py) on DIII-D and, with "
             "--mast-current, MAST batches: the share of the current map not explained as a smooth function of the "
             "map's own flux (R and 1/R terms), current beyond the first surface touching the envelope included")
    ap.add_argument("--fluxfn-out-weight", type=float, default=0.0,
                    help="with --lambda-fluxfn: also add this weight times the share of Σ |J| beyond the first surface "
                         "touching the envelope (the Σ J² share alone barely sees a thin halo over many cells)")
    ap.add_argument("--fluxfn-cold-te", type=float, default=0.0,
                    help="with --lambda-fluxfn: the boundary is also the first surface through a Thomson channel with "
                         "Te at or below this (eV); 0 = off. Sets the plasma's size, which the loss alone leaves free")
    ap.add_argument("--fluxfn-pressure", type=float, default=0.0,
                    help="with --lambda-fluxfn: weight (inside the loss) of the Thomson pressure term, the share of "
                         "p_e = ne·Te not explained by the pressure the fit implies, P(u) = ∫_u^1 R·p' coefficients")
    ap.add_argument("--fluxfn-edge-ext", type=float, default=0.0,
                    help="with --lambda-fluxfn: the fitted profiles may carry current to u = 1 + this beyond the first "
                         "touching surface (0 forces it to zero there, which raises DIII-D li ~+0.02)")
    ap.add_argument("--fluxfn-spread", type=float, default=0.0,
                    help="with --lambda-fluxfn: weight (inside the loss) of relu(floor − current spread / a)², against the "
                         "MAST collapse of the current to a filament")
    ap.add_argument("--fluxfn-spread-floor", type=float, default=0.39)
    ap.add_argument("--mast-jitter", type=float, default=0.0,
                    help="with --decoder linear and --mast-current: weight of the frame-to-frame smoothness of the MAST "
                         "current centroid, |c_t − (c_{t−1} + c_{t+1}) / 2|² / a², on batches of consecutive-frame "
                         "triplets (--mast-batch-size frames in all)")
    ap.add_argument("--fluxfn-frames", type=int, default=512,
                    help="with --lambda-fluxfn: DIII-D frames per batch it is computed on (the first of the batch)")
    ap.add_argument(
        "--lambda-jcur", type=float, default=0.0,
        help="with --decoder linear: weight of an auxiliary loss on the current map, Σ over envelope cells of "
             "((J − J_true) / Ip)², J_true = −Δ*ψ_true / (μ0 R) · ΔR ΔZ from the target, frames with |Ip| ≥ 150 kA")
    ap.add_argument(
        "--history-frames", type=int, default=0,
        help="with --feature-set, also feed the features of the K EFIT frames before and after each frame "
             "(common.stack_history): (2K + 1) stamps and 2K off-the-shot flags. Frames, not milliseconds",
    )
    ap.add_argument(
        "--drop-features", default="",
        help="comma-separated --feature-set columns to leave out of the trunk input",
    )
    ap.add_argument(
        "--pictures", type=Path, default=DEFAULT_BASELINE,
        help="coil pictures for the decoder. baseline_pictures_ecoil135.npz (features_v2.write_decoder_pictures) "
             "scales ECOILA by features_v2.D3D_ECOIL_SCALE; the default leaves it at one turn",
    )
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="train and evaluate in float32 without bf16 autocast or TF32. Slower. The final conv "
             "and scalar head are float32 either way.",
    )
    ap.add_argument(
        "--loss-li",
        action="store_true",
        help="add the score's li term to the loss: li computed on the predicted ψ over the true boundary, "
             "weighted (0.20/7)/Var(li) and scaled by --surrogate-alpha. Needs li_geometry.py output",
    )
    ap.add_argument(
        "--loss-band", action="store_true",
        help="add the mean squared first-order shift of the scorer's boundary, relative to the flux error at "
             "the X-point, weighted W_BAND and scaled by --surrogate-alpha. Needs li_geometry.py output",
    )
    ap.add_argument(
        "--loss-xpoint", action="store_true",
        help="add the flux error within two cells of the X-point relative to the X-point node, over the "
             "boundary's 5th-percentile |∇ψ| squared, weighted W_XPOINT and scaled by --surrogate-alpha",
    )
    ap.add_argument(
        "--loss-volume", action="store_true",
        help="add the score's volume term: the first-order volume change ∮2πR·δn dl, weighted "
             "(0.20/7)/Var(volume) and scaled by --surrogate-alpha",
    )
    ap.add_argument(
        "--surrogate-alpha", type=float, default=3.0,
        help="scale of the score-aligned terms against the existing loss. About 3 matches the existing "
             "loss's own exchange rate with S",
    )
    ap.add_argument("--surrogate-start-epoch", type=int, default=1, help="first epoch with the score-aligned terms")
    ap.add_argument("--surrogate-ramp-epochs", type=int, default=0, help="ramp alpha in linearly over this many epochs")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping; the logged norm is pre-clip")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument(
        "--seed-channels", type=int, default=256,
        help="channels of the 8×8 grid the trunk vector is projected onto. That projection is most of "
             "the parameters; this shrinks it without changing the trunk",
    )
    ap.add_argument("--scalar-width", type=int, default=128)
    ap.add_argument("--scalar-depth", type=int, default=2)
    ap.add_argument("--score-every", type=int, default=5, help="launch score.py every K epochs; 0 disables it")
    ap.add_argument(
        "--score-train-every", type=int, default=10,
        help="on scored epochs that are a multiple of K, also score 100 fixed train shots after the val "
             "subset, in the same subprocess. Logged with source score_train. 0 disables it",
    )
    ap.add_argument("--overfit-shots", type=int, default=0, help="train on this many train shots and score them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--precompute-stats", action="store_true",
        help="fit and cache the train-split statistics (ψ pixel mean and std, coil and scalar stats, and the "
             "--feature-set medians and IQRs) under the cache dir, then exit without training. Later runs load them",
    )
    args = ap.parse_args(_argv_with_window(sys.argv[1:]))
    import mast_region
    args.mast_mask = mast_region.name()   # MAST region (FEC_MAST_MASK), kept in the checkpoint's args
    try:
        offsets = parse_window_ms(args.window_ms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.window_ms = format_window_ms(offsets)
    n_stamps = 1 if len(offsets) == 0 else len(offsets)
    if args.feature_set:
        if len(offsets) or args.extra_safe_features or args.robust_inputs:
            raise SystemExit("--feature-set excludes --window-ms, --extra-safe-features and --robust-inputs")
        fv = importlib.import_module(f"features_{args.feature_set}")
        dropped = [n for n in args.drop_features.split(",") if n]
        unknown = [n for n in dropped if n not in fv.FEATURES or n in fv.FLAGS]
        if unknown:
            raise SystemExit(f"--drop-features: {unknown} are not droppable columns of features_{args.feature_set}")
        keep_cols = [i for i, n in enumerate(fv.FEATURES) if n not in dropped]
        k_hist = int(args.history_frames)
        args.n_in = len(keep_cols) * (2 * k_hist + 1) + 2 * k_hist
    elif args.drop_features or args.history_frames:
        raise SystemExit("--drop-features and --history-frames need --feature-set")
    else:
        args.n_in = feature_width(n_stamps, safe=args.extra_safe_features, robust=args.robust_inputs)
    n_image = image_width(args.robust_inputs)
    n_safe = safe_width(args.robust_inputs)
    if args.epochs is None:
        args.epochs = 400 if args.overfit_shots else 40
    linear = args.decoder == "linear"
    if linear and (args.decoder_norm != "physical" or args.output_prior or args.zoom_aug):
        raise SystemExit("--decoder linear needs --decoder-norm physical and excludes --output-prior and --zoom-aug")
    if (args.lambda_jcur or args.lambda_fluxfn or args.lindec_ip_floor) and not linear:
        raise SystemExit("--lambda-jcur, --lambda-fluxfn and --lindec-ip-floor need --decoder linear")
    if not 0 <= args.stop_epoch <= args.epochs:
        raise SystemExit(f"--stop-epoch must be in [0, {args.epochs}], got {args.stop_epoch}")
    last_epoch = args.stop_epoch or args.epochs
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ready = args.cache / "READY"
    if not ready.exists():
        raise SystemExit(f"cache is not ready ({ready}). Run build_cache.py on the GPU allocation first.")
    shots = pd.read_parquet(args.cache / "shots.parquet")
    assert_disjoint(shots)
    train_rows = _rows(shots, "train")
    if args.overfit_shots:
        train_rows = train_rows.head(args.overfit_shots).reset_index(drop=True)
        print(f"overfit: {len(train_rows)} train shots, {int(train_rows['n'].sum())} frames", flush=True)

    coils_mm = np.load(args.cache / "coils.npy", mmap_mode="r")
    psi_mm = np.load(args.cache / "psi.npy", mmap_mode="r")
    scal_mm = np.load(args.cache / "scal.npy", mmap_mode="r")
    phase_mm = np.load(args.cache / "phase.npy", mmap_mode="r")

    if args.pictures != DEFAULT_BASELINE and not args.pictures.exists():
        raise SystemExit(f"{args.pictures} is missing")
    pictures = load_or_build_baseline(args.pictures, args.data_root, args.cache, DEFAULT_FRAMES)
    print("normalisation from the training split...", flush=True)
    train_key = split_key(train_rows)
    # --overfit-shots fits on a few shots; those stats are never written over the full-split cache.
    save_stats = not args.overfit_shots
    norm = cached_stats(
        args.cache / f"norm_train{'_robust' if args.robust_inputs else ''}.npz", train_key,
        lambda: fit_norm(coils_mm, psi_mm, scal_mm, train_rows, robust=args.robust_inputs), save_stats,
    )
    norm["psi_std"] = np.float32(norm["psi_std"])
    norm["robust_inputs"] = bool(args.robust_inputs)
    norm["amp"] = not args.no_amp
    if not args.feature_set:
        print(
            f"coil picture: {n_image} trunk columns{', robust scaling' if args.robust_inputs else ''} "
            f"({args.n_in} inputs, {n_stamps} stamp{'s' if n_stamps != 1 else ''})",
            flush=True,
        )

    feat_mm = None
    if args.feature_set:
        meta_path = args.cache / f"features_{args.feature_set}.json"
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} is missing. Run my_experiments/nn/build_features.py first.")
        meta = json.loads(meta_path.read_text())
        if meta["features"] != fv.FEATURES:
            raise SystemExit(f"{meta_path} columns differ from features_{args.feature_set}.FEATURES; rebuild the cache")
        feat_mm = np.load(args.cache / f"features_{args.feature_set}.npy", mmap_mode="r")
        if feat_mm.shape != (coils_mm.shape[0], fv.N_FEATURES):
            raise SystemExit(f"features cache {feat_mm.shape} does not match {coils_mm.shape[0]} cache frames")

    def _inputs(rows: pd.DataFrame, label: str) -> np.ndarray:
        if feat_mm is not None:
            index = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n)) for r in rows.itertuples(index=False)])
            return np.asarray(feat_mm[index], dtype=np.float32)
        if len(offsets):
            print(f"building a {len(offsets)}-stamp coil window for the {label} split...", flush=True)
            coils = assemble_window(rows, coils_mm, args.data_root, offsets)
        else:
            coils = _stack_stamp(rows, coils_mm)
        if args.extra_safe_features:
            print(f"building safe features for the {label} split...", flush=True)
            coils = np.concatenate([coils, assemble_safe(rows, args.data_root, args.robust_inputs)], axis=1)
        print(f"building coil-picture features for the {label} split...", flush=True)
        coils = np.concatenate(
            [coils, assemble_image(rows, coils_mm, args.data_root, pictures, args.robust_inputs)], axis=1,
        )
        return coils

    train_inputs = _inputs(train_rows, "train")
    if args.feature_set:
        flags = [fv.FEATURES.index(name) for name in fv.FLAGS]
        # The key also covers the cache's own metadata, so a rebuilt features cache refits its stats.
        feat_key = hashlib.sha256((train_key + meta_path.read_text()).encode()).hexdigest()[:16]

        def _fit_features() -> dict:
            mean, std = robust_center_scale(train_inputs, flag_cols=flags)
            return {"feat_mean": mean, "feat_std": std}

        # Stats are per column, fitted and cached over every column; the dropped ones are sliced away after.
        feat = cached_stats(args.cache / f"features_{args.feature_set}_norm.npz", feat_key, _fit_features,
                            save_stats)
        feat = {k: np.asarray(v)[keep_cols] for k, v in feat.items()}
        feat["feat_mean"], feat["feat_std"] = history_stats(feat["feat_mean"], feat["feat_std"], k_hist)
        train_inputs = _with_history(train_inputs[:, keep_cols], train_rows, k_hist)
        norm = {**norm, "feature_set": args.feature_set, "feature_columns": keep_cols, "history_frames": k_hist,
                "feature_names": [fv.FEATURES[i] for i in keep_cols], **feat}
        if args.output_prior:
            if args.decoder_norm != "physical":
                raise SystemExit("--output-prior needs --decoder-norm physical")
            if "R_ax_th" not in norm["feature_names"]:
                raise SystemExit("--output-prior needs the R_ax_th feature to place the prior")
        if (args.output_prior or linear) and "R_ax_th" in norm["feature_names"]:
            # The current frame's stamp sits in the middle of a history window. The linear decoder has no prior,
            # but --lambda-axis still reads R_ax_th per frame.
            norm["rax_col"] = k_hist * len(keep_cols) + norm["feature_names"].index("R_ax_th")
        print(f"features {args.feature_set}: {len(keep_cols)} of {fv.N_FEATURES} columns"
              f"{' (dropped ' + ', '.join(dropped) + ')' if dropped else ''}, "
              f"{float(np.isnan(train_inputs).mean()):.2%} of train values missing", flush=True)
    else:
        image_flags = len(ROBUST_IMAGE_FLAGS) if args.robust_inputs else None
        image_mean, image_std = _fit_column_norm(train_inputs[:, -n_image:], n_image, "image", image_flags)
        norm = {**norm, "image_mean": image_mean, "image_std": image_std}
    if args.extra_safe_features:
        safe_block = train_inputs[:, -(n_safe + n_image):-n_image]
        safe_flags = len(ROBUST_SAFE_FLAGS) if args.robust_inputs else None
        safe_mean, safe_std = _fit_column_norm(safe_block, n_safe, "safe", safe_flags)
        norm = {**norm, "safe_mean": safe_mean, "safe_std": safe_std}
    if train_inputs.shape[1] != args.n_in:
        raise SystemExit(f"input width {train_inputs.shape[1]} does not match n_in {args.n_in}")
    if args.precompute_stats:
        print("train-split statistics are cached; exiting", flush=True)
        return 0
    print(f"loading train split onto {device}...", flush=True)
    train, coil_lo, coil_hi = load_split(
        coils_mm, psi_mm, scal_mm, phase_mm, train_rows, norm, device, keep_raw_scalars=False,
        coil_rows=train_inputs,
    )
    norm = {**norm, "coil_lo": coil_lo, "coil_hi": coil_hi}
    if args.overfit_shots:
        # The success criterion is R²ψ → 1 on these same shots, so evaluate there.
        eval_bundle, _, _ = load_split(
            coils_mm, psi_mm, scal_mm, phase_mm, train_rows, norm, device, keep_raw_scalars=True,
            coil_rows=train_inputs,
        )
        subset = None
        train_subset = None
    else:
        print("loading val split onto the device...", flush=True)
        val_rows = _rows(shots, "val")
        val_inputs = _inputs(val_rows, "val")
        if args.feature_set:
            val_inputs = _with_history(val_inputs[:, keep_cols], val_rows, k_hist)
        eval_bundle, _, _ = load_split(
            coils_mm, psi_mm, scal_mm, phase_mm, val_rows, norm, device, keep_raw_scalars=True,
            coil_rows=val_inputs,
        )
        subset = pd.read_parquet(args.cache / "score_subset.parquet")
        train_subset = score_subset(shots, split="train")
    nbytes = train["psi"].nbytes + eval_bundle["psi"].nbytes
    size = f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"
    print(
        f"ψ tensors on device: {size} "
        f"({train['psi'].shape[0]} train frames, {eval_bundle['psi'].shape[0]} eval frames)",
        flush=True,
    )

    use_boundary = args.loss_band or args.loss_xpoint or args.loss_volume
    li_geo = load_li_geometry(args.cache, train_rows, device) if (args.loss_li or use_boundary) else None

    torch.manual_seed(args.seed)
    green_t = torch.as_tensor(np.ascontiguousarray(pictures["green"], dtype=np.float32), device=device)
    ecoil_t = torch.as_tensor(np.ascontiguousarray(pictures["ecoil"], dtype=np.float32), device=device)
    if args.decoder_norm == "physical":
        # The training grid's machine. Another machine passes its own (R0, Z0, a) at inference.
        d3d = features_v2.geometry("d3d")
        norm["decoder_geometry"] = (float(d3d["R0"]), float(d3d["Z0"]), float(d3d["a"]))
    net_cls, net_extra = Net, {}
    if linear:
        from lindec import LinearDecoderNet, current_from_psi
        net_cls = LinearDecoderNet
        net_extra = {"erode": args.lindec_erode, "activation": args.lindec_act, "exact_ip": args.lindec_exact_ip,
                     "ip_floor": args.lindec_ip_floor, "exact_ip_mast": args.lindec_exact_ip_mast, "cache": args.cache}
    net = net_cls(
        n_in=args.n_in,
        width=args.width,
        scalar_width=args.scalar_width,
        scalar_depth=args.scalar_depth,
        seed_channels=args.seed_channels,
        pixel_mean=norm["pixel_mean"],
        psi_std=float(norm["psi_std"]),
        decoder_norm=args.decoder_norm,
        output_prior=args.output_prior,
        geometry=norm.get("decoder_geometry"),
        **net_extra,
    ).to(device)
    if linear:
        print(f"linear decoder: {int(net.region_d3d.sum())} DIII-D and {int(net.region_mast.sum())} MAST current "
              f"cells ({args.lindec_act}, eroded {args.lindec_erode})", flush=True)
    zoom = None
    if args.zoom_aug:
        if args.decoder_norm != "physical":
            raise SystemExit("--zoom-aug needs --decoder-norm physical")
        from common import train_files
        from zoom import ZoomAug
        pair = lambda text: tuple(float(v) for v in text.split(","))
        zoom = ZoomAug(norm["decoder_geometry"], pair(args.zoom_r), pair(args.zoom_z), pair(args.zoom_shift),
                       args.zoom_native_frac, train_files(args.data_root)[0], args.cache, device)
        print(f"zoom augmentation: R ×{args.zoom_r}, Z ×{args.zoom_z}, outward shift {args.zoom_shift} a, "
              f"{args.zoom_native_frac:.0%} native", flush=True)
    shape_ref = None
    if args.lambda_shape > 0:
        ip_abs = train["stamp"][:, IP_COL].abs()
        q = torch.quantile(ip_abs[torch.randperm(len(ip_abs), device=ip_abs.device)[:200000]].float(),
                           torch.tensor([0.05, 0.5], device=ip_abs.device))
        shape_ref = (float(q[0]), float(q[1]))
        print(f"shape loss: |Ip| floor {shape_ref[0]:.0f} kA, reference {shape_ref[1]:.0f} kA, "
              f"max weight {(shape_ref[1] / shape_ref[0]) ** 2:.1f}", flush=True)
    cur_d = mast = axis_d = axis_m = te_d = te_m = ff_d = ff_m = None
    want_j = args.lambda_jcur > 0 or args.lambda_fluxfn > 0 or args.mast_jitter > 0
    if args.lambda_te > 0:
        if args.decoder_norm != "physical":
            raise SystemExit("--lambda-te needs --decoder-norm physical")
        from physics_loss import TeOrder
        d3d = features_v2.geometry("d3d")
        env = np.load(HERE.parents[1] / "fusion_scoring/masks/d3d_envelope.npz")["mask_coarse"]
        # Train-bundle row k is cache row train_cache_rows[k] (load_split walks train_rows in order).
        train_cache_rows = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n))
                                           for r in train_rows.itertuples(index=False)])
        if len(train_cache_rows) != train["psi"].shape[0]:
            raise SystemExit(f"train rows {len(train_cache_rows)} != train bundle {train['psi'].shape[0]}")
        te_d = [TeOrder(args.cache / "thomson_raw_d3d.npz", train_cache_rows, d3d["r"], d3d["z"], env, d3d["R0"],
                        device, quantity=q, scale_free=args.order_scale_free)
                        for q in args.order_quantities.split(",")]
    if args.lambda_axis > 0:
        if args.decoder_norm != "physical":
            raise SystemExit("--lambda-axis needs --decoder-norm physical")
        from physics_loss import AxisPull
        d3d = features_v2.geometry("d3d")
        env = np.load(HERE.parents[1] / "fusion_scoring/masks/d3d_envelope.npz")["mask_coarse"]
        axis_d = AxisPull(d3d["r"], d3d["z"], env, d3d["R0"], d3d["a"], device, z0=d3d["Z0"],
                          z_slack=args.axis_z_slack)
    if args.lambda_current > 0 or args.mast_current > 0:
        if args.decoder_norm != "physical":
            raise SystemExit("--lambda-current and --mast-current need --decoder-norm physical")
        from physics_loss import CurrentLoss, MastInputs
        d3d = features_v2.geometry("d3d")
        env = np.load(HERE.parents[1] / "fusion_scoring/masks/d3d_envelope.npz")["mask_coarse"]
        cur_d = CurrentLoss(d3d["r"], d3d["z"], env, device)
        if args.mast_current > 0:
            if not (args.output_prior or linear):
                raise SystemExit("--mast-current needs --output-prior (the MAST prior is passed explicitly)")
            mast = MastInputs(args.cache, norm, device, variant=args.mast_variant,
                              history_step=args.mast_history_step)
            import mast_region
            if args.lambda_axis > 0:
                from physics_loss import AxisPull
                gm = features_v2.geometry("mast")
                envm = mast_region.mask()   # MAST region: FEC_MAST_MASK (mast_region.py)
                axis_m = AxisPull(gm["r"], gm["z"], envm, gm["R0"], gm["a"], device, z0=gm["Z0"],
                                  z_slack=args.axis_z_slack, hi=args.mast_axis_hi)
            if args.lambda_te > 0:
                gm = features_v2.geometry("mast")
                envm = mast_region.mask()   # MAST region: FEC_MAST_MASK (mast_region.py)
                te_m = [TeOrder(args.cache / "thomson_raw_mast.npz", mast.rows, gm["r"], gm["z"], envm, gm["R0"],
                                device, quantity=q, scale_free=args.order_scale_free)
                        for q in args.order_quantities.split(",")]
    if args.lambda_fluxfn > 0:
        from fluxfn_loss import FluxFunctionLoss
        from lindec_build import load_g
        ff, cold = {}, {"d3d": None, "mast": None}
        th = {"d3d": None, "mast": None}
        if args.fluxfn_cold_te > 0 or args.fluxfn_pressure > 0:
            # Thomson channels per frame, rows aligned with the frames each loss is called with: the train bundle
            # (load_split walks train_rows in order) and MastInputs.rows.
            tr = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n))
                                 for r in train_rows.itertuples(index=False)])
            if len(tr) != train["psi"].shape[0]:
                raise SystemExit(f"train rows {len(tr)} != train bundle {train['psi'].shape[0]}")
            for m, rows in (("d3d", tr), ("mast", None if mast is None else mast.rows)):
                if rows is not None:
                    raw = np.load(args.cache / f"thomson_raw_{m}.npz")
                    if args.fluxfn_cold_te > 0:
                        cold[m] = {k: raw[k][rows] for k in ("te", "r", "z")}
                    if args.fluxfn_pressure > 0:
                        th[m] = {k: raw[k][rows] for k in ("te", "r", "z")}
                        th[m]["ne"] = np.load(args.cache / f"thomson_raw_ne_{m}.npz")["te"][rows]
        for m in ("d3d", "mast"):
            gm, gg = features_v2.geometry(m), load_g(m, args.cache)
            ff[m] = FluxFunctionLoss(gm["r"], gm["z"], gg["mask"], gg["idx"], gm["R0"], device,
                                     out_weight=args.fluxfn_out_weight, cold=cold[m], cold_te=args.fluxfn_cold_te,
                                     thomson=th[m], pressure_weight=args.fluxfn_pressure,
                                     edge_ext=args.fluxfn_edge_ext, spread_weight=args.fluxfn_spread,
                                     spread_floor=args.fluxfn_spread_floor, a_norm=float(gm["a"]))
        ff_d = ff["d3d"]
        if args.fluxfn_cold_te > 0:
            print(f"flux-function loss: boundary also at Thomson channels ≤ {args.fluxfn_cold_te:g} eV", flush=True)
    mast_env_r = mast_env_z = None
    mast_a = 1.0
    if args.mast_jitter > 0:
        if not (linear and mast is not None):
            raise SystemExit("--mast-jitter needs --decoder linear and --mast-current")
        from lindec_build import load_g
        gm, gg = features_v2.geometry("mast"), load_g("mast", args.cache)
        rr_m, zz_m = np.meshgrid(gm["r"], gm["z"], indexing="xy")
        mast_env_r = torch.as_tensor(rr_m.ravel()[gg["idx"]].astype(np.float32), device=device)
        mast_env_z = torch.as_tensor(zz_m.ravel()[gg["idx"]].astype(np.float32), device=device)
        mast_a = float(gm["a"])
    synth_sets = []
    if args.synth:
        if args.decoder_norm != "physical":
            raise SystemExit("--synth needs --decoder-norm physical")
        from synth_data import SynthSet
        for spec in args.synth:
            tag, _, w = spec.partition(":")
            synth_sets.append(SynthSet(Path(tag), norm, device, mast_variant=args.mast_variant or "_sol0.5",
                                       d3d_picture=lambda st: _external(st, green_t, ecoil_t),
                                       weight=float(w) if w else 1.0))
    mast_mon = None
    if args.decoder_norm == "physical" and not args.no_mast_monitor:
        from mast_monitor import MastMonitor
        mast_mon = MastMonitor(args.cache, norm, device, variant=args.mast_variant or "_sol0.5", mast=mast)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = train["coils"].shape[0]
    steps_per = max(1, math.ceil(n_train / args.batch_size))
    total_steps = steps_per * args.epochs
    warmup = int(args.warmup_epochs * steps_per)

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "psi_std": float(norm["psi_std"]),
        "scal_lo": norm["scal_lo"].tolist(),
        "scal_hi": norm["scal_hi"].tolist(),
        "steps_per_epoch": steps_per,
        "total_steps": total_steps,
        "li_var": li_geo["var"] if li_geo is not None else None,
        "volume_var": li_geo["vol_var"] if li_geo is not None else None,
        "w_band": W_BAND,
        "w_xpoint": W_XPOINT,
    }, indent=2))

    use_amp = device.type == "cuda" and not args.no_amp
    if args.no_amp:
        # The backward pass runs outside the forward's precision context, so TF32 goes off globally.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    best_proxy = -float("inf")
    score_proc = {"proc": None, "log": None}
    scored_epochs: set[int] = set()
    step = 0
    metrics_path = run_dir / "metrics.jsonl"

    def launch_score(epoch: int, wait_for_previous: bool) -> None:
        if args.overfit_shots or args.score_every <= 0 or subset is None:
            return
        if epoch in scored_epochs:
            return
        proc = score_proc["proc"]
        if proc is not None and proc.poll() is None:
            if not wait_for_previous:
                print(f"epoch {epoch}: score.py still running (pid {proc.pid}), skip", flush=True)
                return
            print(f"waiting for score.py pid {proc.pid} before the final score", flush=True)
            proc.wait()
        pred_dir = run_dir / f"score_ep{epoch:03d}"
        _dump_subset(net, eval_bundle, norm, subset["shot"], pred_dir, device, args.batch_size, green_t, ecoil_t)
        # best.pt is chosen by the val proxy, not by S, and last.pt moves on every epoch. Keep the weights that
        # produced this score so the best-S epoch can be recovered once its score arrives.
        shutil.copyfile(run_dir / "last.pt", run_dir / f"score_ep{epoch:03d}.pt")
        if score_proc["log"] is not None:
            score_proc["log"].close()
        log = open(run_dir / f"score_ep{epoch:03d}.log", "w")
        def _score_cmd(pred: Path, split: str) -> list[str]:
            return [
                sys.executable, str(HERE / "score.py"),
                "--pred", str(pred), "--split", split,
                "--append-metrics", str(metrics_path), "--epoch", str(epoch),
                "--cache", str(args.cache), "--data-root", str(args.data_root),
            ]
        cmd = _score_cmd(pred_dir, "score")
        if args.score_train_every > 0 and epoch % args.score_train_every == 0 and train_subset is not None:
            # Train shots run after the val subset, so the val line lands first.
            train_dir = run_dir / f"score_train_ep{epoch:03d}"
            _dump_subset(net, train, norm, train_subset["shot"], train_dir, device, args.batch_size,
                         green_t, ecoil_t)
            cmd = ["bash", "-c", shlex.join(cmd) + " && " + shlex.join(_score_cmd(train_dir, "train_score"))]
        score_proc["log"] = log
        score_proc["proc"] = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        scored_epochs.add(epoch)
        if args.mast_checks_every > 0 and epoch % args.mast_checks_every == 0:
            ck = run_dir / f"score_ep{epoch:03d}.pt"
            diag = HERE / "mast_diag"
            cmds = " ; ".join(shlex.join([sys.executable, str(diag / f), str(ck)])
                              for f in ("lindec_mast_checks.py", "physics_ckpt.py", "li_ckpt_cli.py"))
            if synth_sets:
                cmds += " ; " + shlex.join([sys.executable, str(diag / "synth_shape_check.py"), str(ck),
                                            *[str(ss_.tag) for ss_ in synth_sets]])
            subprocess.Popen(["bash", "-c", f"export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3; {cmds}"],
                             stdout=open(run_dir / f"mast_checks_ep{epoch:03d}.log", "w"), stderr=subprocess.STDOUT,
                             start_new_session=True)
            print(f"epoch {epoch}: MAST checks -> mast_checks_ep{epoch:03d}.log", flush=True)
        print(f"epoch {epoch}: score.py pid {score_proc['proc'].pid} -> {pred_dir}", flush=True)

    for epoch in range(1, last_epoch + 1):
        t0 = time.perf_counter()
        net.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = mse_sum = hub_sum = g_sum = li_sum = band_sum = xpt_sum = vol_sum = 0.0
        sur_scale = surrogate_scale(epoch, args.surrogate_alpha, args.surrogate_start_epoch,
                                    args.surrogate_ramp_epochs) if li_geo is not None else 0.0
        li_scale = sur_scale if args.loss_li else 0.0
        gn_sum = 0.0
        cur_sum = [0.0, 0.0, 0.0, 0.0]
        jcur_sum = 0.0
        ff_sum = [0.0, 0.0, 0.0, 0.0]
        jit_sum, low_sum, low_n = 0.0, 0.0, 0
        synth_sum = [0.0] * len(synth_sets)
        pres_sum, pres_n = [0.0, 0.0], [0, 0]
        axis_sum = [0.0, 0.0, 0.0, 0.0]
        te_sum = [0.0, 0.0, 0.0, 0.0]
        shape_sum = 0.0
        seen = 0
        last_lr = args.lr
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            b = int(idx.shape[0])
            last_lr = scheduled_lr(step, total_steps, warmup, args.lr)
            for group in opt.param_groups:
                group["lr"] = last_lr
            opt.zero_grad(set_to_none=True)
            J_hat = None
            if zoom is None:
                picture = _external(train["stamp"][idx], green_t, ecoil_t)
                with precision_context(device.type, use_amp):
                    if want_j:
                        psi_hat, scal_hat, J_hat = net(train["coils"][idx], picture, train["stamp"][idx, IP_COL],
                                                       train["rax"][idx], return_current=True)
                    else:
                        psi_hat, scal_hat = net(train["coils"][idx], picture, train["stamp"][idx, IP_COL],
                                                train["rax"][idx])
            else:
                zp = zoom.sample(b)
                zb = zoom.batch(train["stamp"][idx], train["rax"][idx], train["psi"][idx], net.pixel_mean,
                                net.psi_std, zp)
                # Native examples get the exact picture, prior and coordinates that validation uses; the canvas
                # interpolation is off by up to 0.6% near the coils.
                nat = (zp["zr"] == 1) & (zp["zz"] == 1) & (zp["shift"] == 0)
                if bool(nat.any()):
                    exact = _external(train["stamp"][idx][nat], green_t, ecoil_t).float()
                    zb["picture"][nat] = exact
                    zb["prior"][nat] = net.plasma_prior(train["stamp"][idx][nat, IP_COL], train["rax"][idx][nat])
                    zb["coord"][nat] = net.coord.unsqueeze(0).expand(int(nat.sum()), -1, -1, -1)
                with precision_context(device.type, use_amp):
                    psi_hat, scal_hat = net(train["coils"][idx], zb["picture"], train["stamp"][idx, IP_COL],
                                            train["rax"][idx], coord=zb["coord"], prior=zb["prior"])
            psi_hat = psi_hat.float()
            scal_hat = scal_hat.float()
            loss_s = F.huber_loss(scal_hat, train["scal"][idx])
            loss_g = psi_hat.new_zeros(())
            if zoom is None:
                loss_psi = F.mse_loss(psi_hat, train["psi"][idx])
                if args.lambda_grad:
                    loss_g = grad_mse(psi_hat, train["psi"][idx], net.z_coord, net.r_coord)
            else:
                from zoom import masked_losses
                loss_psi, zoom_g = masked_losses(psi_hat, zb["target_n"], zb["mask"], zb["dr"], zb["dz"])
                if args.lambda_grad:
                    loss_g = zoom_g
            loss = loss_psi + args.lambda_s * loss_s + args.lambda_grad * loss_g
            if J_hat is not None and args.lambda_jcur > 0:
                # Auxiliary current loss: the map's current against the target's, per unit of Ip, on the envelope.
                with torch.autocast(device_type=device.type, enabled=False):
                    with torch.no_grad():
                        J_true = current_from_psi(train["psi"][idx] * net.psi_std + net.pixel_mean, net.r_coord,
                                                  net.z_coord)[:, net.idx_d3d]
                    ip_a = train["stamp"][idx, IP_COL].float() * 1000.0
                    big = (ip_a.abs() >= 150e3).float()
                    err = ((J_hat.float() - J_true) / ip_a.abs().clamp_min(150e3).view(-1, 1)).square().sum(1)
                    loss_j = (err * big).sum() / big.sum().clamp_min(1.0)
                jcur_sum += loss_j.item() * b
                loss = loss + args.lambda_jcur * loss_j
            if ff_d is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    nf = min(args.fluxfn_frames, b)
                    loss_fd, fd, fo = ff_d(psi_hat[:nf] * net.psi_std + net.pixel_mean, J_hat[:nf],
                                           train["stamp"][idx[:nf], IP_COL], idx[:nf])
                if fd == fd:
                    ff_sum[0] += fd * b; ff_sum[1] += fo * b
                if ff_d.last_pressure == ff_d.last_pressure:
                    pres_sum[0] += ff_d.last_pressure * b; pres_n[0] += b
                loss = loss + args.lambda_fluxfn * loss_fd
            loss_def = psi_hat.new_zeros(())
            if zoom is not None and args.zoom_default_weight > 0:
                # No EFIT target outside DIII-D's grid: pull the network's part of ψ towards zero there, so the
                # prediction falls back to coil picture + plasma prior instead of anything the loss ignores.
                miss = ~zb["mask"]
                if bool(miss.any()):
                    with torch.autocast(device_type=device.type, enabled=False):
                        phys = psi_hat * net.psi_std + net.pixel_mean
                        scale = flux_scale(train["stamp"][idx, IP_COL], net.r0).view(-1, 1)
                        corr = (phys - zb["picture"] - zb["prior"]) / scale
                        loss_def = (corr.square() * miss.float()).sum() / miss.float().sum()
                    loss = loss + args.zoom_default_weight * loss_def
            if shape_ref is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    if zoom is None:
                        dpsi = psi_hat - train["psi"][idx]
                        ip_s = train["stamp"][idx, IP_COL]
                    else:
                        dpsi = (psi_hat - zb["target_n"])[nat]
                        ip_s = train["stamp"][idx][nat, IP_COL]
                    dpsi = dpsi - dpsi.mean(1, keepdim=True)
                    w = (shape_ref[1] / ip_s.abs().clamp_min(shape_ref[0])).square()
                    loss_sh = (w * dpsi.square().mean(1)).mean() if dpsi.shape[0] else psi_hat.new_zeros(())
                shape_sum += loss_sh.item() * b
                loss = loss + args.lambda_shape * loss_sh
            if cur_d is not None:
                # Current consistency (physics_loss.py): one-signed current inside the vessel, summing to Ip.
                with torch.autocast(device_type=device.type, enabled=False):
                    sel = slice(None) if zoom is None else nat
                    phys = psi_hat[sel] * net.psi_std + net.pixel_mean
                    loss_cd, cw, ct = cur_d(phys, train["stamp"][idx][sel, IP_COL])
                cur_sum[0] += cw * b; cur_sum[1] += ct * b
                if args.lambda_current > 0:
                    loss = loss + args.lambda_current * loss_cd
            if axis_d is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    sel = slice(None) if zoom is None else nat
                    phys = psi_hat[sel] * net.psi_std + net.pixel_mean
                    loss_ad, ad = axis_d(phys, train["stamp"][idx][sel, IP_COL], train["rax"][idx][sel])
                axis_sum[0] += loss_ad.item() * b; axis_sum[1] += (0.0 if ad != ad else ad) * b
                loss = loss + args.lambda_axis * loss_ad
            if te_d is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    sel = slice(None) if zoom is None else nat
                    phys = psi_hat[sel] * net.psi_std + net.pixel_mean
                    for k_q, lq in enumerate(te_d):
                        loss_td, td = lq(phys, train["stamp"][idx][sel, IP_COL], idx[sel])
                        if k_q == 0:
                            te_sum[0] += loss_td.item() * b; te_sum[1] += td * b
                        loss = loss + args.lambda_te * loss_td
            if mast is not None:
                mb = (mast.batch_triplets(args.mast_batch_size // 3) if args.mast_jitter > 0
                      else mast.batch(args.mast_batch_size))
                with precision_context(device.type, use_amp):
                    out_m = net(mb["coils"], mb["picture"], mb["ip"], mb["rax"], coord=mast.coord,
                                prior=mb["prior"], r0=mast.r0,
                                **({"machine": "mast", "return_current": want_j} if linear else {}))
                psi_m = out_m[0]
                if args.mast_jitter > 0:
                    with torch.autocast(device_type=device.type, enabled=False):
                        w = out_m[2].float().abs()
                        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
                        cen = torch.stack([w @ mast_env_r, w @ mast_env_z], 1).view(3, -1, 2)
                        zig = cen[0] - 0.5 * (cen[1] + cen[2])
                        loss_jit = zig.square().sum(1).mean() / mast_a ** 2
                    jit_sum += loss_jit.item() * b
                    loss = loss + args.mast_jitter * loss_jit
                if args.lambda_fluxfn > 0:
                    with torch.autocast(device_type=device.type, enabled=False):
                        loss_fm, fm, fmo = ff["mast"](psi_m.float() * net.psi_std + net.pixel_mean, out_m[2], mb["ip"], mb["idx"])
                    if fm == fm:
                        ff_sum[2] += fm * b; ff_sum[3] += fmo * b
                    if ff["mast"].last_low == ff["mast"].last_low:
                        low_sum += ff["mast"].last_low * b; low_n += b
                    if ff["mast"].last_pressure == ff["mast"].last_pressure:
                        pres_sum[1] += ff["mast"].last_pressure * b; pres_n[1] += b
                    loss = loss + args.lambda_fluxfn * loss_fm
                with torch.autocast(device_type=device.type, enabled=False):
                    loss_cm, mw, mt = mast.loss(psi_m.float() * net.psi_std + net.pixel_mean, mb["ip"])
                cur_sum[2] += mw * b; cur_sum[3] += mt * b
                loss = loss + args.mast_current * loss_cm
                if axis_m is not None:
                    with torch.autocast(device_type=device.type, enabled=False):
                        loss_am, am = axis_m(psi_m.float() * net.psi_std + net.pixel_mean, mb["ip"], mb["rax"])
                    axis_sum[2] += loss_am.item() * b; axis_sum[3] += (0.0 if am != am else am) * b
                    loss = loss + args.lambda_axis * loss_am
                if te_m is not None:
                    with torch.autocast(device_type=device.type, enabled=False):
                        phys_m = psi_m.float() * net.psi_std + net.pixel_mean
                        for k_q, lq in enumerate(te_m):
                            loss_tm, tm = lq(phys_m, mb["ip"], mb["idx"])
                            if k_q == 0:
                                te_sum[2] += loss_tm.item() * b; te_sum[3] += tm * b
                            loss = loss + args.lambda_te * loss_tm
            for k_s, ss in enumerate(synth_sets):
                sidx = ss.batch_idx(args.synth_batch)
                if sidx is None:
                    continue
                with precision_context(device.type, use_amp):
                    loss_syn = ss.loss(net, sidx, linear)
                synth_sum[k_s] += loss_syn.item() * b
                loss = loss + ss.weight * loss_syn
            loss_li = psi_hat.new_zeros(())
            loss_band = psi_hat.new_zeros(())
            loss_xpt = psi_hat.new_zeros(())
            loss_vol = psi_hat.new_zeros(())
            if sur_scale > 0:
                ok = li_geo["valid"][idx]
                if bool(ok.any()):
                    sel = idx[ok]
                    if args.loss_li:
                        d_li, _ = li_error(
                            psi_hat[ok], train["psi"][sel], net.pixel_mean, net.psi_std,
                            li_geo["contour"][sel], unpack_inside(li_geo["inside"][sel]), net.z_coord, net.r_coord,
                        )
                        loss_li = li_geo["weight"] * d_li.square().mean()
                        loss = loss + sur_scale * loss_li
                    if use_boundary:
                        band, d_vol, xpt, saddle = boundary_error(
                            psi_hat[ok], train["psi"][sel], net.pixel_mean, net.psi_std,
                            li_geo["contour"][sel], net.z_coord, net.r_coord,
                        )
                        # Envelope-limited frames count as 0, so each term is a mean over the geometry frames.
                        keep = saddle.float()
                        loss_band = W_BAND * (band * keep).mean()
                        loss_xpt = W_XPOINT * (xpt * keep).mean()
                        loss_vol = li_geo["vol_weight"] * (d_vol.square() * keep).mean()
                        if args.loss_band:
                            loss = loss + sur_scale * loss_band
                        if args.loss_xpoint:
                            loss = loss + sur_scale * loss_xpt
                        if args.loss_volume:
                            loss = loss + sur_scale * loss_vol
            loss.backward()
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            else:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))
            opt.step()
            loss_sum += loss.detach().item() * b
            mse_sum += loss_psi.detach().item() * b
            hub_sum += loss_s.detach().item() * b
            g_sum += loss_g.detach().item() * b
            li_sum += loss_li.detach().item() * b
            band_sum += loss_band.detach().item() * b
            xpt_sum += loss_xpt.detach().item() * b
            vol_sum += loss_vol.detach().item() * b
            gn_sum += gn.detach().item() * b
            seen += b
            step += 1
        stats = evaluate(net, args.batch_size, norm, eval_bundle, device, green_t, ecoil_t)
        synth_eval = {ss.tag: ss.evaluate(net, linear) for ss in synth_sets}
        mon = mast_mon(net, linear) if mast_mon is not None else {}
        record = {
            "epoch": epoch,
            "eval": "overfit" if args.overfit_shots else "val",
            "train_loss": loss_sum / seen,
            "train_mse": mse_sum / seen,
            "train_huber": hub_sum / seen,
            "train_grad": g_sum / seen,
            "train_li": li_sum / seen,
            "train_band": band_sum / seen,
            "train_xpoint": xpt_sum / seen,
            "train_volume": vol_sum / seen,
            "li_scale": li_scale,
            "surrogate_scale": sur_scale,
            "lr": last_lr,
            "grad_norm": gn_sum / seen,
            "d3d_wrong_sign": cur_sum[0] / seen, "d3d_total_err": cur_sum[1] / seen,
            "mast_wrong_sign": cur_sum[2] / seen, "mast_total_err": cur_sum[3] / seen,
            "d3d_axis_loss": axis_sum[0] / seen, "d3d_axis_offset": axis_sum[1] / seen,
            "mast_axis_loss": axis_sum[2] / seen, "mast_axis_offset": axis_sum[3] / seen,
            "train_shape": shape_sum / seen,
            "train_jcur": jcur_sum / seen,
            "d3d_fluxfn": ff_sum[0] / seen, "d3d_fluxfn_out": ff_sum[1] / seen,
            "mast_fluxfn": ff_sum[2] / seen, "mast_fluxfn_out": ff_sum[3] / seen,
            **{f"synth_{ss.tag}_train_loss": synth_sum[k_s] / seen for k_s, ss in enumerate(synth_sets)},
            **{f"synth_{t}_heldout_{k}": v for t, e in synth_eval.items() for k, v in e.items()},
            **mon,
            "mast_jitter": jit_sum / seen, "mast_spread_low": low_sum / max(low_n, 1),
            "d3d_pressure": pres_sum[0] / max(pres_n[0], 1), "mast_pressure": pres_sum[1] / max(pres_n[1], 1),
            "d3d_te_loss": te_sum[0] / seen, "d3d_te_wrong": te_sum[1] / seen,
            "mast_te_loss": te_sum[2] / seen, "mast_te_wrong": te_sum[3] / seen,
            "seconds": time.perf_counter() - t0,
            **stats,
        }
        if device.type == "cuda":
            record["gpu_gib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        append_jsonl(metrics_path, record)
        improved = stats["proxy"] > best_proxy
        if improved:
            best_proxy = stats["proxy"]
            _save(run_dir / "best.pt", net, norm, args, epoch, best_proxy, coil_lo, coil_hi)
        _save(run_dir / "last.pt", net, norm, args, epoch, stats["proxy"], coil_lo, coil_hi)
        gpu = f"  gpu {record['gpu_gib']:.1f} GiB" if epoch == 1 and "gpu_gib" in record else ""
        print(
            f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  "
            f"R²ψ {stats['r2_psi']:.4f}  R²qb {stats['r2_qb']:.4f}  "
            f"proxy {stats['proxy']:.4f}  ∇rmse {stats['rmse_grad']:.4g}  "
            + (f"li {record['train_li']:.3g}×{li_scale:g}  " if args.loss_li else "")
            + (f"band {record['train_band']:.3g} xpt {record['train_xpoint']:.3g} vol {record['train_volume']:.3g}"
               f"×{sur_scale:g}  " if use_boundary else "")
            + (f"cur D {record['d3d_wrong_sign']:.3f}/{record['d3d_total_err']:.3f} "
               f"M {record['mast_wrong_sign']:.3f}/{record['mast_total_err']:.3f}  " if cur_d is not None else "")
            + (f"axis D {record['d3d_axis_offset']:+.3f} M {record['mast_axis_offset']:+.3f}  " if axis_d is not None else "")
            + (f"Te D {record['d3d_te_wrong']:.3f} M {record['mast_te_wrong']:.3f}  " if te_d is not None else "")
            + (f"shape {record['train_shape']:.4g}  " if shape_ref is not None else "")
            + (f"J {record['train_jcur']:.4g}  " if args.lambda_jcur > 0 else "")
            + (f"fluxfn D {record['d3d_fluxfn']:.3f}/{record['d3d_fluxfn_out']:.3f} "
               f"M {record['mast_fluxfn']:.3f}/{record['mast_fluxfn_out']:.3f}  " if args.lambda_fluxfn > 0 else "")
            + (f"pres D {record['d3d_pressure']:.3f} M {record['mast_pressure']:.3f}  " if args.fluxfn_pressure > 0 else "")
            + (f"jit {record['mast_jitter']:.2e}  " if args.mast_jitter > 0 else "")
            + (MastMonitor.line(mon) if mon else "")
            + "".join(f"syn {ss.tag} {record[f'synth_{ss.tag}_train_loss']:.3g}/R²ψ {synth_eval[ss.tag]['r2_psi']:.4f}  "
                      for ss in synth_sets)
            + (f"spread<floor M {record['mast_spread_low']:.3f}  " if args.lambda_fluxfn > 0 and args.fluxfn_spread > 0 else "")
            + f"{record['seconds']:.0f}s" + ("  best" if improved else "") + gpu,
            flush=True,
        )
        if args.score_every and epoch % args.score_every == 0:
            launch_score(epoch, wait_for_previous=False)
    launch_score(last_epoch, wait_for_previous=True)
    peak = f"  gpu peak {record['gpu_gib']:.1f} GiB" if device.type == "cuda" else ""
    print(f"done. best proxy {best_proxy:.4f}{peak} -> {run_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
