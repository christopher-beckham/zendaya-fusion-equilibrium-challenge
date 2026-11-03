"""Paths, the canonical split, and the one featurisation used by training and inference.

`featurize` reads coil currents and `efit_times`. `--extra-safe-features` also reads
Thomson profiles and coil geometry. Neither path reads EFIT targets. Interpolation and the
flat-top sign match the input half of `my_experiments/data_stats.py:process`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from data_fixes import fix_d3d_ip_times  # noqa: E402

DATA_ROOT = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge"))   # FEC_DATA_ROOT: the official challenge data
CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))   # FEC_CACHE: derived caches
TRAIN_CONFIG = "diii_d_train"

SIGS = [
    "ECOILA", "bcoil",
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B",
    "F6A", "F6B", "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
    "plasma_current",
]
# Shaping coils whose vacuum flux can be a fixed picture. Order is the coil-tensor order.
# ECOILA and bcoil are not in this list: the solenoid is not in kA·turn, and bcoil has no poloidal rectangle.
F_COILS = SIGS[2:20]
F_SLICE = slice(2, 20)
# Currents at time t occupy columns 0–20 even when a window is appended after them.
ECOIL_COL = SIGS.index("ECOILA")
BCOIL_COL = SIGS.index("bcoil")
IP_COL = SIGS.index("plasma_current")
IP_FLOOR_KA = 1.0
if F_COILS != [
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B",
    "F6A", "F6B", "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
]:
    raise RuntimeError(f"F-coil columns drifted from SIGS: {F_COILS}")
N_COIL = len(SIGS)
# --extra-safe-features, one copy at the picture time. Order is log10(p_peak),
# p/p_peak at 8 steps from the peak outward, (I_upper - I_lower) / Ip, and
# sum |I_shaping| / |Ip|.
N_SHAPE = 8
N_SAFE = 1 + N_SHAPE + 2
# Trunk columns read off the coil picture, plus the shared scalars and Thomson moments.
# Order: Bz, decay index, hexapole, up-down, null R/Z, X-point R/Z,
# Bt, cylindrical q, loop voltage, dIp/dt, aspect, then five Thomson moments.
N_IMAGE = 18
SAFE_NAMES = (
    ["log10_p_peak"] + [f"p_shape_{i}" for i in range(N_SHAPE)]
    + ["updown_shaping_over_Ip", "total_shaping_over_Ip"]
)
IMAGE_NAMES = [
    "Bz_norm", "decay_index", "hexapole", "updown_flux", "null_R", "null_Z", "xpt_R", "xpt_Z",
    "Bt", "q_cyl", "V_loop", "dIp_dt", "aspect",
    "th_R_peak", "th_Z_peak", "th_peaking", "th_width", "th_steep",
]
# --robust-inputs layout. Constant, duplicate and grid-snapped columns are dropped. Values that
# cannot be computed are NaN, and a 0/1 column after each block says so. The scaler turns NaN
# into 0, the train median, after standardising. Flag columns are not scaled.
_ROBUST_SAFE_DROP = {"p_shape_0"}
_ROBUST_IMAGE_DROP = {"aspect", "th_Z_peak", "null_R", "null_Z", "xpt_R", "xpt_Z"}
ROBUST_SAFE_KEEP = [i for i, n in enumerate(SAFE_NAMES) if n not in _ROBUST_SAFE_DROP]
ROBUST_IMAGE_KEEP = [i for i, n in enumerate(IMAGE_NAMES) if n not in _ROBUST_IMAGE_DROP]
ROBUST_SAFE_FLAGS = ["safe_thomson_missing"]
ROBUST_IMAGE_FLAGS = ["thomson_missing", "rate_missing"]
N_SAFE_ROBUST = len(ROBUST_SAFE_KEEP) + len(ROBUST_SAFE_FLAGS)
N_IMAGE_ROBUST = len(ROBUST_IMAGE_KEEP) + len(ROBUST_IMAGE_FLAGS)
if (len(SAFE_NAMES), len(IMAGE_NAMES)) != (N_SAFE, N_IMAGE) or [
    IMAGE_NAMES.index(n) for n in ("Bt", "q_cyl", "V_loop", "dIp_dt", "th_R_peak")
] != [8, 9, 10, 11, 13]:
    raise RuntimeError("SAFE_NAMES or IMAGE_NAMES drifted from the column order in image_columns")
# True boundary and inside-region per cache row, written by li_geometry.py for --loss-li.
LI_FILES = {
    "contour": "li_contour.npy",
    "inside": "li_inside.npy",
    "li": "li_true.npy",
    "status": "li_status.npy",
}
# Frames in the first rampup samples have |Ip| of a few kA, which makes the Ip ratios explode.
ROBUST_IP_FLOOR_KA = 100.0
# DIII-D toroidal-field coil, 144 turns (Luxon, Nucl. Fusion 42, 2002).
# bcoil is the feed current in kA. The count is not fit from the labels.
D3D_TF_TURNS = 144
_MU0 = 4.0e-7 * np.pi
# A Thomson group is the radial midplane chord when its median |Z| is within this.
SAFE_Z_MAX_M = 0.2
_SHAPE_S = np.linspace(0.0, 1.0, N_SHAPE)
INPUT_CLIP = 5.0
# Stamp at t versus the cache. Same threshold as predict.py's parity check.
_CACHE_ATOL = 1e-4
# Analysis columns stored in scal.npy. The model is trained on the first two only.
SCALAR_COLS = ["efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"]
SCALAR_NAMES = ["q95", "betaN", "li", "R_axis", "Z_axis"]
TARGET_SCALARS = ["q95", "betaN"]  # indices 0, 1 of scal.npy
N_PSI = 65 * 65
GRID = 65

PHASE_TO_CODE = {"rampup": 0, "flattop": 1, "rampdown": 2}
CODE_TO_PHASE = {v: k for k, v in PHASE_TO_CODE.items()}

# Columns featurize is allowed to read. Anything else, especially other efit_* targets, is leakage.
_INPUT_COLS = (
    ["efit_times", "magnetics_time", "magnetics_plasma_current_times"]
    + [f"magnetics_{s}" for s in SIGS]
)


def assert_input_columns(cols) -> None:
    """Leakage guard: no efit_* column except efit_times."""
    for c in cols:
        if c.startswith("efit_") and c != "efit_times":
            raise AssertionError(f"featurize tried to read target column {c}")
        if c.startswith("thomson_") or c.startswith("coil_"):
            continue
        if c != "efit_times" and not c.startswith("magnetics_"):
            raise AssertionError(f"featurize tried to read non-input column {c}")


def _col(table, name) -> np.ndarray:
    """One cell of a one-row parquet, flattened to a numpy vector. Same access as data_stats.process."""
    values = table.column(name).combine_chunks()[0].values
    return values.to_numpy(zero_copy_only=False)


def train_files(data_root: Path | None = None) -> list[Path]:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    files = sorted((root / "data" / TRAIN_CONFIG).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no {TRAIN_CONFIG} parquets under {root / 'data'}. "
            "Pass --data-root to the folder that contains data/."
        )
    return files


def assign_splits(n: int, seed: int = 42) -> np.ndarray:
    """Match control_vs_mirror_val.py: sorted files, RandomState(42).permutation, 70/10/20.

    Returns an object array of length n with values 'train', 'val', 'test', indexed like the
    sorted file list (not like the permutation).
    """
    n_test, n_val = max(1, int(n * 0.2)), max(1, int(n * 0.1))
    n_train = n - n_test - n_val
    perm = np.random.RandomState(seed).permutation(n)
    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"
    return split


def parse_window_ms(text: str | None) -> np.ndarray:
    """Millisecond offsets around each picture time. Empty means the reading at t only.

    0 must appear exactly once. It is moved to the front so columns 0–20 are the
    currents at the picture time. The other offsets stay in the order written.
    """
    raw = (text or "").strip()
    if not raw:
        return np.zeros(0, dtype=np.float64)
    try:
        offsets = np.array([float(part.strip()) for part in raw.split(",") if part.strip()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"--window-ms must be comma-separated milliseconds, got {text!r}") from exc
    if offsets.size == 0 or not np.isfinite(offsets).all():
        raise ValueError(f"--window-ms must be finite milliseconds, got {text!r}")
    if len(set(offsets.tolist())) != len(offsets):
        raise ValueError(f"--window-ms repeats an offset: {text!r}")
    zero = np.flatnonzero(offsets == 0.0)
    if zero.size != 1:
        raise ValueError(f"--window-ms must contain 0 exactly once, got {text!r}")
    order = [int(zero[0])] + [i for i in range(len(offsets)) if i != int(zero[0])]
    return offsets[order]


def format_window_ms(offsets: np.ndarray) -> str:
    """Canonical --window-ms string. Empty if there is no window."""
    if len(offsets) == 0:
        return ""
    return ",".join(f"{float(v):g}" for v in offsets)


def as_bool(value) -> bool:
    """Checkpoint args are bools. A string must not use bool(), because bool('False') is True."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _stat_pair(mean, std, n: int, label: str):
    if mean is None and std is None:
        return None
    if mean is None or std is None:
        raise ValueError(f"{label} mean and std must both be set")
    mean_v = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_v = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_v.shape != (n,) or std_v.shape != (n,):
        raise ValueError(f"{label} stats must have length {n}, got {mean_v.shape} and {std_v.shape}")
    return mean_v, std_v


def image_width(robust: bool) -> int:
    return N_IMAGE_ROBUST if robust else N_IMAGE


def safe_width(robust: bool) -> int:
    return N_SAFE_ROBUST if robust else N_SAFE


def tail_widths(norm: dict) -> tuple[int, int]:
    """(safe, image) column counts at the end of the input, read off the stored stats."""
    n_image = len(np.asarray(norm["image_mean"]).reshape(-1))
    n_safe = 0 if norm.get("safe_mean") is None else len(np.asarray(norm["safe_mean"]).reshape(-1))
    return n_safe, n_image


def check_inputs(coils: np.ndarray, *, n_tail: int, robust: bool, label: str) -> None:
    """Raise unless the input is finite. In the robust layout, NaN marks a missing value and
    is allowed in the last `n_tail` (safe and image) columns."""
    coils = np.asarray(coils)
    head = coils[:, :coils.shape[1] - n_tail] if robust else coils
    if not np.isfinite(head).all() or np.isinf(coils).any():
        raise RuntimeError(f"{label}: inputs are not finite")


def robust_center_scale(values: np.ndarray, flag_cols=()) -> tuple[np.ndarray, np.ndarray]:
    """Median and IQR/1.349 per column, ignoring NaN. Flag columns get 0 and 1.

    A column whose IQR is 0 (over half the frames share one value) falls back to its
    std, and to 1 if that is 0 too.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.isinf(values).any():
        raise RuntimeError("features contain infinities")
    center = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(values, axis=0)
    scale = np.where(scale < 1e-8, std, scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    for j in flag_cols:
        center[j], scale[j] = 0.0, 1.0
    if not (np.isfinite(center).all() and np.isfinite(scale).all()):
        raise RuntimeError("a column has no finite training values")
    return center.astype(np.float32), scale.astype(np.float32)


def broadcast_coil_stats(
    mean, std, n_cols: int, image_mean, image_std, safe_mean=None, safe_std=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the 21 per-coil mean/std once per stamp, then safe columns, then the image columns.

    The coil pair does not depend on the offset. Safe stats cover the columns appended by
    --extra-safe-features. The image columns are always last. Block widths come from the
    stats, so both the default and the --robust-inputs layout work.
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != (N_COIL,) or std.shape != (N_COIL,):
        raise ValueError(f"coil stats must have length {N_COIL}, got {mean.shape} and {std.shape}")
    n_image = len(np.asarray(image_mean).reshape(-1)) if image_mean is not None else N_IMAGE
    if n_image not in (N_IMAGE, N_IMAGE_ROBUST):
        raise ValueError(f"image stats have length {n_image}, expected {N_IMAGE} or {N_IMAGE_ROBUST}")
    image = _stat_pair(image_mean, image_std, n_image, "image")
    if image is None:
        raise ValueError("image mean and std are required")
    n_safe = len(np.asarray(safe_mean).reshape(-1)) if safe_mean is not None else 0
    if safe_mean is not None and n_safe not in (N_SAFE, N_SAFE_ROBUST):
        raise ValueError(f"safe stats have length {n_safe}, expected {N_SAFE} or {N_SAFE_ROBUST}")
    safe = _stat_pair(safe_mean, safe_std, n_safe, "safe")
    coil_cols = n_cols - n_safe - n_image
    if coil_cols <= 0 or coil_cols % N_COIL != 0:
        raise ValueError(
            f"input width {n_cols} is not {n_safe} safe columns "
            f"and {n_image} image columns after a multiple of {N_COIL}"
        )
    reps = coil_cols // N_COIL
    out_mean = np.tile(mean, reps)
    out_std = np.tile(std, reps)
    if safe is not None:
        out_mean = np.concatenate([out_mean, safe[0]])
        out_std = np.concatenate([out_std, safe[1]])
    out_mean = np.concatenate([out_mean, image[0]])
    out_std = np.concatenate([out_std, image[1]])
    return out_mean, out_std


def _interp_offsets(et: np.ndarray, t: np.ndarray, v: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """(T, n_off) currents. np.interp repeats the first or last sample outside the recording."""
    out = np.full((len(et), len(offsets)), np.nan, dtype=np.float32)
    n = min(len(t), len(v))
    tt = np.asarray(t[:n], dtype=np.float64)
    vv = np.asarray(v[:n], dtype=np.float64)
    fin = np.isfinite(vv)
    if int(fin.sum()) <= 1:
        return out
    tt, vv = tt[fin], vv[fin]
    out[:, 0] = np.interp(et, tt, vv).astype(np.float32)
    if len(offsets) > 1:
        query = et[:, None] + offsets[1:][None, :]
        out[:, 1:] = np.interp(query.ravel(), tt, vv).reshape(len(et), -1).astype(np.float32)
    return out


def featurize(path: Path, offsets_ms: np.ndarray | None = None) -> dict:
    """Coil currents at each EFIT time, plus the mirror sign and the Ip phase.

    Returns coils (T, 21) float32, not yet mirrored, s in {+1, -1}, phase (T,) str,
    and t (T,) float64. With offsets, also window (T, n_off, 21), unmirrored, and
    offset 0 in position 0. Training and inference both call this, then `coil_window`.
    """
    path = Path(path)
    assert_input_columns(_INPUT_COLS)
    table = pq.read_table(path, columns=list(_INPUT_COLS))
    et = _col(table, "efit_times").astype(np.float64)
    mt = _col(table, "magnetics_time").astype(np.float64)
    row = {
        "magnetics_plasma_current_times": _col(table, "magnetics_plasma_current_times"),
        "magnetics_plasma_current": _col(table, "magnetics_plasma_current"),
        "magnetics_time": mt,
    }
    ip_t = fix_d3d_ip_times(row)

    use_window = offsets_ms is not None and len(offsets_ms) > 0
    if use_window:
        offsets_ms = np.asarray(offsets_ms, dtype=np.float64)
        if float(offsets_ms[0]) != 0.0:
            raise ValueError("window offsets must start with 0")
    coils = np.empty((len(et), len(SIGS)), dtype=np.float32)
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for j, name in enumerate(SIGS):
        v = _col(table, f"magnetics_{name}").astype(np.float64)
        t = ip_t if name == "plasma_current" else mt
        n = min(len(t), len(v))
        t, v = np.asarray(t[:n], dtype=np.float64), np.asarray(v[:n], dtype=np.float64)
        series.append((t, v))
        fin = np.isfinite(v)
        coils[:, j] = np.interp(et, t[fin], v[fin]) if fin.sum() > 1 else np.nan

    ip = coils[:, SIGS.index("plasma_current")].astype(np.float64)
    peak = np.nanmax(np.abs(ip)) if np.isfinite(ip).any() else np.nan
    flat = np.abs(ip) >= 0.8 * peak if np.isfinite(peak) else np.zeros(len(et), dtype=bool)
    idx = np.arange(len(et))
    if flat.any():
        first, last = int(idx[flat].min()), int(idx[flat].max())
        med = float(np.nanmedian(ip[flat]))
    else:
        first, last = len(et), -1
        med = float("nan")
    phase = np.where(idx < first, "rampup", np.where(idx > last, "rampdown", "flattop"))
    s = float(np.sign(med))
    if s not in (-1.0, 1.0):
        s = 1.0
    out = {"coils": coils, "s": s, "phase": phase, "t": et}
    if use_window:
        blocks = np.empty((len(et), len(offsets_ms), len(SIGS)), dtype=np.float32)
        for j, (t, v) in enumerate(series):
            blocks[:, :, j] = _interp_offsets(et, t, v, offsets_ms)
        # The stamp at t is the same array the cache was built from.
        blocks[:, 0, :] = coils
        out["window"] = blocks
    return out


def mirror(coils: np.ndarray, s: float) -> np.ndarray:
    """Multiply ECOILA, F1A–F9B and Ip by s. bcoil is left unchanged.

    `coils` is (T, 21). A window is mirrored one stamp at a time.
    """
    if coils.shape[1] != N_COIL:
        raise ValueError(f"mirror expected {N_COIL} columns, got {coils.shape[1]}")
    out = np.array(coils, dtype=np.float32, copy=True)
    for j, name in enumerate(SIGS):
        if name != "bcoil":
            out[:, j] *= np.float32(s)
    return out


def coil_window(feat: dict, offsets_ms: np.ndarray | None = None) -> np.ndarray:
    """Mirrored currents. With offsets, (T, 21 * n_off) and columns 0–20 are time t."""
    center = mirror(feat["coils"], feat["s"])
    if offsets_ms is None or len(offsets_ms) == 0:
        return center
    if "window" not in feat:
        raise RuntimeError("featurize was called without these offsets")
    blocks = np.asarray(feat["window"])
    if blocks.ndim != 3 or blocks.shape[1] != len(offsets_ms) or blocks.shape[2] != N_COIL:
        raise RuntimeError(f"window shape {blocks.shape} does not match {len(offsets_ms)} offsets")
    mirrored = [mirror(blocks[:, k, :], feat["s"]) for k in range(blocks.shape[1])]
    return np.concatenate(mirrored, axis=1)


def _ip_denom(ip: np.ndarray, floor: float = IP_FLOOR_KA) -> np.ndarray:
    """Mirrored Ip with magnitude kept at least `floor` kA. A zero uses +floor."""
    denom = np.asarray(ip, dtype=np.float64).copy()
    small = np.abs(denom) < floor
    denom[small] = np.where(denom[small] < 0.0, -floor, floor)
    return denom


def _cell_py(table, name):
    """One cell of a one-row parquet, including nested lists."""
    return table.column(name).combine_chunks()[0].as_py()


def _rect_profile(rows) -> np.ndarray:
    """List of equal-length profiles -> (T, C)."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    width = len(rows[0])
    out = np.empty((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) != width:
            raise RuntimeError(f"Thomson profile length {len(row)} does not match {width}")
        out[i] = row
    return out


def _interp_valid(times: np.ndarray, values: np.ndarray, valid: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate valid Thomson rows onto `query`. Missing rows are left out, not treated as zero."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(times)
    if values.ndim == 1:
        values = values[:, None]
    valid = valid & np.isfinite(values).all(axis=1)
    out = np.zeros((len(query), values.shape[1]), dtype=np.float64)
    times, values = times[valid], values[valid]
    if len(times) == 0:
        return out
    order = np.argsort(times)
    times, values = times[order], values[order]
    keep = np.concatenate([[True], np.diff(times) > 0])
    times, values = times[keep], values[keep]
    if len(times) == 1:
        out[:] = values[0]
        return out
    for k in range(values.shape[1]):
        out[:, k] = np.interp(query, times, values[:, k])
    return out


def _group_key(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) < 2:
        raise RuntimeError(f"thomson chord name {name!r} has no group")
    return "_".join(parts[:2])


def radial_group_key(names, radial, vertical) -> str:
    """Widest Thomson group with at least two distinct R and median |Z| within 0.2 m."""
    names = [str(name) for name in names]
    radial = np.asarray(radial, dtype=np.float64)
    vertical = np.asarray(vertical, dtype=np.float64)
    if not (len(names) == len(radial) == len(vertical)):
        raise RuntimeError(
            f"thomson chords {len(names)}, R {len(radial)}, Z {len(vertical)} differ"
        )
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(_group_key(name), []).append(i)
    best_key = None
    best_span = -1.0
    for key, idx in groups.items():
        chosen = np.asarray(idx, dtype=int)
        rr = radial[chosen]
        zz = vertical[chosen]
        finite = np.isfinite(rr) & np.isfinite(zz)
        if int(finite.sum()) < 2:
            continue
        if float(np.median(np.abs(zz[finite]))) > SAFE_Z_MAX_M:
            continue
        if len(np.unique(np.round(rr[finite], 4))) < 2:
            continue
        span = float(rr[finite].max() - rr[finite].min())
        if span > best_span:
            best_span = span
            best_key = key
    if best_key is None:
        raise RuntimeError("no radial midplane Thomson group")
    return best_key


def _profile_system(key: str) -> str:
    if key == "TS_core":
        return "core"
    if key in ("TS_tangential", "TS_edge"):
        return "edge"
    raise RuntimeError(f"radial group {key} has no Te array")


def _align_channels(te: np.ndarray, ne: np.ndarray, n_chord: int, name: str):
    """Drop a leading channel when the profile is one longer than the chord list."""
    te = np.asarray(te, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    if te.shape != ne.shape or te.ndim != 2:
        raise RuntimeError(f"{name}: Te {te.shape} and ne {getattr(ne, 'shape', None)} differ")
    if te.shape[1] == n_chord + 1:
        te = te[:, 1:]
        ne = ne[:, 1:]
    if te.shape[1] != n_chord:
        raise RuntimeError(f"{name}: {te.shape[1]} channels and {n_chord} chords")
    return te, ne


def _shape_at_time(radial: np.ndarray, te: np.ndarray, ne: np.ndarray):
    """log10(p_peak) and p/p_peak on the outboard side, or None when the time is unusable."""
    pe = te * ne
    ok = np.isfinite(pe) & np.isfinite(radial) & (te != 0.0) & (ne != 0.0) & (pe > 0.0)
    if int(ok.sum()) < 2:
        return None
    peak_i = int(np.argmax(np.where(ok, pe, -np.inf)))
    p_peak = float(pe[peak_i])
    r_peak = float(radial[peak_i])
    outboard = ok & (radial >= r_peak)
    rr = radial[outboard]
    pp = pe[outboard]
    order = np.argsort(rr)
    rr, pp = rr[order], pp[order]
    keep = np.concatenate([[True], np.diff(rr) > 1e-6])
    rr, pp = rr[keep], pp[keep]
    if rr.size < 2 or rr[-1] <= r_peak:
        return None
    query = r_peak + _SHAPE_S * (rr[-1] - r_peak)
    shape = np.interp(query, rr, pp / p_peak)
    return np.log10(p_peak), shape


def _pressure_block(radial: np.ndarray, te: np.ndarray, ne: np.ndarray, times: np.ndarray, query: np.ndarray):
    """(T, 9) log10(p_peak) and the 8-point shape, interpolated onto `query`, and whether any
    Thomson time was usable. With none, the block is zeros."""
    n_time = te.shape[0]
    log_peak = np.zeros(n_time, dtype=np.float64)
    shape = np.zeros((n_time, N_SHAPE), dtype=np.float64)
    valid = np.zeros(n_time, dtype=bool)
    for t in range(n_time):
        row = _shape_at_time(radial, te[t], ne[t])
        if row is None:
            continue
        log_peak[t], shape[t] = row
        valid[t] = True
    at_query = _interp_valid(times, np.column_stack([log_peak, shape]), valid, query)
    if at_query.shape != (len(query), 1 + N_SHAPE):
        raise RuntimeError(f"pressure block shape {at_query.shape}")
    return at_query, bool(valid.any())


def shaping_sides(columns, vertical):
    """Upper columns (every Z > 0), lower columns (every Z < 0), and the rest."""
    groups: dict[str, list[float]] = {}
    for column, z in zip(columns, vertical):
        groups.setdefault(str(column), []).append(float(z))
    upper, lower, skipped = [], [], []
    for column, zs in groups.items():
        zz = np.asarray(zs, dtype=np.float64)
        if np.all(zz > 0.0):
            upper.append(column)
        elif np.all(zz < 0.0):
            lower.append(column)
        else:
            skipped.append(column)
    return upper, lower, skipped


def shaping_ratios(upper: np.ndarray, lower: np.ndarray, ip: np.ndarray, floor: float = IP_FLOOR_KA):
    """(I_upper - I_lower) / Ip and sum |I| / |Ip|. Both are unchanged if all three flip sign."""
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    if upper.ndim == 1:
        upper = upper[:, None]
    if lower.ndim == 1:
        lower = lower[:, None]
    denom = _ip_denom(ip, floor)
    asymmetry = (upper.sum(axis=1) - lower.sum(axis=1)) / denom
    magnitude = (np.abs(upper).sum(axis=1) + np.abs(lower).sum(axis=1)) / np.abs(denom)
    return asymmetry, magnitude


def _at_times(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate finite samples onto `query`. Fewer than two samples is zeros."""
    n = min(len(times), len(values))
    tt = np.asarray(times[:n], dtype=np.float64)
    vv = np.asarray(values[:n], dtype=np.float64)
    ok = np.isfinite(tt) & np.isfinite(vv)
    tt, vv = tt[ok], vv[ok]
    out = np.zeros(len(query), dtype=np.float64)
    if tt.size < 2:
        return out
    order = np.argsort(tt, kind="mergesort")
    tt, vv = tt[order], vv[order]
    keep = np.concatenate([[True], np.diff(tt) > 0])
    tt, vv = tt[keep], vv[keep]
    if tt.size < 2:
        return out
    return np.interp(query, tt, vv)


_GEOM_COLS = (
    "efit_times",
    "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    "coil_input_column", "coil_Z",
)
_PROFILE_COLS = (
    "thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
    "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne",
)


def feature_width(n_stamps: int, *, safe: bool, robust: bool = False) -> int:
    """Input width. The image columns are always last. Safe columns sit just before them."""
    if n_stamps < 1:
        raise ValueError(f"expected at least one stamp, got {n_stamps}")
    width = N_COIL * int(n_stamps) + image_width(robust)
    if safe:
        width += safe_width(robust)
    return width


def safe_facts(path: Path, robust: bool = False) -> dict:
    """The shared columns for one shot, plus the group and coil counts used to build them.

    Default: 11 columns, with zeros where Thomson is unusable. `robust`: p_shape_0 dropped,
    Ip floored at 100 kA, NaN where Thomson is unusable, then one 0/1 missing column.
    """
    path = Path(path)
    schema = set(pq.read_schema(path).names)
    geom = [c for c in _GEOM_COLS if c in schema]
    profiles = [c for c in _PROFILE_COLS if c in schema]
    assert_input_columns([*geom, *profiles])
    table = pq.read_table(path, columns=[*geom, *profiles])
    efit_times = np.asarray(_cell_py(table, "efit_times"), dtype=np.float64)
    names = [str(name) for name in _cell_py(table, "thomson_chord_name")]
    chord_r = np.asarray(_cell_py(table, "thomson_chord_R"), dtype=np.float64)
    chord_z = np.asarray(_cell_py(table, "thomson_chord_Z"), dtype=np.float64)
    group = radial_group_key(names, chord_r, chord_z)
    chosen = np.array([_group_key(name) == group for name in names])
    radial = chord_r[chosen]
    system = _profile_system(group)
    te = _rect_profile(_cell_py(table, f"thomson_{system}_Te"))
    ne = _rect_profile(_cell_py(table, f"thomson_{system}_ne"))
    te, ne = _align_channels(te, ne, int(chosen.sum()), path.name)
    if system == "core":
        core_r = np.asarray(_cell_py(table, "thomson_core_R"), dtype=np.float64)
        if core_r.size == int(chosen.sum()) + 1:
            core_r = core_r[1:]
        if core_r.shape != radial.shape or not np.allclose(core_r, radial, equal_nan=True):
            raise RuntimeError(f"{path.name}: core chord R does not match thomson_core_R")
    times = np.asarray(_cell_py(table, f"thomson_{system}_times"), dtype=np.float64)
    if len(times) != len(te):
        raise RuntimeError(f"{path.name}: {len(times)} {system} times and {len(te)} profiles")
    pressure, has_thomson = _pressure_block(radial, te, ne, times, efit_times)
    columns = [str(c) for c in _cell_py(table, "coil_input_column")]
    coil_z = np.asarray(_cell_py(table, "coil_Z"), dtype=np.float64)
    upper, lower, skipped = shaping_sides(columns, coil_z)
    if not upper or not lower:
        raise RuntimeError(f"{path.name}: shaping coils upper {len(upper)} lower {len(lower)}")
    current_cols = [*upper, *lower, "magnetics_time", "magnetics_plasma_current"]
    if "magnetics_plasma_current_times" in schema:
        current_cols.append("magnetics_plasma_current_times")
    assert_input_columns(current_cols)
    currents = pq.read_table(path, columns=current_cols)
    clock = _col(currents, "magnetics_time").astype(np.float64)
    if "magnetics_plasma_current_times" in schema:
        ip_times = fix_d3d_ip_times({
            "magnetics_plasma_current_times": _col(currents, "magnetics_plasma_current_times"),
            "magnetics_plasma_current": _col(currents, "magnetics_plasma_current"),
            "magnetics_time": clock,
        })
    else:
        ip_times = clock
    ip = _at_times(ip_times, _col(currents, "magnetics_plasma_current").astype(np.float64), efit_times)
    up = np.column_stack([
        _at_times(clock, _col(currents, name).astype(np.float64), efit_times) for name in upper
    ])
    lo = np.column_stack([
        _at_times(clock, _col(currents, name).astype(np.float64), efit_times) for name in lower
    ])
    floor = ROBUST_IP_FLOOR_KA if robust else IP_FLOOR_KA
    asymmetry, magnitude = shaping_ratios(up, lo, ip, floor)
    block = np.column_stack([pressure, asymmetry, magnitude])
    if block.shape != (len(efit_times), N_SAFE):
        raise RuntimeError(f"{path.name}: safe block shape {block.shape}")
    if not np.isfinite(block).all():
        raise RuntimeError(f"{path.name}: safe features are not finite")
    if robust:
        if not has_thomson:
            block[:, :1 + N_SHAPE] = np.nan
        missing = np.full((len(block), 1), 0.0 if has_thomson else 1.0)
        block = np.column_stack([block[:, ROBUST_SAFE_KEEP], missing])
    return {
        "block": block.astype(np.float32),
        "group": group,
        "n_upper": len(upper),
        "n_lower": len(lower),
        "skipped": skipped,
    }


def safe_columns(path: Path, robust: bool = False) -> np.ndarray:
    """(T, 11) shared columns at the EFIT times, or (T, 11) in the robust layout. See `safe_facts`."""
    return safe_facts(path, robust)["block"]


def append_safe_features(coils: np.ndarray, path: Path, *, safe: bool, robust: bool = False) -> np.ndarray:
    """Append the shared columns when `safe` is set. The coil window is unchanged."""
    coils = np.asarray(coils)
    if coils.ndim != 2 or coils.shape[1] < N_COIL or coils.shape[1] % N_COIL != 0:
        raise ValueError(f"expected a coil window, got {coils.shape}")
    n_stamps = coils.shape[1] // N_COIL
    if safe:
        block = safe_columns(path, robust)
        if len(block) != len(coils):
            raise RuntimeError(
                f"{Path(path).name}: safe block has {len(block)} frames, the window has {len(coils)}"
            )
        coils = np.concatenate([coils, block], axis=1)
    expect = N_COIL * n_stamps + (safe_width(robust) if safe else 0)
    if coils.shape[1] != expect:
        raise RuntimeError(f"{Path(path).name}: input width {coils.shape[1]}, expected {expect}")
    return coils


def load_coil_pictures(path: Path | None = None) -> dict:
    """Frozen F-coil and ECOILA Green's pictures. Run greens.py if the file is missing."""
    path = Path(path) if path is not None else Path(__file__).resolve().parent / "baseline_pictures.npz"
    if not path.exists():
        raise FileNotFoundError(f"coil pictures are missing ({path}). Run my_experiments/nn/greens.py")
    with np.load(path) as data:
        green = np.ascontiguousarray(data["green"], dtype=np.float64)
        ecoil = np.ascontiguousarray(data["ecoil"], dtype=np.float64)
    if green.shape != (len(F_COILS), N_PSI) or ecoil.shape != (N_PSI,):
        raise RuntimeError(f"{path} shapes green {green.shape} ecoil {ecoil.shape}")
    if not (np.isfinite(green).all() and np.isfinite(ecoil).all()):
        raise RuntimeError(f"{path} has non-finite pictures")
    return {"green": green, "ecoil": ecoil}


def stack_history(block: np.ndarray, k: int, step: int = 1) -> np.ndarray:
    """One shot's feature rows (n, F) -> (n, (2k + 1)·F + 2k): frames i − k·step … i + k·step, `step` frames apart,
    side by side, then one 0/1 column per non-zero offset that is 1 where that frame falls outside the shot.
    Outside the shot the nearest frame is repeated.

    Offsets are in EFIT frames. Training uses step 1 on DIII-D, whose frames are 20 ms apart. Another machine keeps
    the same window in milliseconds with step = 20 ms / its frame spacing (4 on MAST, 5 ms frames)."""
    block = np.asarray(block, dtype=np.float32)
    if k <= 0:
        return block
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    n = len(block)
    i = np.arange(n)
    stamps, flags = [], []
    for off in range(-k, k + 1):
        j = i + off * step
        stamps.append(block[np.clip(j, 0, n - 1)])
        if off:
            flags.append(((j < 0) | (j >= n)).astype(np.float32))
    return np.concatenate(stamps + [np.stack(flags, axis=1)], axis=1)


def history_stats(mean: np.ndarray, std: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-column stats for stack_history's layout: the base stats once per stamp, then 0/1 for the edge flags."""
    if k <= 0:
        return mean, std
    t = 2 * k + 1
    return (np.concatenate([np.tile(mean, t), np.zeros(2 * k, np.float32)]).astype(np.float32),
            np.concatenate([np.tile(std, t), np.ones(2 * k, np.float32)]).astype(np.float32))


def checkpoint_pictures(ckpt_args: dict) -> dict:
    """The coil pictures a checkpoint was trained with (train.py --pictures). Older checkpoints used the default.

    A relative path is taken from the repository root, where train.py is launched.
    """
    path = ckpt_args.get("pictures")
    if path is None:
        return load_coil_pictures()
    path = Path(path)
    return load_coil_pictures(path if path.is_absolute() else REPO / path)


def external_flux(currents: np.ndarray, green: np.ndarray, ecoil: np.ndarray) -> np.ndarray:
    """(T, 4225) vacuum flux from the F-coil sum plus ECOILA. Currents are physical kA or kA·turn."""
    currents = np.asarray(currents, dtype=np.float64)
    if currents.ndim != 2 or currents.shape[1] < N_COIL:
        raise ValueError(f"expected currents (T, {N_COIL}+), got {currents.shape}")
    flux = currents[:, F_SLICE] @ np.asarray(green, dtype=np.float64)
    flux = flux + currents[:, ECOIL_COL:ECOIL_COL + 1] * np.asarray(ecoil, dtype=np.float64)
    if flux.shape != (len(currents), N_PSI) or not np.isfinite(flux).all():
        raise RuntimeError(f"external flux shape {flux.shape} is not finite {(len(currents), N_PSI)}")
    return np.ascontiguousarray(flux, dtype=np.float32)


def _grid_geometry():
    """DIII-D R, Z, their midpoint, and half the radial grid width."""
    r, z = load_rz()
    if not (np.all(np.diff(r) > 0) and np.all(np.diff(z) > 0)):
        raise RuntimeError("the flux grid is not strictly increasing")
    r_ref = 0.5 * (float(r[0]) + float(r[-1]))
    a_grid = 0.5 * (float(r[-1]) - float(r[0]))
    if a_grid <= 0 or r_ref <= 0:
        raise RuntimeError(f"grid scales r_ref {r_ref} a_grid {a_grid}")
    return r, z, r_ref, a_grid


def _bilinear(field: np.ndarray, r: np.ndarray, z: np.ndarray, r0: float, z0: float) -> float:
    """Sample field[Z, R] at one point. Coordinates outside the grid sit on the edge."""
    r0 = float(np.clip(r0, r[0], r[-1]))
    z0 = float(np.clip(z0, z[0], z[-1]))
    ir = int(np.searchsorted(r, r0, side="right") - 1)
    iz = int(np.searchsorted(z, z0, side="right") - 1)
    ir = min(max(ir, 0), len(r) - 2)
    iz = min(max(iz, 0), len(z) - 2)
    tr = (r0 - float(r[ir])) / float(r[ir + 1] - r[ir])
    tz = (z0 - float(z[iz])) / float(z[iz + 1] - z[iz])
    f00 = field[iz, ir]
    f01 = field[iz, ir + 1]
    f10 = field[iz + 1, ir]
    f11 = field[iz + 1, ir + 1]
    return float((1.0 - tz) * ((1.0 - tr) * f00 + tr * f01) + tz * ((1.0 - tr) * f10 + tr * f11))


def _frame_fields(psi: np.ndarray, r: np.ndarray, z: np.ndarray):
    """B_R, B_Z, ∂B_Z/∂R, and ∂²B_Z/∂R² on one 65×65 frame."""
    dpsi_dz, dpsi_dr = np.gradient(psi, z, r, edge_order=1)
    rr = r[None, :]
    br = -dpsi_dz / rr
    bz = dpsi_dr / rr
    _dbz_dz, dbz_dr = np.gradient(bz, z, r, edge_order=1)
    d2bz_dr2 = np.gradient(dbz_dr, r, axis=1, edge_order=1)
    return br, bz, dbz_dr, d2bz_dr2


def _multipole_row(psi, br, bz, dbz_dr, d2, r, z, r_ref, a_grid, ip_ka, floor=IP_FLOOR_KA) -> list[float]:
    """Eight coil-picture columns for one frame."""
    ip_a = float(_ip_denom(np.asarray([ip_ka]), floor)[0]) * 1000.0
    bz0 = _bilinear(bz, r, z, r_ref, 0.0)
    bz_norm = bz0 / (_MU0 * ip_a / r_ref)
    if abs(bz0) < 1e-4:
        decay = 0.0
        hexa = 0.0
    else:
        decay = -(r_ref / bz0) * _bilinear(dbz_dr, r, z, r_ref, 0.0)
        hexa = (r_ref ** 2 / bz0) * _bilinear(d2, r, z, r_ref, 0.0)
    up = _bilinear(psi, r, z, r_ref, 0.5 * a_grid) - _bilinear(psi, r, z, r_ref, -0.5 * a_grid)
    up = up / (_MU0 * r_ref * ip_a)
    bpol = np.hypot(br, bz)
    iz, ir = np.unravel_index(int(np.argmin(bpol)), bpol.shape)
    off = np.abs(z)[:, None] > (0.25 * a_grid)
    if np.any(off):
        masked = np.where(off, bpol, np.inf)
        izx, irx = np.unravel_index(int(np.argmin(masked)), masked.shape)
    else:
        izx, irx = iz, ir
    return [
        bz_norm, decay, hexa, up,
        float(r[ir]) / r_ref, float(z[iz]) / a_grid,
        float(r[irx]) / r_ref, float(z[izx]) / a_grid,
    ]


def _moments_at_time(radial, vertical, te, ne, r_ref, a_grid):
    """Five Thomson moments, or None when the chord cannot support them."""
    pe = np.asarray(te, dtype=np.float64) * np.asarray(ne, dtype=np.float64)
    radial = np.asarray(radial, dtype=np.float64)
    vertical = np.asarray(vertical, dtype=np.float64)
    ok = np.isfinite(pe) & np.isfinite(radial) & np.isfinite(vertical)
    ok = ok & (te != 0.0) & (ne != 0.0) & (pe > 0.0)
    if int(ok.sum()) < 2:
        return None
    peak_i = int(np.argmax(np.where(ok, pe, -np.inf)))
    rr = radial[ok]
    pp = pe[ok]
    order = np.argsort(rr)
    rr, pp = rr[order], pp[order]
    keep = np.concatenate([[True], np.diff(rr) > 1e-6])
    rr, pp = rr[keep], pp[keep]
    if rr.size < 2:
        return None
    mean_r = float(np.average(rr, weights=pp))
    width = float(np.sqrt(np.average((rr - mean_r) ** 2, weights=pp))) / a_grid
    dp = np.gradient(pp, rr, edge_order=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(dp) > 0.0, pp / np.abs(dp), np.inf)
    usable = scale[np.isfinite(scale) & (scale > 0.0)]
    steep = float(usable.min()) / a_grid if usable.size else 0.0
    return (
        float(radial[peak_i]) / r_ref,
        float(vertical[peak_i]) / a_grid,
        float(pe[peak_i] / pp.mean()),
        width,
        steep,
    )


def _moment_block(radial, vertical, te, ne, times, query, r_ref, a_grid):
    """(T, 5) Thomson moments interpolated onto `query`, and whether any Thomson time was usable.
    With none, the block is zeros."""
    n_time = te.shape[0]
    rows = np.zeros((n_time, 5), dtype=np.float64)
    valid = np.zeros(n_time, dtype=bool)
    for t in range(n_time):
        row = _moments_at_time(radial, vertical, te[t], ne[t], r_ref, a_grid)
        if row is None:
            continue
        rows[t] = row
        valid[t] = True
    at_query = _interp_valid(times, rows, valid, query)
    if at_query.shape != (len(query), 5):
        raise RuntimeError(f"Thomson moment shape {at_query.shape}")
    return at_query, bool(valid.any())


def _thomson_moments(path: Path, query: np.ndarray, r_ref: float, a_grid: float):
    """(T, 5) moments on the same radial midplane chord as the safe block, and whether any
    Thomson time was usable."""
    path = Path(path)
    schema = set(pq.read_schema(path).names)
    geom = [c for c in _GEOM_COLS if c in schema]
    profiles = [c for c in _PROFILE_COLS if c in schema]
    assert_input_columns([*geom, *profiles])
    table = pq.read_table(path, columns=[*geom, *profiles])
    names = [str(name) for name in _cell_py(table, "thomson_chord_name")]
    chord_r = np.asarray(_cell_py(table, "thomson_chord_R"), dtype=np.float64)
    chord_z = np.asarray(_cell_py(table, "thomson_chord_Z"), dtype=np.float64)
    group = radial_group_key(names, chord_r, chord_z)
    chosen = np.array([_group_key(name) == group for name in names])
    radial = chord_r[chosen]
    vertical = chord_z[chosen]
    system = _profile_system(group)
    te = _rect_profile(_cell_py(table, f"thomson_{system}_Te"))
    ne = _rect_profile(_cell_py(table, f"thomson_{system}_ne"))
    te, ne = _align_channels(te, ne, int(chosen.sum()), path.name)
    times = np.asarray(_cell_py(table, f"thomson_{system}_times"), dtype=np.float64)
    if len(times) != len(te):
        raise RuntimeError(f"{path.name}: {len(times)} {system} times and {len(te)} profiles")
    return _moment_block(radial, vertical, te, ne, times, query, r_ref, a_grid)


def image_columns(path: Path, stamp: np.ndarray, green: np.ndarray, ecoil: np.ndarray,
                  robust: bool = False) -> np.ndarray:
    """(T, 18) trunk columns at the stamp's times. The stamp is the mirrored currents at time t.

    dIp/dt is in kA/s. The loop-voltage proxy is in volts. A shot with no usable Thomson
    time contributes zeros for the five moments, and a single-frame shot zeros for the two rates.

    `robust` returns the (T, 14) layout instead: Ip floored at 100 kA, |q_cyl|, the columns in
    ROBUST_IMAGE_KEEP, NaN for the moments and rates that cannot be computed, then the
    missing-Thomson and missing-rate flags.
    """
    path = Path(path)
    stamp = np.asarray(stamp, dtype=np.float64)
    if stamp.ndim != 2 or stamp.shape[1] < N_COIL:
        raise ValueError(f"expected a current stamp, got {stamp.shape}")
    assert_input_columns(["efit_times"])
    times = np.asarray(_cell_py(pq.read_table(path, columns=["efit_times"]), "efit_times"), dtype=np.float64)
    if len(times) != len(stamp):
        raise RuntimeError(f"{path.name}: {len(times)} EFIT times and {len(stamp)} current rows")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise RuntimeError(f"{path.name}: efit_times are not strictly increasing")
    r, z, r_ref, a_grid = _grid_geometry()
    floor = ROBUST_IP_FLOOR_KA if robust else IP_FLOOR_KA
    flux = external_flux(stamp, green, ecoil).reshape(-1, GRID, GRID)
    rows = np.zeros((len(stamp), N_IMAGE), dtype=np.float64)
    center = np.empty(len(stamp), dtype=np.float64)
    for t in range(len(stamp)):
        br, bz, dbz_dr, d2 = _frame_fields(flux[t], r, z)
        rows[t, :8] = _multipole_row(flux[t], br, bz, dbz_dr, d2, r, z, r_ref, a_grid, stamp[t, IP_COL], floor)
        center[t] = _bilinear(flux[t], r, z, r_ref, 0.0)
    ip_a = _ip_denom(stamp[:, IP_COL], floor) * 1000.0
    bt = _MU0 * D3D_TF_TURNS * (stamp[:, BCOIL_COL] * 1000.0) / (2.0 * np.pi * r_ref)
    t_s = times * 1e-3
    if len(times) == 1:
        dpsi = np.zeros(1, dtype=np.float64)
        dip = np.zeros(1, dtype=np.float64)
    else:
        dpsi = np.gradient(center, t_s, edge_order=1)
        dip = np.gradient(stamp[:, IP_COL], t_s, edge_order=1)
    rows[:, 8] = bt
    rows[:, 9] = 2.0 * np.pi * a_grid ** 2 * bt / (_MU0 * r_ref * ip_a)
    rows[:, 10] = -dpsi
    rows[:, 11] = dip
    rows[:, 12] = a_grid / r_ref
    rows[:, 13:], has_thomson = _thomson_moments(path, times, r_ref, a_grid)
    if not np.isfinite(rows).all():
        raise RuntimeError(f"{path.name}: image columns are not finite")
    if robust:
        # q95 is a magnitude. The field direction is still in bcoil and Bt.
        rows[:, 9] = np.abs(rows[:, 9])
        has_rate = len(times) > 1
        if not has_thomson:
            rows[:, 13:] = np.nan
        if not has_rate:
            rows[:, 10:12] = np.nan
        flags = np.column_stack([
            np.full(len(rows), 0.0 if has_thomson else 1.0),
            np.full(len(rows), 0.0 if has_rate else 1.0),
        ])
        rows = np.column_stack([rows[:, ROBUST_IMAGE_KEEP], flags])
    return np.ascontiguousarray(rows, dtype=np.float32)


def append_image_features(coils: np.ndarray, path: Path, pictures: dict,
                          robust: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Append the image columns. Return the wider input and the physical coil picture (T, 4225)."""
    coils = np.asarray(coils)
    if coils.ndim != 2 or coils.shape[1] < N_COIL:
        raise ValueError(f"expected coil columns first, got {coils.shape}")
    stamp = coils[:, :N_COIL]
    block = image_columns(path, stamp, pictures["green"], pictures["ecoil"], robust)
    if len(block) != len(coils):
        raise RuntimeError(f"{Path(path).name}: image block has {len(block)} frames, the window has {len(coils)}")
    picture = external_flux(stamp, pictures["green"], pictures["ecoil"])
    return np.concatenate([coils, block], axis=1).astype(np.float32), picture


def shot_parquet(data_root: Path, row) -> Path:
    name = getattr(row, "filename", None) or f"{row.shot}.parquet"
    return Path(data_root) / "data" / TRAIN_CONFIG / str(name)


def inputs_for_shot(path: Path, offsets_ms: np.ndarray, cached: np.ndarray | None = None) -> np.ndarray:
    """Mirrored window for one shot. With a cache slice, the stamp at t must match it."""
    feat = featurize(path, offsets_ms if offsets_ms is not None and len(offsets_ms) else None)
    coils = coil_window(feat, offsets_ms)
    if cached is None:
        return coils
    cached = np.asarray(cached)
    if cached.shape[1] < N_COIL:
        raise RuntimeError(f"{path.name}: cache coils have {cached.shape[1]} columns")
    if len(cached) != len(coils):
        raise RuntimeError(f"{path.name}: window has {len(coils)} frames, cache has {len(cached)}")
    stamp = coils[:, :N_COIL]
    ref = cached[:, :N_COIL]
    if not np.allclose(stamp, ref, rtol=0.0, atol=_CACHE_ATOL, equal_nan=True):
        diff = np.abs(stamp.astype(np.float64) - ref.astype(np.float64))
        finite = diff[np.isfinite(diff)]
        worst = float(finite.max()) if finite.size else float("inf")
        raise RuntimeError(f"{path.name}: coil readings at t differ from the cache by {worst:.3g}")
    return coils


def score_subset(shots, n: int = 100, n_reversed: int = 10, seed: int = 0, split: str = "val"):
    """100 fixed shots of `split`, seeded, with at least `n_reversed` reversed-current shots.

    `shots` is the shot-level table written by build_cache (one row per shot). The val subset
    is the one build_cache saves. The train subset measures S on shots the model trained on.
    """
    val = shots[shots["split"] == split].sort_values("filename")
    rev = val[val["s"] < 0]
    normal = val[val["s"] > 0]
    if len(rev) < n_reversed:
        raise RuntimeError(f"{split} split has {len(rev)} reversed shots, need {n_reversed}")
    if len(normal) < n - n_reversed:
        raise RuntimeError(f"{split} split has {len(normal)} normal shots, need {n - n_reversed}")
    import pandas as pd
    rng = np.random.RandomState(seed)
    take_rev = rng.choice(len(rev), n_reversed, replace=False)
    take_normal = rng.choice(len(normal), n - n_reversed, replace=False)
    chosen = pd.concat([rev.iloc[take_rev], normal.iloc[take_normal]], ignore_index=True)
    # Stable order so the npz and the plots do not depend on the rng draw order.
    order = np.argsort(chosen["filename"].to_numpy())
    return chosen.iloc[order].reset_index(drop=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line, exclusively locked so train.py and score.py can share the file."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(record)) + "\n"
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def load_rz(repo: Path | None = None):
    """1-D R and Z axes of the DIII-D 65×65 grid, in meters. axis0 of ψ is Z, axis1 is R."""
    mask_path = (repo or REPO) / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)
    r = np.asarray(mask["grid_R"], dtype=np.float64)
    z = np.asarray(mask["grid_Z"], dtype=np.float64)
    if r.shape != (GRID,) or z.shape != (GRID,):
        raise RuntimeError(
            f"{mask_path} grid_R {r.shape} grid_Z {z.shape}, expected ({GRID},) each"
        )
    return r, z
