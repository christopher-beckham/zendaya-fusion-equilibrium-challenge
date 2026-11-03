"""The MAST plasma region used by our pipeline: one switch, read in one place. DIII-D is not affected.

    FEC_MAST_MASK=envelope   (default) fusion_scoring/masks/mast_envelope.npz, mask_coarse: every result before the
                             switch existed was made with this
    FEC_MAST_MASK=limiter    geom_mask/mast_mask_limiter.npz: cell centres inside Zara's hand-drawn MAST-like limiter
                             (geom_mask/build_masks.py; machine geometry only, no EFIT data)

The competition rules allow the envelope only inside the official scoring derivation, so a run that must not use it
sets the limiter. train_mm.py --mast-mask sets the variable before anything loads; other scripts read it as is.

    name()          the selected region ("envelope" or "limiter")
    select(n)       select a region for this process and its children (sets the environment variable)
    mask(n)         (65, 65) bool, rows Z, columns R, on the MAST efit grid
    region(n)       {"r", "z", "mask"}: the grid the mask lives on and the mask
    suffix(n)       "" for the envelope, "_limiter" otherwise: appended to every cached file that depends on the region

With the limiter selected, any attempt in this process to open mast_envelope.npz raises (guard installed on import
and by select()), so no code path can fall back to the envelope unnoticed.
"""
from __future__ import annotations

import builtins
import io
import os
from pathlib import Path

import numpy as np

ENV_VAR = "FEC_MAST_MASK"
CHOICES = ("envelope", "limiter")
DEFAULT = "envelope"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENVELOPE = REPO / "fusion_scoring" / "masks" / "mast_envelope.npz"
_LIMITER_LOCAL = HERE / "geom_mask" / "mast_mask_limiter.npz"
_LIMITER_SHARED = HERE.parents[2] / "multimachine" / "my_experiments" / "nn" / "geom_mask" / "mast_mask_limiter.npz"
LIMITER = _LIMITER_LOCAL if _LIMITER_LOCAL.exists() else _LIMITER_SHARED


def _check(n: str) -> str:
    if n not in CHOICES:
        raise ValueError(f"{ENV_VAR}={n!r}: expected one of {CHOICES}")
    return n


def name() -> str:
    return _check(os.environ.get(ENV_VAR, "") or DEFAULT)


def explicit() -> bool:
    """True when the environment variable is set (not just the default)."""
    return bool(os.environ.get(ENV_VAR, ""))


def select(n: str) -> str:
    os.environ[ENV_VAR] = _check(n)
    if n != "envelope":
        install_guard()
    return n


def suffix(n: str | None = None) -> str:
    n = name() if n is None else _check(n)
    return "" if n == "envelope" else f"_{n}"


def region(n: str | None = None) -> dict:
    n = name() if n is None else _check(n)
    if n == "envelope":
        with np.load(ENVELOPE) as z:
            return {"r": np.asarray(z["grid_R"], np.float64), "z": np.asarray(z["grid_Z"], np.float64),
                    "mask": np.asarray(z["mask_coarse"], bool)}
    with np.load(LIMITER) as z:
        out = {"r": np.asarray(z["r"], np.float64), "z": np.asarray(z["z"], np.float64), "mask": np.asarray(z["mask"], bool)}
    if out["mask"].shape != (65, 65):
        raise RuntimeError(f"{LIMITER}: mask {out['mask'].shape}, expected (65, 65)")
    return out


def mask(n: str | None = None) -> np.ndarray:
    return region(n)["mask"]


# ---------------------------------------------------------------------------------------------------------------
# guard: with the limiter selected, opening the envelope file raises
# ---------------------------------------------------------------------------------------------------------------

_GUARD = {"on": False}


def _is_envelope(path) -> bool:
    try:
        return os.path.basename(os.fspath(path)) == "mast_envelope.npz"
    except TypeError:
        return False


def install_guard() -> None:
    if _GUARD["on"]:
        return
    _GUARD["on"] = True
    orig_open, orig_io_open, orig_load = builtins.open, io.open, np.load

    def _raise(path):
        raise RuntimeError(f"{ENV_VAR}={name()}: refusing to open {path} (the scorer's MAST envelope)")

    def guarded_open(file, *a, **k):
        if name() != "envelope" and _is_envelope(file):
            _raise(file)
        return orig_open(file, *a, **k)

    def guarded_io_open(file, *a, **k):
        if name() != "envelope" and _is_envelope(file):
            _raise(file)
        return orig_io_open(file, *a, **k)

    def guarded_load(file, *a, **k):
        if name() != "envelope" and _is_envelope(file):
            _raise(file)
        return orig_load(file, *a, **k)

    builtins.open, io.open, np.load = guarded_open, guarded_io_open, guarded_load


if name() != "envelope":
    install_guard()
