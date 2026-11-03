"""features_v4 with every column dimensionless and every Thomson column measured over the same span.

Two changes from v4, both so that a value means the same physics on DIII-D and MAST:

- Ip_MA and log10_p_peak are left out. They are absolute (MA, eV m⁻³) and so depend on machine size. Ip
  still enters through q_ITER, betaN_e and the flux scales; the pressure through betaN_e.
- Thomson channels are restricted to (R − R0)/a in THOMSON_WINDOW on both machines. DIII-D's radial
  midplane chord covers about −0.01 to 0.59 (median over train shots; a few layouts reach 0.9), MAST's
  covers the whole plasma. Without the restriction the pressure peak, the axis-shift proxy and the
  profiles' reference point are searched over different regions on the two machines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import features_v2 as fv
import features_v3 as f3
import features_v4 as f4

THOMSON_WINDOW = (-0.02, 0.60)
DROPPED = ("Ip_MA", "log10_p_peak")
FLAGS = fv.FLAGS
FEATURES = [n for n in f4.FEATURES if n not in DROPPED]
N_FEATURES = len(FEATURES)
VERSION = "v5"
_KEEP_V2 = [i for i, n in enumerate(fv.FEATURES) if n not in DROPPED and n not in FLAGS]


def shot_features(path: Path, pictures: dict) -> np.ndarray:
    """(T, N_FEATURES) float32: the v2 columns without DROPPED, the harmonics, the patch, then the flags."""
    path = Path(path)
    base, extra = fv.shot_features(path, pictures, return_inputs=True, thomson_window=THOMSON_WINDOW)
    out = assemble(base, extra, pictures)
    if out.shape[1] != N_FEATURES:
        raise RuntimeError(f"{path.name}: {out.shape[1]} columns, expected {N_FEATURES}")
    return out


def assemble(base: np.ndarray, extra: dict, pictures: dict) -> np.ndarray:
    """v5 columns from the v2 output and inputs of fv.shot_features(..., thomson_window=THOMSON_WINDOW)."""
    harm = f3.harmonics(extra["cur"], extra["ip"], pictures, extra["machine"])
    grid = f4.patch(extra["cur"], extra["ip"], pictures, extra["machine"])
    n_flag = len(FLAGS)
    out = np.concatenate([base[:, _KEEP_V2], harm, grid, base[:, -n_flag:]], axis=1)
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)
