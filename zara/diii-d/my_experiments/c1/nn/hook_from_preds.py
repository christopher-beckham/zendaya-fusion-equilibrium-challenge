"""`your_model_predict` for submission_skeleton.py that serves precomputed per-shot predictions.

Set $NN_PREDS to a directory of <stem>.npz files (from predict_test.py: psi, q95, betaN, t). Each
streamed DIII-D row is matched to a cached shot by its exact `efit_times` array (unique per shot),
so no assumption about the Hugging Face stream order is needed. MAST rows get zeros (Challenge 1).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_INDEX: dict | None = None


def _key(t: np.ndarray) -> tuple:
    t = np.asarray(t, dtype=np.float64)
    return (len(t), tuple(np.round(t[:: max(1, len(t) // 16)], 3).tolist()), round(float(t.sum()), 2))


def _load_index() -> dict:
    d = Path(os.environ["NN_PREDS"])
    index = {}
    for f in sorted(d.glob("*.npz")):
        t = np.load(f)["t"]
        k = _key(t)
        assert k not in index, f"two cached shots share a time axis: {f.name} and {index[k].name}"
        index[k] = f
    return index


def your_model_predict(row: dict, source: str) -> dict:
    global _INDEX
    t = np.asarray(row["efit_times"], dtype=np.float64)
    T = len(t)
    if source != "DIII-D":
        return {"psirz": np.zeros((T, 65, 65), np.float32), "q95": np.zeros(T, np.float32),
                "betaN": np.zeros(T, np.float32)}
    if _INDEX is None:
        _INDEX = _load_index()
    f = _INDEX.get(_key(t))
    if f is None:
        raise KeyError(f"no cached prediction for a DIII-D shot with {T} frames starting at t={t[0]}")
    p = np.load(f)
    assert np.allclose(p["t"], t, atol=1e-3), f"time axis mismatch for {f.name}"
    return {"psirz": p["psi"].astype(np.float32), "q95": p["q95"].astype(np.float32),
            "betaN": p["betaN"].astype(np.float32)}
