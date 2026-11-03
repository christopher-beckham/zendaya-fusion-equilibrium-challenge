"""The starter kit's "Ridge (CV)" baseline from experiments.py, behind the C1Model interface.

21 DIII-D magnetics (19 PF coils, bcoil, Ip with the corrected time axis) linearly interpolated
onto efit_times -> StandardScaler -> RidgeCV onto 50 PCA coefficients of psi; q95 and betaN by
their own RidgeCV. No Thomson (experiments.py's default).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_fixes import fix_d3d_ip_times  # noqa: E402
from experiments import (D3D_MAGNETICS_SIGNALS, TargetPCA,  # noqa: E402
                         interpolate_magnetics_to_efit)

from model import SCALAR_KEYS, target_psi  # noqa: E402

N_PCA = 50
ALPHAS = np.logspace(-3, 3, 10)


def magnetics_features(row) -> np.ndarray:
    """experiments.interpolate_magnetics_to_efit on a row that may lack the targets."""
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_t = fix_d3d_ip_times(row)
    shot = {"efit_times": np.asarray(row["efit_times"], dtype=np.float64), "magnetics": {}}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col in row:
            shot["magnetics"][sig] = {"values": np.asarray(row[col], dtype=np.float32),
                                      "times": ip_t if sig == "plasma_current" else mag_t}
    return interpolate_magnetics_to_efit(shot)


class BaselineModel:
    def fit(self, rows: Callable[[], Iterable[dict]], verbose: bool = True):
        X, Y, S = [], [], []
        for row in rows():
            X.append(magnetics_features(row))
            Y.append(target_psi(row).astype(np.float32))
            S.append(np.column_stack([np.asarray(row[k], dtype=np.float64)
                                      for _, k in SCALAR_KEYS]))
        X, Y, S = np.concatenate(X), np.concatenate(Y), np.concatenate(S)
        ok = np.isfinite(X).all(1) & np.isfinite(Y.reshape(len(Y), -1)).all(1)
        X, Y, S = X[ok], Y[ok], S[ok]
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = TargetPCA(min(N_PCA, len(Z))).fit(Y)
        self.ridge = RidgeCV(alphas=ALPHAS).fit(Z, self.pca.transform(Y))
        self.scal = []
        for j in range(S.shape[1]):
            m = np.isfinite(S[:, j])
            self.scal.append(RidgeCV(alphas=ALPHAS).fit(Z[m], S[m, j]))
        if verbose:
            print(f"  baseline: {len(Z)} frames, PCA {self.pca.n_components} "
                  f"({np.cumsum(self.pca.explained_variance_ratio)[-1]:.4%} var), "
                  f"alpha={self.ridge.alpha_:g}")
        return self

    def predict(self, row) -> dict:
        X = magnetics_features(row)
        X = np.where(np.isfinite(X), X, self.scaler.mean_)     # every frame must be predicted
        Z = self.scaler.transform(X)
        out = {"psirz": self.pca.inverse_transform(self.ridge.predict(Z))}
        for (name, _), m in zip(SCALAR_KEYS, self.scal):
            out[name] = m.predict(Z).astype(np.float32)
        return out
