"""Challenge 1 (DIII-D) model: physics stage + learned plasma residual.

    psi(t) = -sum_F I_F(t) G_F                    F-coils, turns folded into the currents: no fit
           + a_E I_E(t) G_E + a_p Ip(t) G_p + c   E-coil (ambiguous turns), plasma filament, offset
           + Ridge(features(t)) -> 65x65 residual  plasma current distribution, vessel eddies, ...

    q95, betaN = Ridge(features(t))

Inputs are taken in the shot's Ip > 0 frame (features.shot_inputs flips every current of a
reversed-Ip shot). The target psi is NOT flipped: the shipped psirz already follows an
Ip-normalised sign convention (on reversed-Ip shot 83814d2061 the physics stage matches the
target at R2 0.90 unflipped, -4.5 flipped).

Both stages are fitted from running sums (sufficient statistics), so memory is flat in the number
of shots and a fit over the full training split needs no GPU. `fit` takes a zero-argument callable
returning a fresh iterator of training rows because it makes two passes: one for the physics
coefficients and feature scaling, one for the Ridge sums.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from features import shot_inputs
from greens import coil_basis, filament_psi

HERE = Path(__file__).resolve().parent
MASK = np.load(HERE.parents[1] / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GRID_R, GRID_Z = MASK["grid_R"], MASK["grid_Z"]
ENVELOPE = MASK["mask_coarse"].astype(bool)
PLASMA_FILAMENT = (1.7, 0.0)
ALPHAS = np.logspace(-3, 4, 15)
Z_CLIP = 5.0                    # standardized features are clipped to +-Z_CLIP
SCALAR_KEYS = [("q95", "efit_q95"), ("betaN", "efit_beta_n")]


def target_psi(row) -> np.ndarray:
    raw = row["efit_psirz"]
    try:
        psi = np.asarray(raw, dtype=np.float64)
        if psi.ndim == 3:
            return psi
    except (ValueError, TypeError):
        pass
    return np.stack([np.stack([np.asarray(r, dtype=np.float64) for r in f]) for f in raw])


class _RidgeSums:
    """Uncentered sums for multi-output Ridge on standardized features, split by fold."""

    def __init__(self, n_feat: int, n_out: int, n_folds: int):
        self.n = np.zeros(n_folds)
        self.sz = np.zeros((n_folds, n_feat))
        self.szz = np.zeros((n_folds, n_feat, n_feat))
        self.sy = np.zeros((n_folds, n_out))
        self.szy = np.zeros((n_folds, n_feat, n_out))
        self.syy = np.zeros((n_folds, n_out))

    def add(self, fold: int, z: np.ndarray, y: np.ndarray):
        self.n[fold] += len(z)
        self.sz[fold] += z.sum(0)
        self.szz[fold] += z.T @ z
        self.sy[fold] += y.sum(0)
        self.szy[fold] += z.T @ y
        self.syy[fold] += (y * y).sum(0)

    @staticmethod
    def _solve(n, sz, szz, sy, szy, alphas):
        czz = szz - np.outer(sz, sz) / n
        czy = szy - np.outer(sz, sy) / n
        lam, V = np.linalg.eigh(czz)
        proj = V.T @ czy
        for a in alphas:
            w = V @ (proj / (np.maximum(lam, 0) + a)[:, None])
            yield a, w, (sy - sz @ w) / n

    def select_alpha(self, alphas=ALPHAS) -> tuple[float, dict]:
        """Shot-grouped K-fold CV, exact from the sums: pick alpha with least held-out SSE."""
        used = np.nonzero(self.n)[0]
        sse = np.zeros(len(alphas))
        if len(used) < 2:
            return float(alphas[len(alphas) // 2]), {}
        tot = [s.sum(0) for s in (self.sz, self.szz, self.sy, self.szy)]
        for f in used:
            tr = [t - s[f] for t, s in zip(tot, (self.sz, self.szz, self.sy, self.szy))]
            for i, (_, w, b) in enumerate(self._solve(self.n.sum() - self.n[f], *tr, alphas)):
                sse[i] += (self.syy[f].sum() - 2 * b @ self.sy[f] - 2 * np.sum(w * self.szy[f])
                           + self.n[f] * b @ b + 2 * (self.sz[f] @ w) @ b
                           + np.sum(w * (self.szz[f] @ w)))
        return float(alphas[int(np.argmin(sse))]), dict(zip(alphas.tolist(), sse.tolist()))

    def fit(self, alpha: float):
        tot = [s.sum(0) for s in (self.sz, self.szz, self.sy, self.szy)]
        _, w, b = next(self._solve(self.n.sum(), *tot, [alpha]))
        return w, b


class C1Model:
    def __init__(self, n_folds: int = 5, alphas=ALPHAS, physics=()):
        self.n_folds, self.alphas = n_folds, np.asarray(alphas)
        self.physics_opts = tuple(physics)  # optional extra inputs, see physics_features.py
        self._basis_cache: dict = {}

    def _inputs(self, row):
        inp = shot_inputs(row)
        if getattr(self, "physics_opts", ()):  # models pickled before this option have none
            from physics_features import add_physics
            G, _ = self._basis(row, inp["cols"])
            inp = add_physics(row, inp, self.physics_opts, G, GRID_R, GRID_Z)
        return inp

    # ------------------------------------------------------------------ physics stage
    def _basis(self, row, cols):
        key = tuple(np.round(np.concatenate([np.asarray(row[k], dtype=float) for k in
                    ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1",
                     "coil_angle2")]), 5))
        if key not in self._basis_cache:
            names, G = coil_basis(row, GRID_R, GRID_Z)
            RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
            Gp = filament_psi(*PLASMA_FILAMENT, RR, ZZ)
            self._basis_cache[key] = (names, G, Gp)
        names, G, Gp = self._basis_cache[key]
        assert names == cols, "coil geometry rows out of order with the current columns"
        return G, Gp

    def _physics_terms(self, row, inp):
        """fixed (T,65,65) and the free basis (T,3,65,65) for a_E, a_p, c."""
        G, Gp = self._basis(row, inp["cols"])
        is_f = np.array([c != "magnetics_ECOILA" for c in inp["cols"]])
        I = np.nan_to_num(inp["I"])
        fixed = -np.einsum("tc,cij->tij", I[:, is_f], G[is_f])
        ie = I[:, ~is_f].sum(1)
        ip = np.nan_to_num(inp["ip"])
        free = np.stack([ie[:, None, None] * G[~is_f].sum(0), ip[:, None, None] * Gp,
                         np.ones((len(ip), *Gp.shape))], 1)
        return fixed, free

    def physics(self, row, inp):
        fixed, free = self._physics_terms(row, inp)
        return fixed + np.einsum("tkij,k->tij", free, self.theta)

    # ------------------------------------------------------------------ learned stage
    def _z(self, X):
        z = (X - self.mu) / self.sd
        z = np.where(np.isfinite(z), z, 0.0)          # missing input -> training mean
        # A feature with little spread in training (e.g. Thomson channels outside every training
        # plasma) can land thousands of sd away on a new shot; clip so the linear residual cannot
        # swamp the physics stage there.
        return np.clip(z, -Z_CLIP, Z_CLIP)

    def fit(self, rows: Callable[[], Iterable[dict]], verbose: bool = True):
        # pass 1: physics coefficients (fit outside the plasma envelope) + feature moments
        ata, aty, nx, sx, sxx, n_shots = np.zeros((3, 3)), np.zeros(3), 0, 0, 0, 0
        for row in rows():
            inp = self._inputs(row)
            psi = target_psi(row)
            fixed, free = self._physics_terms(row, inp)
            A = free[:, :, ~ENVELOPE].transpose(0, 2, 1).reshape(-1, 3)
            y = (psi - fixed)[:, ~ENVELOPE].ravel()
            ata += A.T @ A
            aty += A.T @ y
            X = inp["X"]
            ok = np.isfinite(X)
            nx = nx + ok.sum(0)
            sx = sx + np.where(ok, X, 0).sum(0)
            sxx = sxx + np.where(ok, X * X, 0).sum(0)
            self.names = inp["names"]
            n_shots += 1
        self.theta = np.linalg.solve(ata, aty)
        self.mu = sx / np.maximum(nx, 1)
        self.sd = np.sqrt(np.maximum(sxx / np.maximum(nx, 1) - self.mu ** 2, 0))
        self.sd[self.sd < 1e-9] = np.inf                # constant / never-seen feature -> 0
        if verbose:
            print(f"  physics: a_E={self.theta[0]:.4g} a_p={self.theta[1]:.4g} "
                  f"c={self.theta[2]:.4g}  ({n_shots} shots)")

        # pass 2: Ridge sums for the psi residual and the two scalars, grouped by shot fold
        n_folds = min(self.n_folds, n_shots)
        F, P = len(self.mu), ENVELOPE.size
        psi_sums = _RidgeSums(F, P, n_folds)
        scal_sums = _RidgeSums(F, len(SCALAR_KEYS), n_folds)
        for s, row in enumerate(rows()):
            inp = self._inputs(row)
            psi = target_psi(row)
            z = self._z(inp["X"])
            resid = (psi - self.physics(row, inp)).reshape(len(z), -1)
            psi_sums.add(s % n_folds, z, resid)
            S = np.column_stack([np.asarray(row[k], dtype=np.float64) for _, k in SCALAR_KEYS])
            ok = np.isfinite(S).all(1)
            scal_sums.add(s % n_folds, z[ok], S[ok])

        self.alpha_psi, cv_psi = psi_sums.select_alpha(self.alphas)
        self.W, self.b = psi_sums.fit(self.alpha_psi)
        self.alpha_scal, _ = scal_sums.select_alpha(self.alphas)
        self.Ws, self.bs = scal_sums.fit(self.alpha_scal)
        if verbose:
            print(f"  ridge: alpha_psi={self.alpha_psi:g}  alpha_scalars={self.alpha_scal:g}  "
                  f"({F} features, {n_folds} CV folds)")
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, row) -> dict:
        inp = self._inputs(row)
        z = self._z(inp["X"])
        psi = self.physics(row, inp) + (z @ self.W + self.b).reshape(len(z), *ENVELOPE.shape)
        S = z @ self.Ws + self.bs
        out = {"psirz": psi.astype(np.float32)}
        for j, (name, _) in enumerate(SCALAR_KEYS):
            out[name] = S[:, j].astype(np.float32)
        return out

    def save(self, path: Path):
        self._basis_cache = {}
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "C1Model":
        with open(path, "rb") as fh:
            return pickle.load(fh)
