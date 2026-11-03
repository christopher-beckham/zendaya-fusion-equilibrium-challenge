"""ml/ch1/train_c1.py = copy of ml/muon/train_px.py (2026-10-07) with Challenge-1 loss / regularisation options (ch1 workstream).

Additions relative to ml/muon/train_px.py (all off by default, so the defaults reproduce train_px.py):
  --scal-range Q_LO,Q_HI,B_LO,B_HI   physical-range filter on the q95 / betaN training labels (frames outside -> no scalar loss)
  --scal-w WQ,WB                      per-scalar loss multipliers (q95, betaN)
  --w-axis W --w-axis-sigma S         extra pixel weight 1 + W exp(-d^2 / 2 S^2) around the true magnetic axis (d in cells),
                                      multiplied into the LCFS-band weights and renormalised to mean 1 per frame
  --lam-grad L --grad-w {env,band,lcfs}  gradient-matching loss (scorer's li uses |grad psi| / R inside the LCFS and on it):
                                      env = uniform over the envelope (train_px behaviour), band = the psi-loss pixel weights,
                                      lcfs = inside the true LCFS polygon (64 points, dilated 1 cell) weighted by R0 / R (dV / R^2)
  --ema D [--ema-start F]             exponential moving average of the weights from fraction F of the steps; the final
                                      evaluation, checkpoint and predictions use the averaged weights (raw-weight metrics logged)
  --in-noise S                        Gaussian noise (sd S, standardised units) on the network inputs during training
  --bag-frac F --bag-seed K           train on a random fraction F of the training shots (holdout split unchanged)

--- docstring of ml/muon/train_px.py ---
ml/muon/train_px.py = copy of c1/nn/train_v7r.py (2026-10-06 22:50 UTC) with an optimiser switch (Muon workstream).

Changes relative to train_v7r.py (nothing else differs; --opt adamw reproduces train_v7r bit-for-bit up to GPU nondeterminism):
  --opt {adamw,muon} and the Muon options of muon_opt.py. With --opt muon the hidden 2D weight matrices (blocks fc1/fc2 and the
  first layer of the scalar head, scal.0) use torch.optim.Muon; biases, LayerNorms and (unless --muon-inp / --muon-head) the
  input layer `inp` and the output layers `head` (65x65 psi residual) and `scal.2` use AdamW with --lr/--wd as before. One
  schedule (warm-up + cosine) multiplies both optimisers' learning rates. --width-scal: width of the scalar head (default 256).
  Lower-memory loading (arrays filled in place instead of concatenated; identical values), so a run fits a 56 GB node.
  --adamw-hidden-wd: AdamW control with Muon's weight-decay placement (hidden matrices only).
  log.json also records the wall-clock of the whole run (wall_s) and the optimiser parameter split.
  Imports (greens.py, firstpass.py) come read-only from my_experiments/c1 and c1/nn.

train_v7r = RECONSTRUCTION of the lost train_v7.py (built 2026-10-05, overwritten 2026-10-06 by an unrelated file).

The original train_v7.py produced runs/v7_AB_band10_s1..s6 (6-seed holdout S 0.99561) and runs/prod_v3/psi_s1..8
(submission nn_prod_v3). It was defined as

    train_v6.py  (Group A physics inputs --px, Group B first-pass inputs --firstpass, everything else)
  + train_v5.py  LCFS band weighting (--w-band, --w-band-sigma; --w-inside was already in v6)
  + NaN/Inf guards: skip optimizer steps whose clipped gradient norm is not finite and count them per epoch
    (skipped_steps in log.json), assert finite weights before saving, assert finite predictions before writing
  + --test-cache DIR / --test-px DIR: after training, predict the public test set through the SAME input pipeline
    as training (Ridge features, Group A standardisation, Group B first pass, helicity, context) and write
    preds_test/<stem>.npz (psi float32 (T, 65, 65), q95, betaN, t).

This file was rebuilt on 2026-10-06 from train_v6.py (taken verbatim), train_v5.py's BandWeight (taken verbatim) and
what the surviving outputs of the original recorded: runs/v7_AB_band10_s1/log.json and runs/prod_v3/psi_s1/log.json
(argument names and order, per-epoch fields incl. skipped_steps, top-level fields), logs/nn-train7-*.out (printed
lines) and the model.pt key set (identical to v6's, so predict scripts written for v6/v7 checkpoints still work).
Things that could not be recovered and were chosen here: the exact form of the gradient guard (here: the
clip_grad_norm_ return value must be finite, otherwise the step is skipped and the LR schedule still advances),
a pre-filter that keeps frames with non-finite psi / coil inputs out of the batches (silent when there are none,
which is the case for the DIII-D caches), band weights precomputed for the training set only (v5 did both splits;
the held-out weights are never used), and freeing the training tensors before the test set is loaded.

    python train_v7r.py --head free --epochs 60 --lam-axis 1.0 --px $FEQ_CACHE/nn/px/diii_d_train --firstpass \
        --w-band 10 --w-band-sigma 1.5 --curves --init-seed 1 --out runs/v7_AB_band10_s1
    production (all shots, test predictions):
    python train_v7r.py ... --holdout 0 --ridge $FEQ_CACHE/nn/ridge_full.npz \
        --test-cache $FEQ_CACHE/nn/diii_d_public_test --test-px $FEQ_CACHE/nn/px/diii_d_public_test

Heads
  ridge     no network: writes the physics + Ridge prediction (pipeline check; must reproduce holdout.py)
  free      psi = ridge(x) + r_scale * MLP(z)                       (direct 65x65 residual)
  filament  psi = vacuum(I) + sum_k J_k(z) G_k + sum_v V_v(z) G_v + c(z)
Both network heads also predict (q95, betaN) from the shared trunk; --scalars-only trains only that head.

The held-out split is the one holdout.py uses (10 %, seed 0, over the sorted parquet stems).
Outputs in --out: model.pt (weights + config), log.json (per-epoch holdout metrics),
preds/<stem>.npz (psi float32 (T,65,65), q95, betaN) for every held-out shot,
preds_test/<stem>.npz (psi, q95, betaN, t) for every shot of --test-cache.
"""
# ml/ch1q copy of ml/ch1/train_c1.py (2026-10-13) + --train-list / --pred-list for out-of-fold psi maps.
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[1] / "c1"
sys.path.insert(0, str(C1))                                    # my_experiments/c1 (greens.py), read-only
sys.path.insert(0, str(C1 / "nn"))                             # firstpass.py, read-only
sys.path.insert(0, str(HERE))                                  # muon_opt.py
from greens import coil_filaments, filament_psi                  # noqa: E402
import muon_opt                                                  # noqa: E402
T_START = time.time()

ROOT = HERE.parents[2]
DATA_DIR = ROOT / "challenge_data" / "data" / "diii_d_train"
MASK = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
GRID_R, GRID_Z = MASK["grid_R"].astype(np.float64), MASK["grid_Z"].astype(np.float64)
ENVELOPE = MASK["mask_coarse"].astype(bool)
NPIX = ENVELOPE.size
N_BD = 64                       # LCFS points per frame kept for the boundary loss


# ----------------------------------------------------------------------------- split / loading
def split_stems(data_dir: Path, frac: float, seed: int, max_shots: int):
    files = sorted(data_dir.glob("*.parquet"))[: max_shots or None]
    stems = [f.stem for f in files]
    perm = np.random.RandomState(seed).permutation(len(stems))
    hold = set(perm[: int(round(frac * len(stems)))].tolist())
    return [s for i, s in enumerate(stems) if i not in hold], [s for i, s in enumerate(stems) if i in hold]


def load_shots(cache: Path, stems: list[str], with_psi: bool = True, workers: int = 16, px: Path | None = None):
    """Concatenate the per-shot cache (and the optional Group A cache) into flat arrays."""
    def one(s):
        d = np.load(cache / f"{s}.npz")
        T = len(d["ip"])
        out = {"X": d["X"], "I": d["I"], "ip": d["ip"], "t": d["t"], "bt": d["bt"]}
        if with_psi:
            out["psi"] = d["psi"].reshape(T, -1)
            out["scal"] = np.column_stack([d["q95"], d["betaN"]])
            lr, lz = d["lcfs_r"], d["lcfs_z"]
            n = np.isfinite(lr).sum(1)
            frac = np.linspace(0.0, 1.0, N_BD, endpoint=False)[None, :]
            idx = np.minimum((frac * n[:, None]).astype(int), np.maximum(n[:, None] - 1, 0))
            bd = np.stack([np.take_along_axis(lr, idx, 1), np.take_along_axis(lz, idx, 1)], -1)
            bd[n < 8] = np.nan
            out["bd"] = bd.astype(np.float32)
        if px is not None:
            q = np.load(px / f"{s}.npz")
            assert q["Xp"].shape[0] == T, (s, q["Xp"].shape, T)
            out["Xp"] = q["Xp"]; out["tste"] = q["te"]; out["tsne"] = q["ne"]
            out["tspos"] = np.broadcast_to(q["pos"], (T,) + q["pos"].shape)
        return out
    # (ml/muon) two passes: frame counts first, then every shot is written straight into preallocated arrays
    # (same values and dtypes as the concatenating version, about half the peak host memory)
    def count(s):
        with np.load(cache / f"{s}.npz") as d:
            return len(d["ip"])
    with ThreadPoolExecutor(workers) as ex:
        T = np.array(list(ex.map(count, stems)))
    off = np.concatenate([[0], np.cumsum(T)[:-1]])
    N = int(T.sum())
    first = one(stems[0])
    out = {k: np.empty((N,) + np.shape(v)[1:], np.float32) for k, v in first.items()}
    out["shot"] = np.empty(N, np.int32)

    def fill(i):
        sh = first if i == 0 else one(stems[i])
        o, n = off[i], T[i]
        assert len(sh["ip"]) == n, (stems[i], len(sh["ip"]), n)
        for k, v in sh.items():
            out[k][o:o + n] = v
        out["shot"][o:o + n] = i
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fill, range(len(stems))))
    del first
    return out, off, T


# ----------------------------------------------------------------------------- physics on the GPU
class Physics:
    """Vacuum field, E-coil / filament / offset terms and the Ridge residual, as in model.C1Model."""

    def __init__(self, basis: dict, ridge: dict, cols: list[str], device):
        G = torch.tensor(basis["G"].reshape(len(cols), -1), dtype=torch.float32, device=device)
        bcols = [str(c) for c in basis["cols"]]
        assert bcols == cols, (bcols, cols)
        self.is_f = torch.tensor([c != "magnetics_ECOILA" for c in cols], device=device)
        self.Gf_neg = -G[self.is_f]                                     # (18, P)
        self.GE = G[~self.is_f].sum(0)                                   # (P,)
        self.Gp = torch.tensor(basis["Gp"].reshape(-1), dtype=torch.float32, device=device)
        self.a_E, self.a_p, self.c = [float(v) for v in ridge["theta"]]
        self.mu = torch.tensor(ridge["mu"], dtype=torch.float32, device=device)
        sd = np.asarray(ridge["sd"], dtype=np.float64)
        self.inv_sd = torch.tensor(np.where(np.isfinite(sd), 1.0 / np.where(sd > 0, sd, 1.0), 0.0),
                                   dtype=torch.float32, device=device)
        self.z_clip = float(ridge["z_clip"])
        self.W = torch.tensor(ridge["W"], dtype=torch.float32, device=device)       # (F, P)
        self.b = torch.tensor(ridge["b"], dtype=torch.float32, device=device)
        self.Ws = torch.tensor(ridge["Ws"], dtype=torch.float32, device=device)     # (F, 2)
        self.bs = torch.tensor(ridge["bs"], dtype=torch.float32, device=device)

    def z(self, X):
        z = (X - self.mu) * self.inv_sd
        z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
        return z.clamp(-self.z_clip, self.z_clip)

    def vacuum(self, I):
        return I[:, self.is_f] @ self.Gf_neg

    def ecoil(self, I):
        return self.a_E * I[:, ~self.is_f].sum(1, keepdim=True) * self.GE

    def ridge_psi(self, z, I, ip):
        return (self.vacuum(I) + self.ecoil(I) + self.a_p * ip[:, None] * self.Gp + self.c
                + z @ self.W + self.b)

    def ridge_scal(self, z):
        return z @ self.Ws + self.bs


def build_cells(step: int = 2, erode: int = 1, n_sub: int = 4):
    """Plasma cells: every `step`-th grid point inside the envelope eroded by `erode` cells."""
    m = ENVELOPE.copy()
    for _ in range(erode):
        e = m.copy()
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    iz, ir = np.nonzero(m)
    keep = (iz % step == 0) & (ir % step == 0)
    return GRID_R[ir[keep]], GRID_Z[iz[keep]]


def build_vessel(n_max: int = 64):
    """Ring of cells just outside the envelope (vessel / eddy currents), subsampled to <= n_max."""
    m = ENVELOPE
    d = m.copy()
    d[1:] |= m[:-1]; d[:-1] |= m[1:]; d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    iz, ir = np.nonzero(d & ~m)
    ang = np.arctan2(GRID_Z[iz] - GRID_Z[iz].mean(), GRID_R[ir] - GRID_R[ir].mean())
    order = np.argsort(ang)
    pick = order[np.linspace(0, len(order) - 1, min(n_max, len(order))).round().astype(int)]
    return GRID_R[ir[pick]], GRID_Z[iz[pick]]


def cell_greens(cr, cz, w, h, n_sub):
    RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
    out = np.zeros((len(cr), NPIX))
    for k in range(len(cr)):
        fr, fz = coil_filaments(cr[k], cz[k], w, h, 0.0, 0.0, n_sub)
        out[k] = filament_psi(fr[:, None, None], fz[:, None, None], RR[None], ZZ[None]).mean(0).ravel()
    return out


# ----------------------------------------------------------------------------- scorer-style axis
def _erode_np(mask, cells):
    m = mask.copy()
    for _ in range(cells):
        e = m.copy()
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m


_AX_OK = _erode_np(ENVELOPE, 2)
_AX_OK[0, :] = _AX_OK[-1, :] = _AX_OK[:, 0] = _AX_OK[:, -1] = False
DR, DZ = float(GRID_R[1] - GRID_R[0]), float(GRID_Z[1] - GRID_Z[0])


class Axis:
    """o_point.find_o_point + parabolic refinement, vectorised in torch (DIII-D: axis = max of -psi)."""

    def __init__(self, device):
        self.ok = torch.tensor(_AX_OK, device=device)
        self.mask = torch.tensor(ENVELOPE, device=device)
        self.R = torch.tensor(GRID_R, dtype=torch.float32, device=device)
        self.Z = torch.tensor(GRID_Z, dtype=torch.float32, device=device)

    @torch.no_grad()
    def cells(self, psi_flat, chunk=16384):
        """Integer peak cell (iz, ir) of every frame, exactly as the scorer picks it."""
        N = psi_flat.shape[0]
        iz = torch.empty(N, dtype=torch.long, device=psi_flat.device)
        ir = torch.empty_like(iz)
        for i in range(0, N, chunk):
            phi = -psi_flat[i:i + chunk].view(-1, 1, 65, 65)
            mp = F.max_pool2d(phi, 3, stride=1, padding=1)
            lm = (phi >= mp) & self.ok
            cand = torch.where(lm, phi, torch.full_like(phi, -float("inf"))).view(len(phi), -1)
            idx = cand.argmax(1)
            none = torch.isinf(cand.max(1).values)
            if none.any():
                g = torch.where(self.mask, phi[none], torch.full_like(phi[none], -float("inf"))).view(int(none.sum()), -1)
                idx[none] = g.argmax(1)
            iz[i:i + chunk], ir[i:i + chunk] = idx // 65, idx % 65
        return iz, ir

    def _gather(self, psi_flat, iz, ir):
        phi = -psi_flat.view(-1, 65, 65)
        ar = torch.arange(len(phi), device=phi.device)
        izc, irc = iz.clamp(1, 63), ir.clamp(1, 63)
        b = phi[ar, izc, irc]
        return (phi[ar, izc - 1, irc], b, phi[ar, izc + 1, irc], phi[ar, izc, irc - 1], phi[ar, izc, irc + 1])

    def refine(self, psi_flat, iz, ir):
        """Scorer's sub-pixel position (R, Z) given the peak cell; exact formula (non-smooth)."""
        az, b, cz, ar_, cr = self._gather(psi_flat, iz, ir)
        dz_den, dr_den = az - 2 * b + cz, ar_ - 2 * b + cr
        dz = torch.where(dz_den == 0, torch.zeros_like(b), (0.5 * (az - cz) / dz_den).clamp(-1, 1))
        dr = torch.where(dr_den == 0, torch.zeros_like(b), (0.5 * (ar_ - cr) / dr_den).clamp(-1, 1))
        return self.R[ir] + dr * DR, self.Z[iz] + dz * DZ

    def offsets_smooth(self, psi_flat, iz, ir):
        """Differentiable sub-pixel offsets (cells) at a given peak cell: tanh-bounded parabola."""
        az, b, cz, ar_, cr = self._gather(psi_flat, iz, ir)
        dz = torch.tanh(0.5 * (az - cz) / (az - 2 * b + cz).clamp(max=-1e-6))
        dr = torch.tanh(0.5 * (ar_ - cr) / (ar_ - 2 * b + cr).clamp(max=-1e-6))
        return dr, dz

    @torch.no_grad()
    def locate(self, psi_flat):
        iz, ir = self.cells(psi_flat)
        R, Z = self.refine(psi_flat, iz, ir)
        return iz, ir, R, Z


# ----------------------------------------------------------------------------- boundary & gradient
class Boundary:
    """First-order LCFS displacement at the true boundary points: (dpsi(p) - dpsi(X)) / |grad psi_gt(p)|."""

    def __init__(self, device):
        self.R0, self.R1 = float(GRID_R[0]), float(GRID_R[-1])
        self.Z0, self.Z1 = float(GRID_Z[0]), float(GRID_Z[-1])
        self.device = device

    def _grid(self, bd):
        """(B, N, 2) physical (R, Z) -> grid_sample coordinates (B, 1, N, 2) in [-1, 1]."""
        x = (bd[..., 0] - self.R0) / (self.R1 - self.R0) * 2 - 1
        y = (bd[..., 1] - self.Z0) / (self.Z1 - self.Z0) * 2 - 1
        return torch.stack([x, y], -1)[:, None]

    def sample(self, maps, bd):
        """Bilinear sample of (B, 65, 65) maps at the boundary points -> (B, N)."""
        g = self._grid(torch.nan_to_num(bd, nan=0.0))
        return F.grid_sample(maps[:, None], g, mode="bilinear", align_corners=True)[:, 0, 0]

    @staticmethod
    def grad(psi_flat):
        p = psi_flat.view(-1, 65, 65)
        gZ, gR = torch.gradient(p, spacing=(DZ, DR), dim=(1, 2))
        return gR, gZ

    @torch.no_grad()
    def prepare(self, psi_flat, bd, chunk=16384):
        """Per frame: weights 1/|grad psi_gt| at the points (clamped), the X-point index, validity."""
        N = psi_flat.shape[0]
        w = torch.zeros((N, bd.shape[1]), device=psi_flat.device)
        ix = torch.zeros(N, dtype=torch.long, device=psi_flat.device)
        ok = torch.isfinite(bd).all(-1).all(-1)
        for i in range(0, N, chunk):
            sl = slice(i, i + chunk)
            gR, gZ = self.grad(psi_flat[sl])
            gr = self.sample(gR, bd[sl]); gz = self.sample(gZ, bd[sl])
            g = torch.sqrt(gr ** 2 + gz ** 2)
            ix[sl] = g.argmin(1)                                       # X-point = weakest |grad| on the LCFS
            med = g.median(1, keepdim=True).values
            w[sl] = 1.0 / g.clamp(min=0.25 * med)
        w[~ok] = 0.0
        return w, ix, ok

    def loss(self, psi_pred, psi_gt, bd, w, ix, ok, scale=0.01):
        """Mean squared normal displacement (m^2) / scale^2 over valid frames."""
        d = self.sample((psi_pred - psi_gt).view(-1, 65, 65), bd)      # (B, N) flux error on the LCFS
        dx = d.gather(1, ix[:, None])
        disp = (d - dx) * w
        per = (disp ** 2).mean(1)
        return (per * ok).sum() / ok.sum().clamp(min=1) / scale ** 2


def grad_loss(psi_pred, psi_gt, inside, g_scale, w=None):
    """Gradient matching inside the envelope (li depends on |grad psi|^2 inside the plasma).
    w: optional (B, P) or (P,) pixel weights (mean ~1 over the envelope); None = uniform."""
    gR_p, gZ_p = Boundary.grad(psi_pred)
    gR_t, gZ_t = Boundary.grad(psi_gt)
    e = ((gR_p - gR_t) ** 2 + (gZ_p - gZ_t) ** 2).view(len(psi_pred), -1)
    if w is None:
        return (e[:, inside]).mean() / g_scale ** 2
    w = w if w.dim() == 2 else w[None]
    return (e * w)[:, inside].mean() / g_scale ** 2


_RINV = torch.tensor(1.0 / GRID_R, dtype=torch.float32)


def gs_op(psi_flat):
    """Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 on the interior grid points -> (B, 63, 63)."""
    p = psi_flat.view(-1, 65, 65)
    c = p[:, 1:-1, 1:-1]
    d2r = (p[:, 1:-1, 2:] - 2 * c + p[:, 1:-1, :-2]) / DR ** 2
    d1r = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2 * DR)
    d2z = (p[:, 2:, 1:-1] - 2 * c + p[:, :-2, 1:-1]) / DZ ** 2
    return d2r - _RINV[1:-1].to(p.device)[None, None, :] * d1r + d2z


def lap_loss(psi_pred, psi_gt, inside, l_scale, w=None):
    """Supervised Delta* matching (plasma current density) on interior envelope points, optional pixel weights."""
    e = (gs_op(psi_pred) - gs_op(psi_gt)) ** 2                       # (B, 63, 63)
    m = inside.view(65, 65)[1:-1, 1:-1].reshape(-1)
    e = e.reshape(len(psi_pred), -1)
    if w is not None:
        w = w if w.dim() == 2 else w[None]
        e = e * w.view(-1, 65, 65)[:, 1:-1, 1:-1].reshape(w.shape[0], -1)
    return e[:, m].mean() / l_scale ** 2


_RR_FLAT = np.meshgrid(GRID_R, GRID_Z)[0].ravel()
_ZZ_FLAT = np.meshgrid(GRID_R, GRID_Z)[1].ravel()


def lcfs_inside(bd, ok, dilate=1):
    """(B, 64, 2) LCFS points (nan-free), ok (B,) -> (B, P) float mask of pixels inside the polygon (even-odd ray cast,
    the scorer's rule), dilated by `dilate` cells; frames without a usable LCFS get an all-zero mask."""
    dev = bd.device
    rq = torch.tensor(_RR_FLAT, dtype=torch.float32, device=dev)[None]
    zq = torch.tensor(_ZZ_FLAT, dtype=torch.float32, device=dev)[None]
    r1, z1 = bd[..., 0], bd[..., 1]
    r2, z2 = torch.roll(r1, -1, 1), torch.roll(z1, -1, 1)
    ins = torch.zeros((bd.shape[0], NPIX), dtype=torch.bool, device=dev)
    for e in range(bd.shape[1]):
        za, zb, ra, rb = z1[:, e:e + 1], z2[:, e:e + 1], r1[:, e:e + 1], r2[:, e:e + 1]
        cross = (za > zq) != (zb > zq)
        den = torch.where(zb == za, torch.ones_like(za), zb - za)
        xint = ra + (zq - za) * (rb - ra) / den
        ins ^= cross & (rq < xint) & (zb != za)
    m = ins.view(-1, 1, 65, 65).float()
    if dilate:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m.view(-1, NPIX) * ok[:, None].float()


class BandWeight:
    """(train_v5) Per-frame pixel weights 1 + w * exp(-d^2 / 2 sigma^2), d = distance (cells) to the nearest true-LCFS point."""

    def __init__(self, device, w: float, sigma_cells: float):
        self.w, self.s = w, sigma_cells
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        self.RR = torch.tensor(RR.ravel() / DR, dtype=torch.float32, device=device)
        self.ZZ = torch.tensor(ZZ.ravel() / DZ, dtype=torch.float32, device=device)

    def weights(self, bd, ok):
        """bd (B, 64, 2) physical LCFS points (nan-free), ok (B,) -> (B, P) weights, mean 1 per frame."""
        br, bz = bd[..., 0] / DR, bd[..., 1] / DZ                                  # (B, 64)
        d2 = ((self.RR[None, :, None] - br[:, None, :]) ** 2 + (self.ZZ[None, :, None] - bz[:, None, :]) ** 2).amin(2)
        w = 1.0 + self.w * torch.exp(-0.5 * d2 / self.s ** 2) * ok[:, None].float()
        return w / w.mean(1, keepdim=True)


# ----------------------------------------------------------------------------- network
class Block(nn.Module):
    def __init__(self, d, mult=2, drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.fc2(F.silu(self.fc1(self.ln(x)))))


class Net(nn.Module):
    def __init__(self, n_in, width, depth, n_out, drop=0.0, width_scal=256):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[Block(width, drop=drop) for _ in range(depth)])
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_out)
        self.scal = nn.Sequential(nn.Linear(width, width_scal), nn.SiLU(), nn.Linear(width_scal, 2))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, z):
        h = self.ln(self.blocks(self.inp(z)))
        return self.head(h), self.scal(h)


class Model:
    """Head-specific assembly of psi from network outputs."""

    def __init__(self, head, phys: Physics, n_in, width, depth, drop, device, r_scale, width_scal=256):
        self.head, self.phys, self.device, self.r_scale = head, phys, device, r_scale
        if head == "free":
            n_out = NPIX
        elif head == "scalar":
            n_out = 1                                   # unused dummy head
        elif head == "filament":
            cr, cz = build_cells()
            vr, vz = build_vessel()
            dR, dZ = float(GRID_R[1] - GRID_R[0]), float(GRID_Z[1] - GRID_Z[0])
            self.Gfil = torch.tensor(cell_greens(cr, cz, 2 * dR, 2 * dZ, 4), dtype=torch.float32, device=device)
            self.Gves = torch.tensor(cell_greens(vr, vz, dR, dZ, 2), dtype=torch.float32, device=device)
            self.K, self.V = len(cr), len(vr)
            n_out = self.K + self.V + 2
            print(f"  filament head: {self.K} plasma cells, {self.V} vessel cells", flush=True)
        else:
            raise ValueError(head)
        self.net = Net(n_in, width, depth, n_out, drop, width_scal).to(device)

    def psi(self, out, z, I, ip):
        if self.head == "scalar":
            return self.phys.ridge_psi(z, I, ip)
        if self.head == "free":
            return self.phys.ridge_psi(z, I, ip) + self.r_scale * out
        u, v, s, c = out[:, :self.K], out[:, self.K:self.K + self.V], out[:, -2], out[:, -1]
        J = -ip[:, None] * torch.exp(s - 0.05)[:, None] * torch.softmax(u, 1)          # A per cell
        return (self.phys.vacuum(I) + self.phys.ecoil(I) + J @ self.Gfil + (1e4 * v) @ self.Gves
                + self.phys.c + 0.01 * c[:, None])

    def forward(self, z_in, z, I, ip):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, scal = self.net(z_in)
        return self.psi(out.float(), z, I, ip), scal.float()


def param_role(name: str, p) -> str:
    """muon_opt roles of the Net parameters: inp = first layer, head = psi head and the last scalar layer."""
    if p.ndim != 2:
        return "other"
    if name.startswith("inp."):
        return "inp"
    if name.startswith("head.") or name == "scal.2.weight":
        return "head"
    return "hidden"                     # blocks.*.fc1 / fc2, scal.0


# ----------------------------------------------------------------------------- NaN/Inf guards
def check_finite_weights(net: nn.Module, what: str):
    """Refuse to write a checkpoint whose parameters are not all finite."""
    bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"non-finite weights in {bad[:5]}{' ...' if len(bad) > 5 else ''}; {what} not written")


def check_finite_preds(psi: np.ndarray, sc: np.ndarray, what: str):
    n_psi, n_sc = int((~np.isfinite(psi)).any(1).sum()), int((~np.isfinite(sc)).any(1).sum())
    if n_psi or n_sc:
        raise RuntimeError(f"non-finite {what}: psi in {n_psi} frames, scalars in {n_sc} frames; nothing written")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(f"{FEQ_CACHE}/nn/diii_d_train"))
    ap.add_argument("--ridge", type=Path, default=Path(f"{FEQ_CACHE}/nn/ridge_holdout.npz"))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR, help="parquet dir, for the split order")
    ap.add_argument("--head", choices=["ridge", "free", "filament", "scalar"], default="free")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lam-scal", type=float, default=0.2)
    ap.add_argument("--lam-axis", type=float, default=0.0,
                    help="weight of the differentiable axis loss (scorer's parabola at the true peak cell)")
    ap.add_argument("--w-inside", type=float, default=1.0, help="relative MSE weight of grid points inside the envelope")
    ap.add_argument("--lam-bd", type=float, default=0.0, help="weight of the LCFS displacement loss")
    ap.add_argument("--lam-grad", type=float, default=0.0, help="weight of the gradient-matching loss inside the envelope")
    ap.add_argument("--test-cache", type=Path, default=None,
                    help="after training, predict every shot of this cache (no targets needed) -> <out>/preds_test")
    ap.add_argument("--test-px", type=Path, default=None, help="Group A cache of the --test-cache split (needed with --px)")
    ap.add_argument("--w-band", type=float, default=0.0, help="extra pixel weight in a band around the true LCFS")
    ap.add_argument("--w-band-sigma", type=float, default=1.5, help="cells")
    ap.add_argument("--px", type=Path, default=None, help="Group A physics-feature cache dir (physics_plus.py)")
    ap.add_argument("--firstpass", action="store_true", help="add Group B first-pass equilibrium inputs (needs --px)")
    ap.add_argument("--save-epochs", default="", help="comma-separated epochs at which to save ckpt_epNNN.pt")
    ap.add_argument("--curves", action="store_true",
                    help="every epoch, log RMSE and R2 of psi, q95 and betaN on a fixed training sample and on the holdout")
    ap.add_argument("--scalars-only", action="store_true",
                    help="train only the (q95, betaN) head; psi predictions are the Ridge's (combine later)")
    ap.add_argument("--ctx", action="store_true",
                    help="shot-context inputs: per-shot mean of the standardized features and a 3-frame average")
    ap.add_argument("--helicity", action="store_true",
                    help="add sign(B_T)*sign(Ip) and the signed B_T as inputs (bt_sign.py output)")
    ap.add_argument("--bt-sign", type=Path, default=Path(f"{FEQ_CACHE}/nn/bt_sign_diii_d_train.json"))
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0, help="held-out split seed (keep 0 for comparability)")
    ap.add_argument("--init-seed", type=int, default=None, help="torch seed for init/shuffling (default: --seed)")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--width-scal", type=int, default=256, help="(ml/muon) width of the scalar head")
    ap.add_argument("--scal-range", default="", help="(ch1) Q_LO,Q_HI,B_LO,B_HI physical range of q95 / betaN training labels")
    ap.add_argument("--scal-w", default="1,1", help="(ch1) loss multipliers of q95, betaN")
    ap.add_argument("--w-axis", type=float, default=0.0, help="(ch1) extra pixel weight around the true axis")
    ap.add_argument("--w-axis-sigma", type=float, default=2.0, help="(ch1) cells")
    ap.add_argument("--w-xpt", type=float, default=0.0, help="(ch1) extra pixel weight around the true X-point (weakest |grad psi| on the LCFS)")
    ap.add_argument("--w-xpt-sigma", type=float, default=2.0, help="(ch1) cells")
    ap.add_argument("--grad-w", choices=["env", "band", "lcfs"], default="env", help="(ch1) pixel weights of --lam-grad")
    ap.add_argument("--lam-grad2", type=float, default=0.0, help="(ch1) weight of a second gradient-matching term")
    ap.add_argument("--grad-w2", choices=["env", "band", "lcfs"], default="lcfs", help="(ch1) pixel weights of --lam-grad2")
    ap.add_argument("--lam-lap", type=float, default=0.0, help="(ch1) weight of supervised Delta* psi matching")
    ap.add_argument("--lap-w", choices=["env", "band", "lcfs"], default="band", help="(ch1) pixel weights of --lam-lap")
    ap.add_argument("--ema", type=float, default=0.0, help="(ch1) EMA decay of the weights (0 = off)")
    ap.add_argument("--ema-start", type=float, default=0.5, help="(ch1) fraction of the steps after which the EMA starts")
    ap.add_argument("--in-noise", type=float, default=0.0, help="(ch1) input noise sd (standardised units), training only")
    ap.add_argument("--bag-frac", type=float, default=1.0, help="(ch1) fraction of training shots used")
    ap.add_argument("--bag-seed", type=int, default=0)
    ap.add_argument("--train-list", type=Path, default=None, help="(ch1q) text file of training stems (overrides the split)")
    ap.add_argument("--pred-list", type=Path, default=None, help="(ch1q) text file of stems to predict as the 'holdout' "
                    "(out-of-fold maps -> <out>/preds)")
    ap.add_argument("--adamw-hidden-wd", type=float, default=None,
                    help="(ml/muon control) --opt adamw: decoupled weight decay of the hidden 2D matrices (the ones Muon would "
                         "take); all other parameters keep --wd")
    muon_opt.add_args(ap)
    args = ap.parse_args()
    if args.test_cache is not None and args.px is not None and args.test_px is None:
        ap.error("--test-px is required with --test-cache when the model was trained with --px")
    torch.manual_seed(args.seed if args.init_seed is None else args.init_seed)
    if args.scalars_only:
        args.head = "scalar"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.cache / "meta.json").read_text())
    basis = dict(np.load(args.cache / "basis.npz"))
    ridge = dict(np.load(args.ridge))
    assert list(ridge["names"]) == meta["names"], "feature names differ between cache and ridge model"
    names = meta["names"]

    tr_stems, ho_stems = split_stems(args.data_dir, args.holdout, args.seed, args.max_shots)
    if args.train_list is not None:   # (ch1q) explicit fold lists: train on --train-list, predict --pred-list
        tr_stems = args.train_list.read_text().split()
        ho_stems = args.pred_list.read_text().split()
        assert not set(tr_stems) & set(ho_stems), "train and predict lists overlap"
        print(f"fold lists: {len(tr_stems)} training shots ({args.train_list.name}), {len(ho_stems)} predicted "
              f"({args.pred_list.name})", flush=True)
    if args.bag_frac < 1.0:
        rng = np.random.RandomState(1000 + args.bag_seed)
        keep = np.sort(rng.choice(len(tr_stems), int(round(args.bag_frac * len(tr_stems))), replace=False))
        tr_stems = [tr_stems[i] for i in keep]
        print(f"bagging: {len(tr_stems)} training shots (fraction {args.bag_frac}, bag seed {args.bag_seed})", flush=True)
    missing = [s for s in tr_stems + ho_stems if not (args.cache / f"{s}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shots missing from the cache, e.g. {missing[:3]}")
    t0 = time.time()
    tr, _, _ = load_shots(args.cache, tr_stems, px=args.px)
    final = len(ho_stems) == 0
    if final:                      # --holdout 0: evaluate on a small training sample instead (sanity only)
        ho_stems_eval = tr_stems[:: max(1, len(tr_stems) // 100)]
        ho, ho_off, ho_T = load_shots(args.cache, ho_stems_eval, px=args.px)
        print(f"FINAL fit: {len(tr_stems)} shots ({len(tr['ip']):,} frames); 'holdout' metrics below are on "
              f"{len(ho_stems_eval)} TRAINING shots (in-sample)", flush=True)
    else:
        ho, ho_off, ho_T = load_shots(args.cache, ho_stems, px=args.px)
    print(f"loaded {len(tr_stems)} train shots ({len(tr['ip']):,} frames) and {len(ho_stems)} held-out "
          f"({len(ho['ip']):,} frames) in {time.time() - t0:.0f}s", flush=True)
    ho_used = ho_stems_eval if final else ho_stems

    def add_helicity(d, stems, bt_path: Path):
        """sign(B_T) * sign(Ip) per frame (numpy dict, before to_dev)."""
        bt = json.loads(bt_path.read_text())
        h = np.array([bt[s]["bt_sign"] * bt[s]["ip_sign"] for s in stems], np.float32)
        d["hel"] = h[d["shot"]]

    if args.helicity:
        add_helicity(tr, tr_stems, args.bt_sign)
        add_helicity(ho, ho_used, args.bt_sign)
        print(f"helicity inputs on: train frames with B_T x Ip > 0: {(tr['hel'] > 0).mean():.3f}", flush=True)

    phys = Physics(basis, ridge, meta["cols"], dev)

    def add_context(d):
        """Per-shot mean of z and the mean of z over the previous/current/next frame (same shot)."""
        z = phys.z(d["X"])
        shot = d["shot"].long()
        n_shot = int(shot.max()) + 1
        cnt = torch.zeros(n_shot, device=dev).index_add_(0, shot, torch.ones(len(shot), device=dev))
        mean = torch.zeros(n_shot, z.shape[1], device=dev).index_add_(0, shot, z) / cnt[:, None]
        i = torch.arange(len(shot), device=dev)
        prev = torch.where((i > 0) & (shot == shot[(i - 1).clamp(min=0)]), (i - 1).clamp(min=0), i)
        nxt = torch.where((i < len(shot) - 1) & (shot == shot[(i + 1).clamp(max=len(shot) - 1)]),
                          (i + 1).clamp(max=len(shot) - 1), i)
        d["ctx"] = torch.cat([mean[shot], (z[prev] + z + z[nxt]) / 3], 1)

    def to_dev(d):
        return {k: torch.from_numpy(v).to(dev) for k, v in d.items()}
    tr, ho = to_dev(tr), to_dev(ho)

    # ---- Group A (--px) and Group B (--firstpass) inputs: statistics from the training frames, then the same
    #      transformation for the held-out set and, after training, for --test-cache
    extra_stats = {}
    fpm = None
    ibz = names.index("bz_over_bip") if args.firstpass else None

    def fit_stats(F0):
        fin = torch.isfinite(F0)
        cnt = fin.sum(0).clamp(min=1)
        mu = torch.where(fin, F0, torch.zeros_like(F0)).sum(0) / cnt
        sd = (torch.where(fin, (F0 - mu) ** 2, torch.zeros_like(F0)).sum(0) / cnt).sqrt()
        sd = torch.where(sd > 1e-9, sd, torch.full_like(sd, float("inf")))
        return mu, sd, fin

    def standardize(F0, mu, sd):
        return torch.nan_to_num((F0 - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)

    def first_pass_raw(d):
        """Group B features of one split from the Ridge map (chunked); NaN where undefined."""
        outs = []
        for i in range(0, len(d["ip"]), 8192):
            sl = slice(i, i + 8192)
            z = phys.z(d["X"][sl])
            P = phys.ridge_psi(z, d["I"][sl], d["ip"][sl])
            Pv = phys.vacuum(d["I"][sl]) + phys.ecoil(d["I"][sl])
            outs.append(fpm(P, d["ip"][sl], d["bt"][sl], d["X"][sl][:, ibz], d["tspos"][sl], d["tste"][sl], d["tsne"][sl], Pv))
        return torch.cat(outs)

    def add_extra_inputs(d):
        """Group A / Group B / context columns of a non-training split, with the training statistics."""
        if args.px is not None:
            d["zp"] = standardize(d.pop("Xp"), *extra_stats["px"])
        if args.firstpass:
            d["zf"] = standardize(first_pass_raw(d), *extra_stats["fp"])
        for k in ("tspos", "tste", "tsne"):
            d.pop(k, None)
        if args.ctx:
            add_context(d)

    if args.px is not None:
        mu_p, sd_p, _ = fit_stats(tr["Xp"])
        extra_stats["px"] = (mu_p, sd_p)
        tr["zp"] = standardize(tr.pop("Xp"), mu_p, sd_p)
        px_names = json.loads((args.px / "meta.json").read_text())["names"]
        assert len(px_names) == tr["zp"].shape[1], (len(px_names), tr["zp"].shape)
        print(f"Group A physics inputs: {len(px_names)}", flush=True)
    if args.firstpass:
        assert args.px is not None, "--firstpass needs --px for the Thomson channels"
        from firstpass import FirstPass
        fpm = FirstPass(GRID_R, GRID_Z, ENVELOPE, dev)
        F0 = first_pass_raw(tr)
        mu_f, sd_f, fin = fit_stats(F0)
        extra_stats["fp"] = (mu_f, sd_f)
        tr["zf"] = standardize(F0, mu_f, sd_f)
        print(f"Group B first-pass inputs: {len(fpm.names)}; NaN share on train "
              f"{float((~fin).float().mean()):.3f}", flush=True)
        del F0, fin
    for k in ("tspos", "tste", "tsne"):
        tr.pop(k, None)
    if args.ctx:
        add_context(tr)
        print(f"context inputs on: +{tr['ctx'].shape[1]} columns", flush=True)
    add_extra_inputs(ho)

    axis = Axis(dev)
    for d in (tr, ho):
        d["ax_iz"], d["ax_ir"], d["ax_R"], d["ax_Z"] = axis.locate(d["psi"])
        d["ax_dr"], d["ax_dz"] = axis.offsets_smooth(d["psi"], d["ax_iz"], d["ax_ir"])
    print(f"true axis (scorer functional): Z sd {float(ho['ax_Z'].std()):.4f} m, R sd {float(ho['ax_R'].std()):.4f} m", flush=True)
    w_pix = torch.ones(NPIX, device=dev)
    w_pix[torch.from_numpy(ENVELOPE.ravel()).to(dev)] = args.w_inside
    w_pix = w_pix / w_pix.mean()
    inside = torch.from_numpy(ENVELOPE.ravel()).to(dev)
    bdry = Boundary(dev)
    for d in (tr, ho):
        d["bd_w"], d["bd_ix"], d["bd_ok"] = bdry.prepare(d["psi"], d["bd"])
    print(f"LCFS loss points: {N_BD}/frame, frames without usable LCFS: train {int((~tr['bd_ok']).sum())}, "
          f"held-out {int((~ho['bd_ok']).sum())}", flush=True)

    # LCFS band weighting (train_v5): per-frame pixel weights, precomputed in float16 for the training frames
    bandw = BandWeight(dev, args.w_band, args.w_band_sigma) if args.w_band > 0 else None
    if bandw is not None:
        t0 = time.time()
        bd0 = torch.nan_to_num(tr["bd"], nan=0.0)
        wb = torch.empty((len(tr["ip"]), NPIX), dtype=torch.float16, device=dev)
        for i in range(0, len(tr["ip"]), 4096):
            sl = slice(i, i + 4096)
            wb[sl] = bandw.weights(bd0[sl], tr["bd_ok"][sl]).half()
        tr["w_band"] = wb                                              # (N, P) weights, mean 1 per frame
        del bd0, wb
        print(f"LCFS band weighting: +{args.w_band} within sigma {args.w_band_sigma} cells of the true LCFS "
              f"(precomputed in {time.time() - t0:.0f}s)", flush=True)

    if ((args.lam_grad > 0 and args.grad_w == "lcfs") or (args.lam_lap > 0 and args.lap_w == "lcfs")
            or (args.lam_grad2 > 0 and args.grad_w2 == "lcfs")):   # (ch1) inside-LCFS masks of the training frames, precomputed (uint8)
        t0 = time.time()
        bd0 = torch.nan_to_num(tr["bd"], nan=0.0)
        li_ = torch.empty((len(tr["ip"]), NPIX), dtype=torch.uint8, device=dev)
        for i in range(0, len(tr["ip"]), 4096):
            sl = slice(i, i + 4096)
            li_[sl] = lcfs_inside(bd0[sl], tr["bd_ok"][sl]).to(torch.uint8)
        tr["lcfs_in"] = li_
        del bd0, li_
        print(f"inside-LCFS masks precomputed in {time.time() - t0:.0f}s", flush=True)
    RRc = torch.tensor(_RR_FLAT / DR, dtype=torch.float32, device=dev)          # (ch1) pixel coordinates in cells
    ZZc = torch.tensor(_ZZ_FLAT / DZ, dtype=torch.float32, device=dev)
    r_over = torch.tensor(float(np.mean(GRID_R)) / _RR_FLAT, dtype=torch.float32, device=dev)[None]   # R0 / R

    # NaN/Inf guard on the training pool: frames with non-finite psi targets or coil inputs never enter a batch
    # (package) chunked over frames: the same per-frame result as torch.isfinite(tr["psi"]).all(1) without its full-size
    # temporaries (a float copy + masks of the whole training set, ~40 GB at 7,041 shots), so training fits an 80 GB GPU
    pool_ok = (torch.cat([torch.isfinite(tr["psi"][i:i + 65536]).all(1) for i in range(0, len(tr["psi"]), 65536)])
               & torch.isfinite(tr["I"]).all(1) & torch.isfinite(tr["ip"]))
    n_bad = int((~pool_ok).sum())
    if n_bad:
        print(f"NaN/Inf guard: {n_bad} training frames with non-finite psi / coil inputs are excluded from the batches",
              flush=True)
    pool = torch.nonzero(pool_ok)[:, 0]
    del pool_ok

    # network inputs: the Ridge's standardized/clipped features + two Thomson-availability flags
    core = torch.tensor([n.startswith(("coreTe", "coreNe")) for n in names], device=dev)
    tan = torch.tensor([n.startswith(("tanTe", "tanNe")) for n in names], device=dev)

    i_bcoil = names.index("bcoil")

    def inputs(d, sl):
        X = d["X"][sl]
        z = phys.z(X)
        fin = torch.isfinite(X)
        parts = [z, torch.stack([(~fin[:, core]).all(1), (~fin[:, tan]).all(1)], 1).float()]
        if args.helicity:
            h = d["hel"][sl][:, None]
            parts += [h, z[:, i_bcoil:i_bcoil + 1] * h]
        if args.ctx:
            parts.append(d["ctx"][sl])
        if "zp" in d:
            parts.append(d["zp"][sl])
        if "zf" in d:
            parts.append(d["zf"][sl])
        return torch.cat(parts, 1), z

    # (ch1) physical-range filter of the scalar labels: frames outside the range get NaN (no scalar loss)
    if args.scal_range:
        qlo, qhi, blo, bhi = [float(v) for v in args.scal_range.split(",")]
        bad = ~((tr["scal"][:, 0] >= qlo) & (tr["scal"][:, 0] <= qhi) & (tr["scal"][:, 1] >= blo) & (tr["scal"][:, 1] <= bhi))
        bad &= torch.isfinite(tr["scal"]).all(1)
        tr["scal"][bad] = float("nan")
        print(f"scalar label range filter q95 [{qlo}, {qhi}], betaN [{blo}, {bhi}]: {int(bad.sum())} training frames "
              f"without scalar loss", flush=True)
    scal_w = torch.tensor([float(v) for v in args.scal_w.split(",")], device=dev)
    # scalar standardisation (train frames with labels)
    ok = torch.isfinite(tr["scal"]).all(1)
    s_mu, s_sd = tr["scal"][ok].mean(0), tr["scal"][ok].std(0)

    # residual scale from a training sample
    with torch.no_grad():
        idx = torch.randperm(len(tr["ip"]), device=dev)[:65536]
        zin, z = inputs(tr, idx)
        r = tr["psi"][idx] - phys.ridge_psi(z, tr["I"][idx], tr["ip"][idx])
        r_scale = float(r.std())
        psi_sd = float(tr["psi"][idx].std())
        gR, gZ = Boundary.grad(r)
        g_scale = float(torch.sqrt((gR ** 2 + gZ ** 2).view(len(r), -1)[:, inside].mean()))
        l_scale = float(torch.sqrt((gs_op(r) ** 2).reshape(len(r), -1)[:, inside.view(65, 65)[1:-1, 1:-1].reshape(-1)].mean()))
    print(f"psi sd {psi_sd:.4f}, ridge residual sd {r_scale:.5f} (R2 ~ {1 - (r_scale / psi_sd) ** 2:.5f})", flush=True)

    @torch.no_grad()
    def predict(model, d, chunk=8192):
        psi_out = torch.empty_like(d["psi"]) if "psi" in d else torch.empty((len(d["ip"]), NPIX), device=dev)
        sc_out = torch.empty((len(d["ip"]), 2), device=dev)
        for i in range(0, len(d["ip"]), chunk):
            sl = slice(i, i + chunk)
            zin, z = inputs(d, sl)
            if model is None:
                psi_out[sl] = phys.ridge_psi(z, d["I"][sl], d["ip"][sl])
                sc_out[sl] = phys.ridge_scal(z)
            else:
                p, s = model.forward(zin, z, d["I"][sl], d["ip"][sl])
                psi_out[sl] = p
                sc_out[sl] = s * s_sd + s_mu
        return psi_out, sc_out

    def evaluate(model):
        psi_p, sc_p = predict(model, ho)
        g = ho["psi"]
        r2_psi = 1 - ((psi_p - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()
        r2s = []
        for j in range(2):
            m = torch.isfinite(ho["scal"][:, j])
            y, p = ho["scal"][m, j], sc_p[m, j]
            r2s.append(float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        inside = torch.from_numpy(ENVELOPE.ravel()).to(dev)
        rmse_in = float(((psi_p - g)[:, inside] ** 2).mean().sqrt())
        bd_mm = []
        for i in range(0, len(psi_p), 16384):
            sl = slice(i, i + 16384)
            bd_mm.append(float(bdry.loss(psi_p[sl], g[sl], ho["bd"][sl], ho["bd_w"][sl], ho["bd_ix"][sl],
                                         ho["bd_ok"][sl], scale=1.0)) * int(ho["bd_ok"][sl].sum()))
        bd_rms_mm = 1000.0 * math.sqrt(sum(bd_mm) / max(1, int(ho["bd_ok"].sum())))
        _, _, Rp, Zp = axis.locate(psi_p)
        r2_R = float(1 - ((Rp - ho["ax_R"]) ** 2).sum() / ((ho["ax_R"] - ho["ax_R"].mean()) ** 2).sum())
        r2_Z = float(1 - ((Zp - ho["ax_Z"]) ** 2).sum() / ((ho["ax_Z"] - ho["ax_Z"].mean()) ** 2).sum())
        return {"r2_psi": float(r2_psi), "r2_q95": r2s[0], "r2_betaN": r2s[1], "rmse_psi_inside": rmse_in,
                "r2_Raxis": r2_R, "r2_Zaxis": r2_Z, "lcfs_disp_rms_mm": bd_rms_mm}

    def target_curves(model, d):
        psi_p, sc_p = predict(model, d)
        out = {}
        g = d["psi"]
        e = psi_p - g
        out["psi_rmse"] = float((e ** 2).mean().sqrt())
        out["psi_r2"] = float(1 - (e ** 2).sum() / ((g - g.mean()) ** 2).sum())
        for j, nm in enumerate(("q95", "betaN")):
            m = torch.isfinite(d["scal"][:, j])
            y, pj = d["scal"][m, j], sc_p[m, j]
            out[f"{nm}_rmse"] = float(((pj - y) ** 2).mean().sqrt())
            out[f"{nm}_r2"] = float(1 - ((pj - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        return out

    tr_sample = None
    if args.curves:
        gsel = torch.Generator(device=dev).manual_seed(123)
        idx_s = torch.randperm(len(tr["ip"]), generator=gsel, device=dev)[: len(ho["ip"])]
        tr_sample = {k: v[idx_s] for k, v in tr.items()
                     if torch.is_tensor(v) and v.shape[:1] == tr["ip"].shape[:1] and k not in ("w_band", "lcfs_in")}
        print(f"curves on: training sample of {len(idx_s):,} frames vs {len(ho['ip']):,} held-out frames", flush=True)

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "n_train": len(tr_stems), "n_holdout": len(ho_stems),
           "frames_train": int(len(tr["ip"])), "frames_holdout": int(len(ho["ip"])), "r_scale": r_scale, "epochs": []}
    base = evaluate(None)
    print(f"ridge baseline on held-out: {base}", flush=True)
    log["ridge_baseline"] = base

    def checkpoint(model, n_in, extra):
        return {"state": model.net.state_dict(), "head": args.head, "width": args.width, "depth": args.depth,
                "drop": args.drop, "n_in": n_in, "r_scale": r_scale, "s_mu": s_mu.cpu(), "s_sd": s_sd.cpu(),
                "ridge": str(args.ridge), "names": names, "helicity": args.helicity, "ctx": args.ctx,
                "width_scal": args.width_scal, **extra}

    model = None
    if args.head != "ridge":
        n_in = (len(names) + 2 + (2 if args.helicity else 0) + (2 * len(names) if args.ctx else 0)
                + (tr["zp"].shape[1] if "zp" in tr else 0) + (tr["zf"].shape[1] if "zf" in tr else 0))
        model = Model(args.head, phys, n_in, args.width, args.depth, args.drop, dev, r_scale, args.width_scal)
        n_par = sum(p.numel() for p in model.net.parameters())
        print(f"network: width {args.width} depth {args.depth} -> {n_par / 1e6:.1f} M parameters", flush=True)
        if args.opt == "adamw" and args.adamw_hidden_wd is not None:
            hid = [p for n, p in model.net.named_parameters() if param_role(n, p) == "hidden"]
            rest = [p for n, p in model.net.named_parameters() if param_role(n, p) != "hidden"]
            opts = [torch.optim.AdamW([{"params": hid, "weight_decay": args.adamw_hidden_wd}, {"params": rest}],
                                      lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))]
            log["opt_split"] = {"adamw:hidden": sum(p.numel() for p in hid), "adamw:rest": sum(p.numel() for p in rest)}
            print(f"optimiser: AdamW, hidden matrices wd {args.adamw_hidden_wd:g}, rest wd {args.wd:g}", flush=True)
        elif args.opt == "adamw":
            opts = [torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))]
            log["opt_split"] = {"adamw:all": n_par}
        else:
            mu_p, ad_p, summ = muon_opt.split_params(model.net.named_parameters(), param_role, args)
            opts = [muon_opt.make_muon(mu_p, args)] + ([torch.optim.AdamW(ad_p, lr=args.lr, weight_decay=args.wd,
                                                                         betas=(0.9, 0.98))] if ad_p else [])
            log["opt_split"] = summ
            print(f"optimiser: Muon ({args.muon_adjust}, lr {args.muon_lr:g}, wd {args.muon_wd:g}, momentum {args.muon_mom}, "
                  f"nesterov {bool(args.muon_nesterov)}, ns {args.muon_ns_steps}) + AdamW (lr {args.lr:g}, wd {args.wd:g}); "
                  f"split: {muon_opt.describe(summ)}", flush=True)
        opt = muon_opt.Combo(opts)
        N = len(pool)
        steps_per_epoch = N // args.batch
        total = args.epochs * steps_per_epoch
        warm = max(1, min(1000, total // 20))
        lam = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
        scheds = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in opts]
        step = 0
        ema_p = None
        net_params = [p for p in model.net.parameters()]
        ema_from = int(args.ema_start * total)
        for ep in range(args.epochs):
            model.net.train()
            t0 = time.time()
            perm = pool[torch.randperm(N, device=dev)]
            tot_psi = tot_scal = tot_axis = tot_bd = tot_gr = 0.0
            skipped = 0
            for i in range(steps_per_epoch):
                idx = perm[i * args.batch:(i + 1) * args.batch]
                zin, z = inputs(tr, idx)
                if args.in_noise > 0:
                    zin = zin + args.in_noise * torch.randn_like(zin)
                psi_p, sc_p = model.forward(zin, z, tr["I"][idx], tr["ip"][idx])
                wp = w_pix if bandw is None else w_pix * tr["w_band"][idx].float()
                if args.w_axis > 0:
                    d2 = ((RRc[None] - tr["ax_R"][idx, None] / DR) ** 2 + (ZZc[None] - tr["ax_Z"][idx, None] / DZ) ** 2)
                    wa = 1.0 + args.w_axis * torch.exp(-0.5 * d2 / args.w_axis_sigma ** 2)
                    wp = wp * wa
                    wp = wp / wp.mean(1, keepdim=True)
                if args.w_xpt > 0:
                    xp = torch.nan_to_num(tr["bd"][idx], nan=0.0).gather(1, tr["bd_ix"][idx][:, None, None].expand(-1, 1, 2))[:, 0]
                    d2 = ((RRc[None] - xp[:, :1] / DR) ** 2 + (ZZc[None] - xp[:, 1:] / DZ) ** 2)
                    wx = 1.0 + args.w_xpt * torch.exp(-0.5 * d2 / args.w_xpt_sigma ** 2) * tr["bd_ok"][idx][:, None].float()
                    wp = (wp if wp.dim() == 2 else wp[None]) * wx
                    wp = wp / wp.mean(1, keepdim=True)
                loss_psi = (((psi_p - tr["psi"][idx]) ** 2) * wp).mean() / r_scale ** 2
                if args.lam_axis > 0:
                    dr_p, dz_p = axis.offsets_smooth(psi_p, tr["ax_iz"][idx], tr["ax_ir"][idx])
                    loss_axis = ((dr_p - tr["ax_dr"][idx]) ** 2 + (dz_p - tr["ax_dz"][idx]) ** 2).mean()
                else:
                    loss_axis = psi_p.new_zeros(())
                y = (tr["scal"][idx] - s_mu) / s_sd
                m = torch.isfinite(y)
                loss_scal = (((sc_p - torch.nan_to_num(y)) ** 2) * scal_w)[m].mean() if m.any() else psi_p.new_zeros(())
                loss_bd = (bdry.loss(psi_p, tr["psi"][idx], tr["bd"][idx], tr["bd_w"][idx], tr["bd_ix"][idx],
                                     tr["bd_ok"][idx]) if args.lam_bd > 0 else psi_p.new_zeros(()))
                if args.lam_grad > 0:
                    if args.grad_w == "env":
                        gw = None
                    elif args.grad_w == "band":
                        gw = wp if wp.dim() == 2 else None
                    else:
                        gw = tr["lcfs_in"][idx].float() * r_over
                        gw = gw / gw[:, inside].mean().clamp(min=1e-6)
                    loss_gr = grad_loss(psi_p, tr["psi"][idx], inside, g_scale, gw)
                else:
                    loss_gr = psi_p.new_zeros(())
                if args.lam_grad2 > 0:
                    if args.grad_w2 == "env":
                        gw2 = None
                    elif args.grad_w2 == "band":
                        gw2 = wp if wp.dim() == 2 else None
                    else:
                        gw2 = tr["lcfs_in"][idx].float() * r_over
                        gw2 = gw2 / gw2[:, inside].mean().clamp(min=1e-6)
                    loss_gr = loss_gr + (args.lam_grad2 / max(args.lam_grad, 1e-12)) * grad_loss(psi_p, tr["psi"][idx], inside, g_scale, gw2)
                if args.lam_lap > 0:
                    if args.lap_w == "env":
                        lw = None
                    elif args.lap_w == "band":
                        lw = wp if wp.dim() == 2 else None
                    else:
                        lw = tr["lcfs_in"][idx].float() * r_over
                        lw = lw / lw[:, inside].mean().clamp(min=1e-6)
                    loss_lap = lap_loss(psi_p, tr["psi"][idx], inside, l_scale, lw)
                else:
                    loss_lap = psi_p.new_zeros(())
                if args.head == "scalar":
                    loss = loss_scal
                else:
                    loss = (loss_psi + args.lam_scal * loss_scal + args.lam_axis * loss_axis
                            + args.lam_bd * loss_bd + args.lam_grad * loss_gr + args.lam_lap * loss_lap)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
                if torch.isfinite(gnorm):
                    opt.step()
                    if args.ema > 0 and step >= ema_from:
                        with torch.no_grad():
                            if ema_p is None:
                                ema_p = [p.detach().clone() for p in net_params]
                            else:
                                torch._foreach_lerp_(ema_p, [p.detach() for p in net_params], 1.0 - args.ema)
                    tot_psi += float(loss_psi.detach()); tot_scal += float(loss_scal.detach())
                    tot_axis += float(loss_axis.detach()); tot_bd += float(loss_bd.detach()); tot_gr += float(loss_gr.detach())
                else:                                   # NaN/Inf gradient: drop this step, keep the schedule going
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                for sc_ in scheds:
                    sc_.step()
                step += 1
            model.net.eval()
            n_ok = max(1, steps_per_epoch - skipped)
            rec = {"epoch": ep + 1, "train_loss_psi": tot_psi / n_ok,
                   "train_loss_scal": tot_scal / n_ok, "train_loss_axis": tot_axis / n_ok,
                   "train_loss_bd": tot_bd / n_ok, "train_loss_grad": tot_gr / n_ok,
                   "lr": scheds[-1].get_last_lr()[0], "skipped_steps": skipped,
                   "time_s": time.time() - t0}
            if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
                rec.update(evaluate(model))
                if args.curves:
                    rec["curves_train"] = target_curves(model, tr_sample)
                    rec["curves_val"] = target_curves(model, ho)
            if str(ep + 1) in args.save_epochs.split(","):
                check_finite_weights(model.net, f"ckpt_ep{ep + 1:03d}.pt")
                torch.save(checkpoint(model, n_in, {"epoch": ep + 1}), args.out / f"ckpt_ep{ep + 1:03d}.pt")
            log["epochs"].append(rec)
            print(f"ep {ep + 1:3d}  loss_psi {rec['train_loss_psi']:.4f}  loss_scal {rec['train_loss_scal']:.4f}  "
                  f"loss_axis {rec['train_loss_axis']:.4f}  bd {rec['train_loss_bd']:.4f}  grad {rec['train_loss_grad']:.4f}  "
                  + (f"| holdout R2psi {rec['r2_psi']:.6f}  q95 {rec['r2_q95']:.4f}  betaN {rec['r2_betaN']:.4f}  "
                     f"Raxis {rec['r2_Raxis']:.4f}  Zaxis {rec['r2_Zaxis']:.4f}  lcfs {rec['lcfs_disp_rms_mm']:.2f}mm  "
                     f"rmse_in {rec['rmse_psi_inside']:.5f}  " if "r2_psi" in rec else "")
                  + (f"skipped {skipped}  " if skipped else "")
                  + f"{rec['time_s']:.0f}s", flush=True)
            log["wall_s"] = time.time() - T_START
            (args.out / "log.json").write_text(json.dumps(log, indent=1))
        if ema_p is not None:                      # (ch1) evaluate the raw weights, then switch to the averaged weights
            log["final_raw"] = evaluate(model)
            with torch.no_grad():
                for p, e in zip(net_params, ema_p):
                    p.copy_(e)
            log["final_ema"] = evaluate(model)
            print(f"EMA {args.ema} from step {ema_from}: raw {log['final_raw']}\n    ema {log['final_ema']}", flush=True)
            del ema_p
        check_finite_weights(model.net, "model.pt")
        torch.save(checkpoint(model, n_in, {"px": str(args.px) if args.px else None, "firstpass": args.firstpass,
                                            "extra_stats": {k: (v[0].cpu(), v[1].cpu()) for k, v in extra_stats.items()}}),
                   args.out / "model.pt")

    # ---- public test set through the training-time input pipeline
    if args.test_cache is not None:
        for k in ("psi", "w_band", "bd", "bd_w", "ax_dr", "ax_dz", "lcfs_in"):   # free training memory before loading the test set
            tr.pop(k, None)
        if tr_sample is not None:
            tr_sample.clear()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        stems_te = sorted(p.stem for p in args.test_cache.glob("*.npz") if p.stem not in ("basis", "meta"))
        te, te_off, te_T = load_shots(args.test_cache, stems_te, with_psi=False, px=args.test_px)
        if args.helicity:
            add_helicity(te, stems_te, args.test_cache.parent / f"bt_sign_{args.test_cache.name}.json")
        te = to_dev(te)
        add_extra_inputs(te)
        psi_t, sc_t = predict(model, te)
        psi_t, sc_t, t_te = psi_t.cpu().numpy(), sc_t.cpu().numpy(), te["t"].cpu().numpy()
        check_finite_preds(psi_t, sc_t, "test predictions")
        tdir = args.out / "preds_test"
        tdir.mkdir(exist_ok=True)
        for s, o, n in zip(stems_te, te_off, te_T):
            np.savez(tdir / f"{s}.npz", psi=psi_t[o:o + n].reshape(n, 65, 65).astype(np.float32),
                     q95=sc_t[o:o + n, 0].astype(np.float32), betaN=sc_t[o:o + n, 1].astype(np.float32),
                     t=t_te[o:o + n].astype(np.float32))
        print(f"wrote {len(stems_te)} test predictions -> {tdir}", flush=True)
        del te, psi_t, sc_t

    if final:
        (args.out / "log.json").write_text(json.dumps(log, indent=1))
        print("final fit: no held-out predictions written", flush=True)
        return 0
    # held-out predictions for the competition scorer
    psi_p, sc_p = predict(model, ho)
    pdir = args.out / "preds"
    pdir.mkdir(exist_ok=True)
    psi_p, sc_p = psi_p.cpu().numpy(), sc_p.cpu().numpy()
    check_finite_preds(psi_p, sc_p, "held-out predictions")
    for s, o, n in zip(ho_stems, ho_off, ho_T):
        np.savez(pdir / f"{s}.npz", psi=psi_p[o:o + n].reshape(n, 65, 65).astype(np.float32),
                 q95=sc_p[o:o + n, 0].astype(np.float32), betaN=sc_p[o:o + n, 1].astype(np.float32))
    log["wall_s"] = time.time() - T_START
    (args.out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {len(ho_stems)} held-out predictions -> {pdir}; wall-clock {log['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
