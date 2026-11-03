"""GPU-resident frame sets for pegs training / prediction (one machine per set).

A FrameSet holds, for every kept frame: embeddings (X, Thomson tokens), the first-pass current, coil currents,
Ip, TF current, first-pass geometry, raw Thomson values, and for DIII-D training shots the targets: stored psi,
q95, betaN, the scorer functionals on the true psi and the scorer's true LCFS (64 points).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_cache import cache_dir, FP_KEYS                    # noqa: E402
from machine import MULTI_MACHINES                           # noqa: E402
from shots import synth_root                                 # noqa: E402

NN_CACHE = {"train": Path(f"{R1_ROOT}/cache/nn/diii_d_train"), "test": Path(f"{R1_ROOT}/cache/nn/diii_d_public_test")}
REFS = Path(f"{R1_ROOT}/cache/pegs/refs_d3d")
AUX = Path(f"{R1_ROOT}/cache/upegs")          # aux_d3d<tag>/<stem>.npz from build_aux.py (optional)
LABELS = Path(f"{R1_ROOT}/cache/xg/labels_d3d")


DATA_DIR = HERE.parents[3] / "challenge_data" / "data" / "diii_d_train"


def holdout_split(frac=0.10, seed=0):
    """The project's fixed validation split (same as c1/nn/train_v5.split_stems: 704 shots, seed 0)."""
    stems = [f.stem for f in sorted(DATA_DIR.glob("*.parquet"))]
    perm = np.random.RandomState(seed).permutation(len(stems))
    hold = set(perm[: int(round(frac * len(stems)))].tolist())
    return [s for i, s in enumerate(stems) if i not in hold], [s for i, s in enumerate(stems) if i in hold]


class FrameSet:
    def __init__(self, machine, split, stems, device, stride=1, labels=True, tag="", workers=16, aux=False):
        self.machine, self.split, self.stems, self.dev = machine, split, list(stems), device
        cdir = cache_dir(machine, split, tag)
        lab_d3d = labels and machine == "DIII-D" and split == "train"
        lab_multi = labels and machine in MULTI_MACHINES          # synthetic training machines: labels in the record itself
        lab = lab_d3d or lab_multi
        adir = AUX / f"aux_d3d{tag}"
        use_aux = lab_d3d and aux and adir.exists()
        self.has_aux = use_aux or (lab_multi and aux)

        def one(s):
            d = np.load(cdir / f"{s}.npz")
            T = len(d["ip"])
            keep = np.arange(0, T, stride)
            o = {k: d[k][keep] for k in ["X", "tok", "tmask", "Jfp", "I", "ip", "itf", "ts_te", "ts_ne"] + [f"fp_{k}" for k in FP_KEYS]}
            o["pts"] = d["ts_pts"]
            o["t"] = d["t"][keep]
            fidx = d["fidx"][keep] if "fidx" in d.files else keep          # frame index in the full shot
            o["T_full"] = T
            o["keep"] = keep
            if lab_multi:
                r = np.load(synth_root(machine) / f"{s}.npz")
                o["psi"] = r["psi"].reshape(len(keep), -1)[keep] if r["psi"].shape[0] == T else r["psi"].reshape(T, -1)[keep]
                o["q95"], o["betaN"] = r["q95"][keep], r["betaN"][keep]
                o["cons"], o["lcfs"] = r["cons"][keep], r["lcfs64"][keep]
                if aux:        # exact coils, no vessel: the true external-field correction is zero; no EFIT profile label
                    o["harm_star"] = np.zeros((len(keep), 8), np.float32)
                    o["coef"] = np.full((len(keep), 5), np.nan, np.float32)
            if lab_d3d:
                g = np.load(NN_CACHE[split] / f"{s}.npz")
                o["psi"] = g["psi"][fidx].reshape(len(keep), -1)
                o["q95"], o["betaN"] = g["q95"][fidx], g["betaN"][fidx]
                r = np.load(REFS / f"{s}.npz")
                o["cons"], o["lcfs"] = r["cons"][fidx], r["lcfs"][fidx]
                if use_aux:
                    x = np.load(adir / f"{s}.npz")        # built on the cache's own frames; subsample like the cache
                    assert np.array_equal(x["fidx"][keep], fidx), f"aux fidx mismatch {s}"
                    o["harm_star"], o["coef"] = x["harm"][keep].astype(np.float32), x["coef"][keep].astype(np.float32)
            return o
        with ThreadPoolExecutor(workers) as ex:
            shots = list(ex.map(one, self.stems))
        self.T = np.array([len(s["ip"]) for s in shots])
        self.T_full = np.array([s["T_full"] for s in shots])
        self.keep = [s["keep"] for s in shots]
        self.times = [s["t"] for s in shots]
        self.off = np.concatenate([[0], np.cumsum(self.T)[:-1]])
        self.N = len(self.T) and int(self.T.sum())
        nch = max(s["pts"].shape[0] for s in shots)
        def padn(a, axis_len=nch, fill=np.nan):
            if a.shape[1] == axis_len:
                return a
            pad = [(0, 0)] * a.ndim; pad[1] = (0, axis_len - a.shape[1])
            return np.pad(a, pad, constant_values=fill)
        cat = lambda k, f=lambda a: a: np.concatenate([f(s[k]) for s in shots])
        tt = lambda a, dt=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.d = {"X": tt(cat("X")), "tok": tt(cat("tok", lambda a: padn(a, fill=0)), torch.float16),
                  "tmask": tt(cat("tmask", lambda a: padn(a, fill=False)), torch.bool),
                  "Jfp": tt(cat("Jfp"), torch.float16), "I": tt(cat("I")), "ip": tt(cat("ip")).clamp(min=1e4), "itf": tt(cat("itf")),
                  "ts_te": tt(cat("ts_te", padn), torch.float16), "ts_ne": tt(cat("ts_ne", padn), torch.float16),
                  **{f"fp_{k}": tt(cat(f"fp_{k}")) for k in FP_KEYS}}
        self.pts = tt(np.stack([padn(s["pts"][None])[0] for s in shots]))           # (S, N, 2)
        self.shot_of = torch.tensor(np.repeat(np.arange(len(shots)), self.T), device=device)
        if lab:
            # physical-range filters (isfinite alone lets absurd values through): out of range -> NaN -> masked in the losses
            psi = cat("psi"); q95 = cat("q95").astype(np.float64); bn = cat("betaN").astype(np.float64)
            psi[~(np.isfinite(psi) & (np.abs(psi) < 50)).all(1)] = np.nan
            q95[~((q95 > 1.0) & (q95 < 50.0))] = np.nan
            bn[~((bn >= 0.0) & (bn < 10.0))] = np.nan
            cons, lcfs = cat("cons"), cat("lcfs")
            cons[~np.isfinite(cons)] = np.nan
            self.n_bad = {"psi_frames": int(np.isnan(psi[:, 0]).sum()), "q95": int(np.isnan(q95).sum()), "betaN": int(np.isnan(bn).sum()),
                          "cons": int(np.isnan(cons).any(1).sum())}
            self.d["psi"] = tt(psi)
            self.d["q95"], self.d["betaN"] = tt(q95), tt(bn)
            self.d["cons"], self.d["lcfs"] = tt(cons), tt(lcfs, torch.float16)
            if self.has_aux:
                hs, cf = cat("harm_star").astype(np.float64), cat("coef").astype(np.float64)
                hs[~(np.isfinite(hs) & (np.abs(hs) < 0.2)).all(1)] = np.nan          # > 0.2 mu0 Ip R0 at the edge: not a correction
                cf[~np.isfinite(cf).all(1)] = np.nan
                self.n_bad["harm_star"] = int(np.isnan(hs[:, 0]).sum()); self.n_bad["coef"] = int(np.isnan(cf[:, 0]).sum())
                self.d["harm_star"], self.d["coef"] = tt(hs), tt(cf)
        self.labelled = lab
        # float16 storage: re-check the tokens and Thomson values after loading
        tokf = self.d["tok"].float()
        tok_ok = torch.isfinite(tokf).all(-1)
        self.d["tmask"] &= tok_ok
        self.d["tok"] = torch.where(tok_ok[..., None], self.d["tok"], torch.zeros_like(self.d["tok"]))
        for k in ("X", "Jfp", "I", "ip", "itf"):
            if not torch.isfinite(self.d[k].float() if k != "X" else torch.nan_to_num(self.d[k], nan=0.0)).all():
                raise ValueError(f"{machine} {split}: non-finite values in {k}")

    def batch(self, idx):
        b = {k: (v[idx].float() if v.dtype == torch.float16 else v[idx]) for k, v in self.d.items()}
        b["pts"] = self.pts[self.shot_of[idx]]
        return b

    def __len__(self):
        return self.N
