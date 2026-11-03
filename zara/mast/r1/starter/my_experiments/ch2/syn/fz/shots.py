"""Frozen copy of xg/shots.py (2026-10-05 11:30 UTC) for the pegs model.

Shot records in one machine-agnostic format, from any dataset adapter.

A shot record is a dict of per-frame arrays:
    machine  str          name of the MachineSpec it belongs to
    t        (T,)  ms
    I        (T, C) current-column values, in the units the MachineSpec's el_scale expects
    ip       (T,)  A, > 0 (the model's frame: a reversed-Ip shot has I and ip multiplied by sign)
    sign     float +1 / -1
    itf      (T,)  A per TF turn (|.|): B_phi R = mu0 tf_turns itf / (2 pi)
    ts_pts   (N, 2) Thomson scattering positions R, Z (m); NaN rows = unused
    ts_te    (T, N) keV, ts_ne (T, N) 1e19 m^-3 (NaN = missing)
  labelled datasets only:
    psi      (T, nZ, nR) stored convention of the dataset; q95, betaN (T,)
    lab      dict of profile labels from labels.py (coef, beta_p, li, geometry), when available
Adapters below read the challenge caches; a new tokamak only needs a function returning this dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from machine import D3D_CACHE, D3D_PX, MAST_CACHE, SYNTH_SETS, SYNTHSET_CACHE, MULTI_ROOT, MULTI_MACHINES, multi_root
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

LABELS_D3D = Path(f"{R1_ROOT}/cache/xg/labels_d3d")


def clean_ip(ip, frac=0.05, floor=2e4):
    """Frames whose Ip is below frac x the shot maximum (or below floor A) while the equilibrium exists are
    measurement / timing errors (e.g. DIII-D Ip time axis): replace them by the nearest reliable frame's Ip."""
    ip = np.asarray(ip, np.float64).copy()
    good = ip > max(frac * np.nanmax(ip), floor)
    if good.any() and not good.all():
        idx = np.arange(len(ip))
        gi = idx[good]
        nearest = gi[np.abs(idx[:, None] - gi[None]).argmin(1)]
        ip[~good] = ip[nearest[~good]]
    return ip, good


def d3d_shot(stem: str, split: str = "train", labels: bool = True) -> dict:
    d = np.load(D3D_CACHE[split] / f"{stem}.npz")
    p = np.load(D3D_PX[split] / f"{stem}.npz")
    rec = {"machine": "DIII-D", "stem": stem, "t": d["t"].astype(np.float64), "I": d["I"].astype(np.float64),
           "ip": clean_ip(d["ip"])[0], "ip_raw": d["ip"].astype(np.float64), "sign": float(d["sign"]),
           "itf": np.abs(d["bt"].astype(np.float64)),
           "ts_pts": p["pos"].astype(np.float64), "ts_te": p["te"].astype(np.float64), "ts_ne": p["ne"].astype(np.float64)}
    if "psi" in d.files:
        rec.update(psi=d["psi"], q95=d["q95"].astype(np.float64), betaN=d["betaN"].astype(np.float64))
        if labels and (LABELS_D3D / f"{stem}.npz").exists():
            rec["lab"] = {k: v.astype(np.float64) for k, v in np.load(LABELS_D3D / f"{stem}.npz").items()}
    return rec


def mast_shot(stem: str, split: str = "test") -> dict:
    d = np.load(MAST_CACHE[split] / f"{stem}.npz")
    R = np.concatenate([d["ts_R"], d["te_R"]]).astype(np.float64)
    pts = np.stack([R, np.zeros_like(R)], 1)                                     # MAST Thomson is on the midplane
    te = np.concatenate([d["ts_Te"], d["te_Te"]], 1).astype(np.float64) / 1e3    # eV -> keV
    ne = np.concatenate([d["ts_ne"], d["te_ne"]], 1).astype(np.float64) / 1e19   # m^-3 -> 1e19
    ip = d["ip"].astype(np.float64)
    s = 1.0 if np.nanmedian(ip) >= 0 else -1.0
    return {"machine": "MAST", "stem": stem, "t": d["t"].astype(np.float64), "I": d["I"].astype(np.float64) * s,
            "ip": clean_ip(ip * s)[0], "ip_raw": ip * s, "sign": s, "itf": np.abs(d["itf"].astype(np.float64)), "ts_pts": pts, "ts_te": te, "ts_ne": ne}


def synth_root(machine: str) -> Path:
    """Single-frame record directories: test sets (cache/synthsets/<m>/synth) and training machines (cache/pegs/multi/<m>/src)."""
    return multi_root(machine) / machine / "src" if machine in MULTI_MACHINES else SYNTHSET_CACHE / machine / "synth"


def synth_shot(machine: str, stem: str) -> dict:
    """Single-frame records of the synthetic test sets (cache/synthsets/<machine>/synth, xg/synthset_to_cache.py format:
    t, I (A-turn per circuit), ip (A), itf (A-turn total), ts_pts, ts_te (keV), ts_ne (1e19))."""
    d = np.load(synth_root(machine) / f"{stem}.npz")
    ip = d["ip"].astype(np.float64)
    s = 1.0 if np.nanmedian(ip) >= 0 else -1.0
    return {"machine": machine, "stem": stem, "t": d["t"].astype(np.float64), "I": d["I"].astype(np.float64) * s,
            "ip": np.clip(ip * s, 1e4, None), "ip_raw": ip * s, "sign": s, "itf": np.abs(d["itf"].astype(np.float64)),
            "ts_pts": d["ts_pts"].astype(np.float64), "ts_te": d["ts_te"].astype(np.float64), "ts_ne": d["ts_ne"].astype(np.float64)}


def stems(machine: str, split: str) -> list[str]:
    if machine in SYNTH_SETS or machine in MULTI_MACHINES:
        return sorted(p.stem for p in synth_root(machine).glob("*.npz"))
    root = {"DIII-D": D3D_CACHE, "MAST": MAST_CACHE}[machine][split]
    return sorted(p.stem for p in root.glob("*.npz") if p.stem not in ("basis", "meta"))
