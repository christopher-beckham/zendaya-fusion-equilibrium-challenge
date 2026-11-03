"""ch2/intr: MAST machine-model variants for the centre-column intrusion work (label-free; machine geometry + inputs only).

A variant adds a missing external field to the MAST vacuum flux, consistently in the first pass, the embedding features and the
network's GS layer, by extending the machine's vacuum matrix M with extra "columns" and the shot current arrays I with their
values per frame:
  bv       uniform vertical field (mT, + = outward push on an Ip > 0 plasma): one column, flux B_z R^2 / 2
  passive  induced vessel / casing currents from the circuit model (ch2/syn/fz/passive.py output, A per element at the EFIT times):
           130 columns (one per passive element, flux per ampere = Gp), x scale, optionally x a per-group scale
           (groups: vessel = vessel elements at R >= 0.3 m, column = centre-column elements at R < 0.3 m, casing = coil casings)
  scale    multiplies existing coil columns (e.g. {"magnetics_p5u_current": 0.97})
Set by the environment variable INTR_VARIANT (JSON); empty = the shipped machine. Caches built under a variant use tags that
start with "_iv" and live in $R1_ROOT/cache/ch2/intr/caches."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

CACHE_ROOT = Path(f"{R1_ROOT}/cache/ch2/g4/caches")            # ch2/g4 copy (real-test cache symlinked from ch2/intr)
PASSIVE_DIRS = {"r1": f"{R1_ROOT}/cache/ch2/syn/passive/MAST", "r5": f"{R1_ROOT}/cache/ch2/syn/passive_r5/MAST"}
GEOM = Path(f"{R1_ROOT}/cache/xg/geometry/omas")


def cfg() -> dict:
    s = os.environ.get("INTR_VARIANT", "").strip()
    return json.loads(s) if s else {}


def _arr(txt, name):
    m = re.search(r"\b" + name + r"\s*=\s*(.*?)(?=\n\s*[A-Za-z_0-9]+\s*=|\n\s*/)", txt, re.S)
    out = []
    for tok in m.group(1).replace(",", " ").split():
        if "*" in tok:
            n, v = tok.split("*"); out += [float(v)] * int(n)
        else:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return np.array(out)


def passive_groups() -> np.ndarray:
    """(130,) 0 = vessel, 1 = centre column, 2 = casing; same element order as ch2/syn/fz/passive.py (MAST)."""
    txt = (GEOM / "mast/0/mhdin.dat").read_text()
    fcid = _arr(txt, "FCID").astype(int); rf = _arr(txt, "RF")
    m = np.isin(fcid, list(set(range(1, 13)) | set(range(21, 39))))
    g, r = fcid[m], rf[m]
    return np.where((g >= 21) & (g <= 34), np.where(r < 0.3, 1, 0), 2)


def extra_columns(c: dict | None = None) -> list:
    c = cfg() if c is None else c
    cols = []
    if c.get("bv") is not None:
        cols.append("intr_bv")
    if c.get("passive"):
        cols += [f"intr_pv{j}" for j in range(130)]
    return cols


def apply_machine(m, c: dict | None = None):
    """Extend / rescale the vacuum matrix of a MAST Machine in place (M rows follow m.spec.columns + extra_columns())."""
    c = cfg() if c is None else c
    if not c or m.name != "MAST":
        return m
    M = np.asarray(m.M, np.float64).copy()
    cols = list(m.spec.columns)
    for k, s in (c.get("scale") or {}).items():
        M[cols.index(k)] *= float(s)
    rows = []
    R, Z = np.asarray(m.grid_R, float), np.asarray(m.grid_Z, float)
    RR, _ = np.meshgrid(R, Z)
    if c.get("bv") is not None:
        rows.append((1e-3 * RR ** 2 / 2).ravel())                     # flux of a uniform 1 mT vertical field
    if c.get("passive"):
        Gp = np.load(Path(PASSIVE_DIRS[c["passive"].get("model", "r5")]) / "Gp.npy").astype(np.float64)
        rows += list(Gp)
    if rows:
        M = np.concatenate([M, np.stack(rows)], 0)
    m.M = M.astype(np.float32)
    return m


def augment_I(stem: str, t_ms: np.ndarray, I: np.ndarray, c: dict | None = None) -> np.ndarray:
    """Append the variant's extra column values to a shot's (T, 13) current array (Ip > 0 frame)."""
    c = cfg() if c is None else c
    if not c:
        return I
    T = len(I)
    ext = []
    if c.get("bv") is not None:
        if c["bv"] == "file":                                 # per-frame values {stem: (T,)} from a calibration step
            d = np.load(c["bv_file"])
            v = d[stem].astype(np.float64) if stem in d.files else np.zeros(T)
            ext.append(v[:T, None] if len(v) >= T else np.pad(v, (0, T - len(v)), mode="edge")[:, None])
        else:
            ext.append(np.full((T, 1), float(c["bv"])))
    if c.get("passive"):
        p = c["passive"]
        f = Path(PASSIVE_DIRS[p.get("model", "r5")]) / f"{stem}.npz"
        if f.exists():
            d = np.load(f); Iv = d["Iv"].astype(np.float64); tv = d["t"].astype(np.float64) * 1e3
            if len(Iv) != T or np.max(np.abs(tv - t_ms)) > 0.6:
                Iv = np.stack([np.interp(t_ms, tv, Iv[:, j]) for j in range(Iv.shape[1])], 1)
        else:
            Iv = np.zeros((T, 130))
        gsc = p.get("groups")
        if gsc:
            g = passive_groups()
            Iv = Iv * np.array([gsc.get(n, 1.0) for n in ("vessel", "column", "casing")])[g][None]
        ext.append(Iv * float(p.get("scale", 1.0)))
    return np.concatenate([I] + ext, 1) if ext else I


def apply_wall(m):
    """ch2/intr (INTR_WALL=<R metres>, MAST-geometry machines only): the GS layer's inner wall at the real centre-column radius.
    The shipped domain stops 3 cm outside the solenoid (first node R 0.2116 m), so a diverted plasma needs R_in > 0.2116 m and
    near-column plasmas (R_in 0.20-0.21 m) can only be represented as limited. Extend the domain one node inward where it is
    bounded by the solenoid clearance; physics.GSLayer then enforces the wall at exactly R = INTR_WALL by interpolation."""
    wall = os.environ.get("INTR_WALL", "").strip()
    if not wall or not m.name.startswith("MAST"):
        return m
    import hashlib
    from machine import Machine, XG_CACHE
    wR = float(wall)
    R, Z = np.asarray(m.grid_R, float), np.asarray(m.grid_Z, float)
    dom = np.asarray(m.domain, bool).copy()
    i1 = int(np.argmin(np.abs(R - R[dom.any(0)].min())))            # innermost domain column (0.2116 m on the MAST grid)
    i0 = i1 - 1
    if not (R[i0] < wR < R[i1]):
        raise ValueError(f"wall {wR} not between nodes {R[i0]:.4f} and {R[i1]:.4f}")
    rows = np.nonzero(dom[:, i1] & ~dom[:, i0])[0]
    dom[rows, i0] = True
    gkey = hashlib.sha1(np.concatenate([m.spec.grid_R, m.spec.grid_Z]).tobytes()).hexdigest()[:12]
    Gf = np.load(XG_CACHE / "machines" / f"cells_{gkey}.npy", mmap_mode="r")
    iz, ir = np.nonzero(dom)
    G = np.ascontiguousarray(Gf[iz * len(R) + ir])
    m2 = Machine(m.spec, m.M, dom, iz, ir, G)
    m2.wall_R, m2.wall_i0, m2.wall_i1, m2.wall_rows = wR, i0, i1, rows
    return m2


def shifted_currents(stem: str, t_ms: np.ndarray, shift_ms: float):
    """(T, 13) coil currents (A, cache column order) and Ip (A) re-sampled from the full-rate magnetics at t + shift (1 ms window
    mean, as c2/preprocess_mast.py does at t). Diagnostic for a time-base offset between magnetics and EFIT / Thomson."""
    import sys as _sys
    import pandas as pd
    _sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "c1"))   # this package's c1
    from features import window_mean
    cols = ["magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current", "magnetics_p3u_current",
            "magnetics_p4l_current", "magnetics_p4u_current", "magnetics_p5l_current", "magnetics_p5u_current",
            "magnetics_p6l_current", "magnetics_p6u_current", "magnetics_sol_current", "magnetics_tf_current",
            "magnetics_efps_current"]
    f = Path(f"{R1_ROOT}/fusion-equilibrium-challenge-starter/challenge_data/data/mast_public_test") / f"{stem}.parquet"
    row = pd.read_parquet(f, columns=["magnetics_time", "magnetics_plasma_current"] + cols).iloc[0]
    tm = np.asarray(row["magnetics_time"], np.float64)
    tq = t_ms + shift_ms
    I = np.column_stack([window_mean(tm, np.asarray(row[c], np.float64), tq, 1.0) for c in cols]) * 1e3
    ip = window_mean(tm, np.asarray(row["magnetics_plasma_current"], np.float64), tq, 1.0) * 1e3
    return I, ip
