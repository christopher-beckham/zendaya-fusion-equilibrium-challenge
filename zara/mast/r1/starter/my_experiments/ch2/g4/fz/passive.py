"""Passive-structure (vessel / casing) currents for any machine, from public geometry and the measured currents only.

The vessel and coil casings carry induced currents whenever the coil currents or Ip change (current ramps). They are
not among the inputs, and they shift the plasma (DIII-D holdout: the radial axis bias is 0 mm in steady frames and
-42 mm in the fastest-ramp quarter). Circuit model, integrated along each shot's full-rate signals:

    M_vv dI_v/dt + R_v I_v = - d/dt ( M_vc I_c + M_vp Ip )

M_vv: mutual inductances between passive elements (filament Green's functions, self-inductance of a rectangular
section on the diagonal), M_vc: coil columns -> passive elements (the machine's own coil elements and per-column units,
plus the uniform solenoid term where a spec has one), M_vp: a plasma filament at the domain centre. Implicit Euler on
a 1 ms grid. The induced currents' flux on the grid is added to the vacuum flux at the EFIT times.

Geometry: OMAS EFIT machine files ($R1_ROOT/cache/xg/geometry/omas): DIII-D vessel segments with EFIT's
resistances; MAST (EFUND 2019) vessel, centre column and casing groups, resistances for stainless steel (no resistances
in the file). Machine geometry only (allowed under rule 3).
    python passive.py --machine DIII-D --split train --subset holdout --max-shots 100     # caches I_v at EFIT times
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2] / "c1")); sys.path.insert(0, str(HERE.parents[3]))   # ch2/syn: c1 + repo root
from greens import coil_filaments, filament_psi        # noqa: E402

MU0 = 4e-7 * np.pi
GEOM = Path(f"{R1_ROOT}/cache/xg/geometry/omas")
OUT = Path(os.environ.get("PASSIVE_OUT", f"{R1_ROOT}/cache/ch2/syn/passive"))   # ch2/syn copy of xg/passive.py
RHO_SS = 7.4e-7 * float(os.environ.get("RHO_MULT", "1"))   # stainless steel resistivity (Ohm m) x RHO_MULT (ch2/syn sensitivity test)
DT = 1e-3                # integration step (s)


def _arr(txt, name):
    m = re.search(r"\b" + name + r"\s*=\s*(.*?)(?=\n\s*[A-Za-z_0-9]+\s*=|\n\s*/)", txt, re.S)
    if not m:
        return None
    out = []
    for tok in m.group(1).replace(",", " ").split():
        if "*" in tok:
            n, v = tok.split("*")
            try:
                out += [float(v)] * int(n)
            except ValueError:
                pass
        else:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return np.array(out)


def passive_elements(machine: str):
    """(N, 6) R, Z, W, H, angle1, angle2 and (N,) resistance (Ohm) of each passive element."""
    if machine == "DIII-D":
        txt = (GEOM / "d3d/168191/mhdin.dat").read_text()
        el = np.stack([_arr(txt, k) for k in ("RVS", "ZVS", "WVS", "HVS", "AVS", "AVS2")], 1)
        res = _arr(txt, "RSISVS")
        return el, res
    if machine == "MAST":
        txt = (GEOM / "mast/0/mhdin.dat").read_text()
        rf, zf, wf, hf = (_arr(txt, k) for k in ("RF", "ZF", "WF", "HF"))
        fcid = _arr(txt, "FCID").astype(int)
        passive_groups = set(range(1, 13)) | set(range(21, 39))   # casings 1-12, 35-38; vessel + centre column 21-34
        m = np.isin(fcid, list(passive_groups))
        el = np.stack([rf[m], zf[m], wf[m], hf[m], np.zeros(m.sum()), np.zeros(m.sum())], 1)
        res = RHO_SS * 2 * np.pi * el[:, 0] / (el[:, 2] * el[:, 3])
        return el, res
    raise ValueError(machine)


def _filaments(el, n=3):
    return [coil_filaments(*e, n) for e in el]


def mutual(src_fil, dst_fil):
    """Mutual inductance 2 pi G averaged over sub-filaments: (len(src), len(dst))."""
    M = np.zeros((len(src_fil), len(dst_fil)))
    for i, (sr, sz) in enumerate(src_fil):
        for j, (dr, dz) in enumerate(dst_fil):
            M[i, j] = 2 * np.pi * filament_psi(sr[:, None], sz[:, None], dr[None], dz[None]).mean()
    return M


class Passive:
    def __init__(self, m, machine: str):
        self.m = m
        el, self.Rv = passive_elements(machine)
        self.el = el
        fil = _filaments(el)
        N = len(el)
        self.Mvv = mutual(fil, fil)
        rg = 0.2235 * (el[:, 2] + el[:, 3])
        self.Mvv[np.arange(N), np.arange(N)] = MU0 * el[:, 0] * (np.log(8 * el[:, 0] / rg) - 2.0)
        # coil columns -> passive elements, with the spec's per-column element scales
        spec = m.spec
        cfil = _filaments(spec.elements, 2)
        Mce = mutual(cfil, fil)                                               # (E, N)
        self.Mvc = np.zeros((N, len(spec.columns)))
        for e in range(len(spec.elements)):
            self.Mvc[:, spec.el_col[e]] += spec.el_scale[e] * Mce[e]
        for c, u in spec.col_uniform.items():                                 # uniform solenoid flux: 2 pi u per unit
            self.Mvc[:, spec.columns.index(c)] += 2 * np.pi * u
        Rc, Zc = float(m.grid_R[m.cell_ir].mean()), float(m.grid_Z[m.cell_iz].mean())
        self.Mvp = 2 * np.pi * filament_psi(Rc, Zc, el[:, 0], el[:, 1])        # plasma filament at the domain centre
        # flux of each passive element on the grid, per ampere
        RR, ZZ = np.meshgrid(m.grid_R, m.grid_Z)
        self.Gp = np.stack([filament_psi(fr[:, None, None], fz[:, None, None], RR[None], ZZ[None]).mean(0).ravel()
                            for fr, fz in fil])                               # (N, P)
        A = self.Mvv / DT + np.diag(self.Rv)
        self.A_inv = np.linalg.inv(A)
        w = np.linalg.eigvals(np.linalg.solve(np.diag(self.Rv), self.Mvv))
        self.tau_ms = np.sort(np.real(w))[::-1][:3] * 1e3                     # longest L/R time constants

    def integrate(self, t, I, ip):
        """t (n,) s uniform DT, I (n, C) column values, ip (n,) A -> I_v (n, N)."""
        n = len(t)
        Iv = np.zeros((n, len(self.Rv)))
        drive = I @ self.Mvc.T + ip[:, None] * self.Mvp[None]                 # (n, N) flux linkage from coils + plasma
        for k in range(1, n):
            rhs = self.Mvv @ Iv[k - 1] / DT - (drive[k] - drive[k - 1]) / DT
            Iv[k] = self.A_inv @ rhs
        return Iv


# ----------------------------------------------------------------------------- full-rate signals per machine
def d3d_signals(stem, split):
    import pandas as pd
    from data_fixes import fix_d3d_ip_times
    from machine import ROOT
    sub = {"train": "diii_d_train", "test": "diii_d_public_test"}[split]
    row = pd.read_parquet(ROOT / "challenge_data" / "data" / sub / f"{stem}.parquet").iloc[0]
    from machine import d3d_spec
    cols = d3d_spec().columns
    tm = np.asarray(row["magnetics_time"], np.float64) / 1e3
    I = np.stack([np.asarray(row[c], np.float64) * (1e3 if c != "magnetics_ECOILA" else 1e3) for c in cols], 1)   # kA -> A
    ipt = np.asarray(fix_d3d_ip_times(row), np.float64) / 1e3
    ip = np.asarray(row["magnetics_plasma_current"], np.float64) * 1e3
    te = np.asarray(row["efit_times"], np.float64) / 1e3
    return tm, I, ipt, ip, te


def mast_signals(stem, split):
    import pandas as pd
    from machine import ROOT, mast_spec
    row = pd.read_parquet(ROOT / "challenge_data" / "data" / "mast_public_test" / f"{stem}.parquet").iloc[0]
    cols = mast_spec().columns
    tm = np.asarray(row["magnetics_time"], np.float64) / 1e3
    I = np.stack([np.asarray(row[c], np.float64) * 1e3 for c in cols], 1)
    ip = np.asarray(row["magnetics_plasma_current"], np.float64) * 1e3
    te = np.asarray(row["efit_times"], np.float64) / 1e3
    return tm, I, tm, ip, te


def shot_passive(args):
    machine, stem, split, sol_factor = args
    from machine import d3d, mast
    import shots as S
    out = OUT / machine / f"{stem}.npz"
    if out.exists():
        return stem
    P = _PASSIVE.get(machine)
    tm, I, ipt, ip, te = (d3d_signals if machine == "DIII-D" else mast_signals)(stem, split)
    # the cache's sign convention: currents x the shot's Ip sign (Ip > 0 frame)
    rec = S.d3d_shot(stem, split, labels=False) if machine == "DIII-D" else S.mast_shot(stem, split)
    s = rec["sign"]
    ok = np.isfinite(tm)
    tm, I = tm[ok], np.nan_to_num(I[ok]) * s
    o = np.argsort(tm, kind="stable"); tm, I = tm[o], I[o]
    o = np.argsort(ipt); ipt, ip = ipt[o], np.nan_to_num(ip[o]) * s
    t0, t1 = max(tm.min(), ipt.min()), te.max() + 0.002
    tg = np.arange(t0, t1, DT)
    Ig = np.stack([np.interp(tg, tm, I[:, j]) for j in range(I.shape[1])], 1)
    ipg = np.interp(tg, ipt, ip)
    Iv = P.integrate(tg, Ig, ipg)
    Ive = np.stack([np.interp(te, tg, Iv[:, j]) for j in range(Iv.shape[1])], 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, Iv=Ive.astype(np.float32), t=te.astype(np.float32))
    os.replace(tmp, out)
    return stem


_PASSIVE = {}


def _init(machine, sol_factor):
    from machine import d3d, mast
    m = d3d() if machine == "DIII-D" else mast(sol_factor)
    _PASSIVE[machine] = Passive(m, machine)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["DIII-D", "MAST"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", default="all", choices=["all", "holdout"])
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--sol-factor", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import shots as S
    if args.machine == "DIII-D" and args.subset == "holdout":
        sys.path.insert(0, str(HERE.parent / "c1" / "nn"))
        from train_v5 import split_stems, DATA_DIR
        stems = split_stems(DATA_DIR, 0.10, 0, 0)[1]
    else:
        stems = S.stems(args.machine, args.split)
    stems = stems[: args.max_shots or None]
    _init(args.machine, args.sol_factor)
    P = _PASSIVE[args.machine]
    print(f"{args.machine}: {len(P.Rv)} passive elements, longest L/R time constants {np.round(P.tau_ms, 2).tolist()} ms; "
          f"{len(stems)} shots", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.machine, args.sol_factor)) as ex:
        for i, _ in enumerate(ex.map(shot_passive, [(args.machine, s, args.split, args.sol_factor) for s in stems], chunksize=4)):
            if i % 50 == 0:
                print(f"  {i}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    np.save(OUT / args.machine / "Gp.npy", P.Gp.astype(np.float32))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
