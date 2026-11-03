#!/usr/bin/env python3
"""Quality-control relabelling of every accepted design point in the four synthetic sets.

For each accepted raw record (raw/point_XXXXX.npz):
  1. re-solve the equilibrium from the record's stored inputs (coil currents, Ip, field, profile
     parameters, same initial guess and tolerance as the generator);
  2. drop the record if the re-solved flux does not reproduce the stored flux (R2 < 0.9999): the
     inputs then admit more than one equilibrium and the label is ambiguous;
  3. take the plasma boundary from the challenge scorer's own extraction (fusion_scoring.lcfs)
     with a geometry-only mask, replacing FreeGSNKE's separatrix polygon, which occasionally drops
     part of the boundary;
  4. recompute every boundary-dependent label (elongation, minor radius, geometric axis, volume,
     <p>_V, beta_t, beta_N, beta_p, li) and the synthetic Thomson profiles (and, for SPARC/ITER,
     the Greenwald-derived core density and temperature);
  5. drop the record if a label falls outside the plausibility limits below.
The record keeps its Sobol index; qc_done / qc_drop / qc_reason / qc_resolve_r2 are added. The
assemblers then keep the 500 lowest-index records that passed.

    python relabel_qc.py --set mast --worker k --n-workers N
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/fusion_scoring")
import qc_boundary as qb  # noqa: E402
import synth_common as sc  # noqa: E402

R2_MIN = 0.9999
TIMEOUT_S = 40
LIMITS = {  # dataset -> label -> (lo, hi)
    "mast": {"q95": (1.5, 40), "beta_n": (0.0, 7.0), "li": (0.3, 3.0), "elongation": (1.0, 3.0)},
    "d3d": {"q95": (1.5, 40), "beta_n": (0.0, 6.0), "li": (0.3, 3.0), "elongation": (1.0, 3.0)},
    "sparc": {"q95": (1.8, 40), "beta_n": (0.0, 5.0), "li": (0.3, 3.0), "elongation": (1.0, 3.0)},
    "iter": {"q95": (1.8, 40), "beta_n": (0.0, 5.0), "li": (0.3, 3.0), "elongation": (1.0, 3.0)},
}


class Machine:
    """Per-dataset solver, grid, mask, input conventions and Thomson chords."""

    def __init__(self, ds):
        self.ds = ds
        if ds == "mast":
            import gen_mast as g
            self.g, self.solver = g, g.Solver()
            d = np.load(os.path.join(GEN, "mast_synthetic", "mast_synthetic_500.npz"))
            self.chords = (d["thomson_chord_R"], d["thomson_chord_Z"])
            self.tol, self.iters, self.seed = g.TOL, 60, g.SEED
        elif ds == "d3d":
            import gen_d3d as g
            self.g, self.solver = g, g.Solver()
            self.tol, self.iters, self.seed = g.TOL, 60, g.SEED
        else:
            import gen_machine as g
            self.g, self.solver = g, g.MachineSolver(ds.upper())
            self.chords = g.chords(ds.upper())[:2]
            self.tol, self.iters, self.seed = g.TOL, 80, g.MACHINES[ds.upper()]["seed"]
            if ds == "sparc":
                self.bases = json.load(open(g.bases_path("SPARC")))
        self.eq = self.solver.eq
        self.gs = self.solver.gs
        self.R, self.Z = self.eq.R[:, 0], self.eq.Z[0, :]
        self.mask = qb.limiter_mask(ds, self.R, self.Z)

    def names(self):
        if self.ds in ("mast", "d3d"):
            return self.g.DIM_NAMES
        c = self.g.MACHINES[self.ds.upper()]
        return [d[0] for d in c["dims"] + c.get("derived_dims", [])]

    def setup(self, rec):
        """currents, Ip (positive, as solved), Ip as recorded, fvac, betap, guess, psi sign."""
        p = dict(zip(self.names(), rec["params"]))
        lab = json.loads(str(rec["labels"]))
        if self.ds == "mast":
            mm = __import__("mast_machine")
            inp = json.loads(str(rec["inputs"]))
            cur = mm.coil_currents_A(inp, self.g.SOL_FACTOR)
            ip = inp["magnetics_plasma_current"] * 1e3
            fvac = mm.fvac_from_tf(inp["magnetics_tf_current"])
            guess = (0.9, 0.4, 1.6)
            sign, ip_rec = 1.0, ip
        elif self.ds == "d3d":
            dm = __import__("d3d_machine")
            inp = json.loads(str(rec["inputs"]))
            cur = dm.coil_currents_A(inp)
            ip = inp["ip"] * 1e3
            ip_rec = ip
            fvac = dm.fvac_from_bcoil(inp["magnetics_bcoil"])
            if ip < 0:
                cur = {k: -v for k, v in cur.items()}
                ip = -ip
            guess = (1.7, 0.4, 1.8)
            sign = -1.0
        else:
            cur = json.loads(str(rec["currents"]))
            ip, fvac = lab["ip"], lab["fvac"]
            ip_rec = ip
            if self.ds == "sparc":
                b = self.bases[int(p["base_index"])]
                guess = (b["R0"], 0.6 * b["a"], b["kappa"])
            else:
                guess = (p["R0"], 0.6 * p["a"], p["kappa"])
            sign = 1.0
        betap = lab.get("betap_target", p.get("betap"))
        return p, lab, cur, ip, ip_rec, fvac, betap, guess, sign

    def process(self, rec):
        from freegsnke.jtor_update import ConstrainBetapIp
        from lcfs import extract_lcfs
        p, lab_old, cur, ip, ip_rec, fvac, betap, guess, sign = self.setup(rec)
        eq = self.eq
        for k, v in cur.items():
            eq.tokamak.set_coil_current(k, v)
        eq.plasma_psi = sc.gaussian_psi_guess(eq, guess[0], 0.0, guess[1], guess[2], ip)
        prof = ConstrainBetapIp(eq=eq, betap=betap, Ip=ip, fvac=fvac, alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])
        with sc.quiet():
            self.gs.solve(eq=eq, profiles=prof, constrain=None, target_relative_tolerance=self.tol,
                          max_solving_iterations=self.iters)
        out = {"qc_done": True, "qc_drop": False, "qc_reason": ""}
        psi = sign * eq.psi().T
        old = np.asarray(rec["psi"], float)
        r2 = float(1 - np.sum((old - psi) ** 2) / np.sum((old - old.mean()) ** 2))
        out["qc_resolve_r2"] = r2
        if float(self.gs.relative_change) > 10 * self.tol or prof.flag_limiter or r2 < R2_MIN:
            out.update(qc_drop=True, qc_reason=f"not reproducible (R2 {r2:.6f})")
            return out
        C = extract_lcfs(psi, self.R, self.Z, qb.SCORER_MACHINE[self.ds], self.mask,
                         self.mask.astype(float), n_points=256)
        if C is None:
            out.update(qc_drop=True, qc_reason="scorer finds no boundary")
            return out
        C = np.vstack([C, C[:1]])
        lab = sc.equilibrium_labels(eq, prof, fvac, C=C)
        if lab is None:
            out.update(qc_drop=True, qc_reason="labels failed")
            return out
        for k in ["betap_target", "fvac", "rel_change", "dsep"]:
            if k in lab_old and k not in ("dsep",):
                lab[k] = lab_old[k]
        lab["ip"] = ip_rec
        lab["betap_target"] = betap
        lab["fvac"] = fvac
        from matplotlib.path import Path
        if not Path(C).contains_point((lab["r_axis"], lab["z_axis"])):
            out.update(qc_drop=True, qc_reason="axis outside boundary")
            return out
        rng = np.random.default_rng([self.seed, int(rec["index"])])
        if self.ds in ("sparc", "iter"):
            with sc.quiet():
                p_axis = float(np.atleast_1d(eq.pressure(np.array([0.0])))[0])
            n_gw = ip / 1e6 / (np.pi * lab["minor_radius"] ** 2) * 1e20
            ne0 = p["greenwald_fraction"] * n_gw / (p["ne_edge_frac"] + (1 - p["ne_edge_frac"]) * 0.6)
            te0 = p_axis / (sc.E_CHARGE * ne0 * (1 + p["ti_te"])) / 1e3
            lab.update(p_axis=p_axis, ne0=ne0, greenwald_fraction=p["greenwald_fraction"], te0_keV=te0)
            if not 1.0 <= te0 <= 35.0:
                out.update(qc_drop=True, qc_reason=f"core Te {te0:.1f} keV out of range")
                return out
            R, Z = self.chords
            ne, Te = self.g.synth_thomson(eq, prof, p, rng, R, Z, ne0)
        else:
            if self.ds == "d3d":
                _, R, Z = self.g.chords_of(os.path.join(self.g.DATA, str(rec["shot"])))
            else:
                R, Z = self.chords
            ne, Te = sc.thomson_profiles(eq, prof, R, Z, float(rec["ne0"]), p["ne_edge_frac"], p["ne_a"],
                                         p["ne_b"], p["te_edge"], p["sol_width"], rng)
        for k, (lo, hi) in LIMITS[self.ds].items():
            v = lab[k]
            if not (np.isfinite(v) and lo <= v <= hi):
                out.update(qc_drop=True, qc_reason=f"{k}={v:.3g} outside [{lo}, {hi}]")
                break
        lcfs = lab.pop("lcfs")
        out.update(psi=psi, lcfs=lcfs, labels=json.dumps(lab), ne=ne, Te=Te)
        return out


def run(ds, worker, n_workers):
    raw = os.path.join(GEN, f"{ds}_synthetic", "raw")
    files = sorted(glob.glob(os.path.join(raw, "point_*.npz")))
    m = Machine(ds)
    todo = []
    for f in files:
        r = np.load(f)
        if str(r["status"]) == "ok" and "qc_done" not in r.files:
            todo.append(f)
    mine = todo[worker::n_workers]
    for f in mine:
        rec = dict(np.load(f))
        tmp = f + ".qc.npz"
        t0 = time.time()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                try:
                    upd = m.process(rec)
                except Exception as e:
                    upd = {"qc_done": True, "qc_drop": True, "qc_reason": f"error {type(e).__name__}: {str(e)[:80]}"}
                np.savez(tmp, **upd)
                code = 0
            finally:
                os._exit(code)
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.time() > t0 + TIMEOUT_S:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                break
            time.sleep(0.05)
        upd = dict(np.load(tmp)) if os.path.exists(tmp) else {"qc_done": True, "qc_drop": True, "qc_reason": "timeout"}
        if os.path.exists(tmp):
            os.remove(tmp)
        rec.update(upd)
        np.savez(f + ".new.npz", **rec)
        os.replace(f + ".new.npz", f)
        print(f"[{ds} w{worker}] {os.path.basename(f)}: {'DROP ' + str(upd['qc_reason']) if bool(upd['qc_drop']) else 'ok'} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["mast", "d3d", "sparc", "iter"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    run(a.set, a.worker, a.n_workers)
