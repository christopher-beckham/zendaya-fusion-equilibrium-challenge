"""Synthetic DIII-D equilibria with FreeGSNKE, same recipe as gen_mast.py, in the DIII-D schema.

Purpose: calibrate the synthetic-data proxy. Real DIII-D labels exist, so a model's score on this
set can be compared with its score on the real held-out shots the synthetic frames came from.

Recipe (mirrors the MAST batch so the two are comparable):
  * Input pool: every EFIT time with finite inputs and |Ip| > 100 kA from the 704 shots of the
    C1 validation split (10 %, seed 0, over the sorted diii_d_train file names; see
    my_experiments/c1/holdout.py), sorted by |Ip|. Only inputs are read for generation.
  * Sobol design: 4,096 points over the same 15 dimensions and ranges as MAST, mapped to DIII-D
    coil groups: scale_vertical -> F6, F7 (outer, vertical field); scale_divertor -> the other
    F-coils (shaping); scale_solenoid -> the E-coil.
  * Static forward solve on the challenge 65x65 DIII-D grid; accept converged, diverted, in-range
    equilibria; keep the 500 with the lowest Sobol index.
  * Labels from the FreeGSNKE solution. psi is written in the DIII-D EFIT convention (magnetic
    axis at the flux minimum), which is the negative of FreeGSNKE's psi for these currents.
  * --validate N: solves N real held-out frames unscaled (generic profiles, as in the design) and
    compares FreeGSNKE psi with the real EFIT psi; writes validation_vs_efit.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/fsx/zara.ercan/fusion-equilibrium-challenge-starter")
import d3d_machine as dm  # noqa: E402
import synth_common as sc  # noqa: E402
from data_fixes import fix_d3d_ip_times  # noqa: E402

DATA = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/challenge_data/data/diii_d_train"
SCHEMA_REF = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/parquet_data/d3d_shot_203702.parquet"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MACHINE_DIR = os.path.join(ROOT, "machines", "DIIID_challenge")
POOL_PATH = os.path.join(ROOT, "cache", "d3d_input_pool.npz")
OUT = os.path.join(ROOT, "d3d_synthetic")
N_POINTS = 4096
N_KEEP = 500
SEED = 2028
TOL = 1e-5
WINDOW_MS = 100.0
POINT_TIMEOUT_S = 30
GRID = (0.84, 2.54, -1.6, 1.6)

SIG_COLUMNS = dm.COIL_COLUMNS + ["magnetics_bcoil", "ip"]
VERTICAL = ["magnetics_F6A", "magnetics_F6B", "magnetics_F7A", "magnetics_F7B"]
SHAPING = [c for c in dm.F_COLUMNS if c not in VERTICAL]
DIMS = [
    ("frame", 0.0, 1.0),
    ("betap_offset", -0.35, 0.10),
    ("alpha_m", 1.0, 2.5),
    ("alpha_n", 0.7, 2.5),
    ("scale_vertical", 0.92, 1.12),
    ("scale_divertor", 0.85, 1.15),
    ("scale_solenoid", 0.80, 1.20),
    ("scale_ip", 0.90, 1.10),
    ("scale_tf", 0.85, 1.15),
    ("ne0_scale", 0.7, 1.3),
    ("ne_edge_frac", 0.10, 0.40),
    ("ne_a", 1.5, 4.0),
    ("ne_b", 0.5, 1.5),
    ("te_edge", 10.0, 40.0),
    ("sol_width", 0.03, 0.12),
]
DIM_NAMES = [d[0] for d in DIMS]


def holdout_files():
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    perm = np.random.RandomState(0).permutation(len(files))
    hold = set(perm[: int(round(0.10 * len(files)))].tolist())
    return [f for i, f in enumerate(files) if i in hold]


def sobol_design():
    u = qmc.Sobol(d=len(DIMS), scramble=True, seed=SEED).random(N_POINTS)
    lo = np.array([d[1] for d in DIMS])
    hi = np.array([d[2] for d in DIMS])
    return lo + u * (hi - lo)


def read_signals(path, cols_extra=()):
    cols = ["efit_times", "magnetics_time", "magnetics_plasma_current",
            "magnetics_plasma_current_times", "magnetics_bcoil"] + dm.COIL_COLUMNS + list(cols_extra)
    row = pq.read_table(path, columns=cols).to_pylist()[0]
    sig = {c: (np.asarray(row["magnetics_time"], float), np.asarray(row[c], float))
           for c in dm.COIL_COLUMNS + ["magnetics_bcoil"]}
    sig["ip"] = (np.asarray(fix_d3d_ip_times(row), float), np.asarray(row["magnetics_plasma_current"], float))
    return row, sig


def at(sig, c, t):
    tt, vv = sig[c]
    return float(np.interp(t, tt, vv))


# ---------------------------------------------------------------------------------------------
def _pool_one(path):
    row, sig = read_signals(path)
    out = []
    for t in np.asarray(row["efit_times"], float):
        v = [at(sig, c, t) for c in SIG_COLUMNS]
        if np.all(np.isfinite(v)) and abs(v[-1]) > 100:
            out.append((os.path.basename(path), t, v))
    return out


def build_pool(nproc):
    with Pool(nproc) as p:
        res = p.map(_pool_one, holdout_files(), chunksize=4)
    rows = [r for rr in res for r in rr]
    vals = np.array([r[2] for r in rows])
    order = np.argsort(np.abs(vals[:, -1]))
    os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
    np.savez(POOL_PATH, shot=np.array([rows[i][0] for i in order]),
             t=np.array([rows[i][1] for i in order]), values=vals[order], columns=np.array(SIG_COLUMNS))
    print(f"pool: {len(rows)} frames from {len(holdout_files())} held-out shots -> {POOL_PATH}")


def scaled_inputs(base, p):
    out = dict(base)
    for c in VERTICAL:
        out[c] *= p.get("scale_vertical", 1.0)
    for c in SHAPING:
        out[c] *= p.get("scale_divertor", 1.0)
    out[dm.E_COLUMN] *= p.get("scale_solenoid", 1.0)
    out["ip"] *= p.get("scale_ip", 1.0)
    out["magnetics_bcoil"] *= p.get("scale_tf", 1.0)
    return out


class Solver:
    def __init__(self):
        from freegsnke import GSstaticsolver, equilibrium_update
        Rmin, Rmax, Zmin, Zmax = GRID
        with sc.quiet():
            self.tok = dm.build_tokamak(MACHINE_DIR)
            self.eq = equilibrium_update.Equilibrium(tokamak=self.tok, Rmin=Rmin, Rmax=Rmax,
                                                     Zmin=Zmin, Zmax=Zmax, nx=65, ny=65)
            self.gs = GSstaticsolver.NKGSsolver(self.eq)
        self.bz = {lab: float(c.controlBz(np.array([1.7]), np.array([0.0]))[0]) for lab, c in self.tok.coils}

    def shafranov_betap(self, currents, Ip, R0=1.7, a=0.6, kappa=1.8, li=0.9):
        Bv = sum(self.bz[k] * currents[k] for k in currents)
        lam = -Bv * 4 * np.pi * R0 / (sc.MU0 * Ip) - np.log(8 * R0 / (a * np.sqrt(kappa))) + 1.5
        return lam - li / 2

    def run(self, inputs, p):
        from freegsnke.jtor_update import ConstrainBetapIp
        eq = self.eq
        currents = dm.coil_currents_A(inputs)
        Ip = inputs["ip"] * 1e3
        fvac = dm.fvac_from_bcoil(inputs["magnetics_bcoil"])
        # Reversed-current shots (about 7% of DIII-D frames): FreeGSNKE's profile and q-profile
        # code assume Ip > 0, so solve the mirror problem with every current sign-flipped.
        # Flipping all currents only flips psi, and the written psi (-psi_FreeGSNKE, axis at
        # the minimum) then matches real DIII-D EFIT, which keeps the axis at the minimum for
        # both current directions.
        if Ip < 0:
            currents = {k: -v for k, v in currents.items()}
            Ip = -Ip
        betap = float(np.clip(self.shafranov_betap(currents, Ip) + p["betap_offset"], 0.03, 1.2))
        for c, v in currents.items():
            eq.tokamak.set_coil_current(c, v)
        eq.plasma_psi = sc.gaussian_psi_guess(eq, 1.7, 0.0, 0.4, 1.8, Ip)
        prof = ConstrainBetapIp(eq=eq, betap=betap, Ip=Ip, fvac=fvac,
                                alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])
        with sc.quiet():
            self.gs.solve(eq=eq, profiles=prof, constrain=None, target_relative_tolerance=TOL,
                          max_solving_iterations=60)
        rel = float(self.gs.relative_change)
        if rel > 10 * TOL:
            return "not_converged", None
        if prof.flag_limiter:
            return "limited", None
        lab = sc.equilibrium_labels(eq, prof, fvac)
        if lab is None:
            return "no_lcfs", None
        if not (1.3 < lab["r_axis"] < 2.1 and abs(lab["z_axis"]) < 0.4):
            return "axis_out_of_range", None
        if not (5.0 < lab["volume"] < 40.0):
            return "volume_out_of_range", None
        if not (np.isfinite(lab["q95"]) and 1.5 < lab["q95"] < 40):
            return "bad_q95", None
        lab.update(betap_target=betap, fvac=fvac, ip=inputs["ip"] * 1e3, rel_change=rel)
        return "ok", (prof, lab)


def real_ne_level(path, t):
    """90th-percentile core density of the real Thomson nearest t; zeros mean missing."""
    try:
        row = pq.read_table(path, columns=["thomson_core_times", "thomson_core_ne"]).to_pylist()[0]
        tt = np.asarray(row["thomson_core_times"], float)
        k = int(np.argmin(np.abs(tt - t)))
        if abs(tt[k] - t) > 15:
            return np.nan
        v = np.asarray(row["thomson_core_ne"][k], float)
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.percentile(v, 90)) if v.size >= 8 else np.nan
    except Exception:
        return np.nan


def chords_of(path):
    row = pq.read_table(path, columns=["thomson_chord_name", "thomson_chord_R", "thomson_chord_Z"]).to_pylist()[0]
    return (np.asarray(row["thomson_chord_name"]).astype(str), np.asarray(row["thomson_chord_R"], float),
            np.asarray(row["thomson_chord_Z"], float))


def run_worker(worker, n_workers, out_dir, limit=None):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    pool = np.load(POOL_PATH)
    X = sobol_design()
    solver = Solver()
    for i in range(worker, N_POINTS if limit is None else limit, n_workers):
        fn = os.path.join(raw, f"point_{i:05d}.npz")
        if os.path.exists(fn):
            continue
        p = dict(zip(DIM_NAMES, X[i]))
        k = min(int(p["frame"] * len(pool["t"])), len(pool["t"]) - 1)
        base = dict(zip(SIG_COLUMNS, pool["values"][k]))
        inputs = scaled_inputs(base, p)
        shot, t = str(pool["shot"][k]), float(pool["t"][k])
        t0 = time.time()
        pid = os.fork()  # killed after POINT_TIMEOUT_S, as in gen_machine.py
        if pid == 0:
            code = 1
            try:
                try:
                    status, res = solver.run(inputs, p)
                except Exception:
                    status, res = "error", None
                rec = dict(index=i, status=status, pool_index=k, shot=shot, t=t, params=X[i],
                           sec=time.time() - t0)
                if status == "ok":
                    prof, lab = res
                    rng = np.random.default_rng([SEED, i])
                    path = os.path.join(DATA, shot)
                    ne0 = real_ne_level(path, t)
                    ne0_src = "thomson"
                    if not np.isfinite(ne0) or not (5e18 < ne0 < 2e20):
                        ne0, ne0_src = 5e19, "default"
                    ne0 *= p["ne0_scale"]
                    names, cR, cZ = chords_of(path)
                    ne, Te = sc.thomson_profiles(solver.eq, prof, cR, cZ, ne0, p["ne_edge_frac"],
                                                 p["ne_a"], p["ne_b"], p["te_edge"], p["sol_width"], rng)
                    # DIII-D EFIT convention: magnetic axis at the flux minimum
                    rec.update(psi=-solver.eq.psi().T.copy(), ne=ne, Te=Te, ne0=ne0, ne0_source=ne0_src,
                               lcfs=lab.pop("lcfs"), labels=json.dumps(lab), inputs=json.dumps(inputs))
                np.savez(fn + ".tmp.npz", **rec)
                os.replace(fn + ".tmp.npz", fn)
                code = 0
            finally:
                os._exit(code)
        deadline = t0 + POINT_TIMEOUT_S
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.time() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                break
            time.sleep(0.05)
        if not os.path.exists(fn):
            np.savez(fn, index=i, status="timeout", pool_index=k, shot=shot, t=t, params=X[i],
                     sec=time.time() - t0)
        print(f"[w{worker}] point {i}: {np.load(fn)['status']} ({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------------------------
def assemble(out_dir):
    raw = os.path.join(out_dir, "raw")
    recs, counts = [], {}
    for i in range(N_POINTS):
        fn = os.path.join(raw, f"point_{i:05d}.npz")
        if not os.path.exists(fn):
            counts["missing"] = counts.get("missing", 0) + 1
            continue
        r = np.load(fn)
        s = str(r["status"])
        counts[s] = counts.get(s, 0) + 1
        if s == "ok" and len(recs) < N_KEEP:
            recs.append({k: r[k] for k in r.files})
    print("design outcomes:", counts, "-> keeping", len(recs))
    pq_dir = os.path.join(out_dir, "parquet")
    os.makedirs(pq_dir, exist_ok=True)
    ref = pq.read_schema(SCHEMA_REF)
    first = sorted(glob.glob(os.path.join(DATA, "*.parquet")))[0]
    static = pq.read_table(first, columns=dm.GEOM_COLUMNS + ["efit_grid_R", "efit_grid_Z"]).to_pylist()[0]
    label_names = ["q95", "beta_n", "beta_p", "beta_t", "li", "r_axis", "z_axis", "geom_axis_r",
                   "geom_axis_z", "elongation", "minor_radius", "volume", "psi_axis", "psi_boundary",
                   "dsep", "betap_target", "ip", "fvac"]
    cons = {k: [] for k in ["psi", "lcfs_r", "lcfs_z", "inputs", "labels", "idx", "params", "shot", "t"]}
    for r in recs:
        i = int(r["index"])
        lab = json.loads(str(r["labels"]))
        lab["psi_axis"], lab["psi_boundary"] = -lab["psi_axis"], -lab["psi_boundary"]  # DIII-D sign
        inputs = json.loads(str(r["inputs"]))
        p = dict(zip(DIM_NAMES, r["params"]))
        t = float(r["t"])
        shot = str(r["shot"])
        src, sig = read_signals(os.path.join(DATA, shot))
        tm = sig["magnetics_bcoil"][0]
        w = (tm >= t - WINDOW_MS) & (tm <= t + WINDOW_MS)
        series = {c: sig[c][1][w] for c in dm.COIL_COLUMNS + ["magnetics_bcoil"]}
        series["ip"] = np.interp(tm[w], *sig["ip"])
        sc_series = scaled_inputs(series, p)
        names, cR, cZ = chords_of(os.path.join(DATA, shot))
        thom = pq.read_table(os.path.join(DATA, shot), columns=["thomson_core_R", "thomson_edge_spatial"]).to_pylist()[0]
        core = np.char.startswith(names, "TS_core")
        tang = np.char.startswith(names, "TS_tangential")
        ne, Te = np.asarray(r["ne"]), np.asarray(r["Te"])
        C = np.asarray(r["lcfs"])
        out = {"source": "DIII-D_synthetic_freegsnke", "efit_times": [t],
               "efit_psirz": [np.asarray(r["psi"]).tolist()]}
        for c in dm.COIL_COLUMNS + ["magnetics_bcoil"]:
            out[c] = sc_series[c].tolist()
        out.update({
            "thomson_core_times": [t], "thomson_core_Te": [Te[core].tolist()],
            "thomson_core_ne": [ne[core].tolist()], "thomson_core_R": thom["thomson_core_R"],
            "thomson_edge_times": [t], "thomson_edge_Te": [Te[tang].tolist()],
            "thomson_edge_ne": [ne[tang].tolist()], "thomson_edge_spatial": thom["thomson_edge_spatial"],
            "magnetics_plasma_current": sc_series["ip"].tolist(),
            "magnetics_plasma_current_times": tm[w].tolist(),
            "magnetics_dsep": [lab["dsep"]], "magnetics_dsep_times": [t],
            "magnetics_time": tm[w].tolist(),
            "efit_beta_n": [lab["beta_n"]], "efit_li": [lab["li"]], "efit_q95": [lab["q95"]],
            "efit_r_axis": [lab["r_axis"]], "efit_z_axis": [lab["z_axis"]],
            "efit_lcfs_n": [float(len(C))], "efit_lcfs_r": [C[:, 0].tolist()], "efit_lcfs_z": [C[:, 1].tolist()],
            "efit_grid_R": static["efit_grid_R"], "efit_grid_Z": static["efit_grid_Z"],
        })
        for c in dm.GEOM_COLUMNS:
            out[c] = static[c]
        out.update({"thomson_chord_name": names.tolist(), "thomson_chord_R": cR.tolist(),
                    "thomson_chord_Z": cZ.tolist()})
        out.update({"synth_sobol_index": i, "synth_params": json.dumps(p),
                    "synth_origin_shot": shot.replace(".parquet", ""), "synth_origin_time": t,
                    "synth_beta_p": lab["beta_p"], "synth_elongation": lab["elongation"],
                    "synth_psi_axis": lab["psi_axis"], "synth_psi_boundary": lab["psi_boundary"],
                    "synth_volume": lab["volume"], "synth_minor_radius": lab["minor_radius"],
                    "synth_e_turns": dm.E_TURNS, "synth_ne0": float(r["ne0"])})
        fields = [ref.field(k) if k in ref.names else pa.field(k, pa.array([v]).type) for k, v in out.items()]
        pq.write_table(pa.Table.from_pylist([out], schema=pa.schema(fields)),
                       os.path.join(pq_dir, f"d3d_synth_{i:05d}.parquet"), compression="zstd")
        cons["psi"].append(np.asarray(r["psi"], np.float32))
        cons["lcfs_r"].append(C[:, 0])
        cons["lcfs_z"].append(C[:, 1])
        cons["inputs"].append([inputs[c] for c in SIG_COLUMNS])
        cons["labels"].append([lab[k] for k in label_names])
        cons["idx"].append(i)
        cons["params"].append(r["params"])
        cons["shot"].append(shot.replace(".parquet", ""))
        cons["t"].append(t)
    nb = max(len(x) for x in cons["lcfs_r"])
    pad = lambda x: np.r_[x, np.full(nb - len(x), np.nan)]  # noqa: E731
    np.savez_compressed(
        os.path.join(out_dir, f"d3d_synthetic_{len(recs)}.npz"),
        psi=np.stack(cons["psi"]), grid_R=np.asarray(static["efit_grid_R"]), grid_Z=np.asarray(static["efit_grid_Z"]),
        lcfs_r=np.stack([pad(x) for x in cons["lcfs_r"]]), lcfs_z=np.stack([pad(x) for x in cons["lcfs_z"]]),
        inputs=np.array(cons["inputs"]), input_names=np.array(SIG_COLUMNS),
        labels=np.array(cons["labels"]), label_names=np.array(label_names),
        sobol_index=np.array(cons["idx"]), sobol_params=np.stack(cons["params"]),
        sobol_param_names=np.array(DIM_NAMES), origin_shot=np.array(cons["shot"]), origin_time=np.array(cons["t"]),
    )
    json.dump({"design_points": N_POINTS, "kept": len(recs), "outcomes": counts, "dims": DIMS,
               "seed": SEED, "e_turns": dm.E_TURNS, "source_shots": "C1 validation split (10%, seed 0)"},
              open(os.path.join(out_dir, "design_summary.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------
def validate(n_frames, out_dir, worker=0, n_workers=1):
    """Unscaled real held-out frames, generic profiles: FreeGSNKE psi vs real EFIT psi."""
    pool = np.load(POOL_PATH)
    rng = np.random.default_rng(99)
    pick = np.sort(rng.choice(len(pool["t"]), n_frames, replace=False))
    solver = Solver()
    p = {"betap_offset": 0.0, "alpha_m": 1.8, "alpha_n": 1.2}
    rows = []
    for j, k in enumerate(pick):
        if j % n_workers != worker:
            continue
        base = dict(zip(SIG_COLUMNS, pool["values"][k]))
        shot, t = str(pool["shot"][k]), float(pool["t"][k])
        tmp = os.path.join(out_dir, f"_validation_w{worker:02d}_frame.json")
        t0 = time.time()
        pid = os.fork()  # killed after POINT_TIMEOUT_S, as in run_worker
        if pid == 0:
            code = 1
            try:
                try:
                    status, res = solver.run(base, p)
                except Exception:
                    status = "error"
                rec = dict(shot=shot, t=t, status=status)
                if status == "ok":
                    row = pq.read_table(os.path.join(DATA, shot), columns=["efit_times", "efit_psirz"]).to_pylist()[0]
                    jt = int(np.argmin(np.abs(np.asarray(row["efit_times"]) - t)))
                    efit = np.asarray(row["efit_psirz"][jt], float)
                    syn = -solver.eq.psi().T
                    rec["r2_psi"] = float(1 - np.sum((efit - syn) ** 2) / np.sum((efit - efit.mean()) ** 2))
                    rec["corr"] = float(np.corrcoef(efit.ravel(), syn.ravel())[0, 1])
                json.dump(rec, open(tmp, "w"))
                code = 0
            finally:
                os._exit(code)
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.time() > t0 + POINT_TIMEOUT_S:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                break
            time.sleep(0.05)
        rec = json.load(open(tmp)) if os.path.exists(tmp) else dict(shot=shot, t=t, status="timeout")
        if os.path.exists(tmp):
            os.remove(tmp)
        rows.append(rec)
        if rec["status"] == "ok":
            print(f"[v{worker}] {shot} t={t:.0f}: R2 {rec['r2_psi']:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out_dir, f"_validation_part{worker:02d}.json"), "w"))


def merge_validation(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "_validation_part*.json"))):
        rows += json.load(open(f))
    ok = [r for r in rows if r["status"] == "ok"]
    r2 = np.array([r["r2_psi"] for r in ok])
    summary = {"frames": len(rows), "accepted": len(ok),
               "r2_psi_median": float(np.median(r2)), "r2_psi_p10": float(np.percentile(r2, 10)),
               "r2_psi_p90": float(np.percentile(r2, 90)), "r2_psi_pooled_mean": float(np.mean(r2)),
               "note": "FreeGSNKE psi (generic profiles, force-balance beta_p, no fitting) vs real EFIT psi "
                       "on unscaled held-out frames"}
    json.dump({"summary": summary, "frames": rows}, open(os.path.join(out_dir, "validation_vs_efit.json"), "w"), indent=1)
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--merge-validation", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.build_pool:
        build_pool(a.nproc)
    elif a.assemble:
        assemble(a.out)
    elif a.validate:
        validate(a.validate, a.out, a.worker, a.n_workers)
    elif a.merge_validation:
        merge_validation(a.out)
    else:
        try:
            run_worker(a.worker, a.n_workers, a.out, a.limit)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
