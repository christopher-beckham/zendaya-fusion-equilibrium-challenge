"""Second synthetic MAST batch, calibrated to the real MAST TF current and Thomson levels.

Differences from gen_mast.py (batch 1):
  * TF current: each source frame's real value, unscaled (MAST ran TF at two fixed levels).
  * Plasma pressure: the axis pressure comes from that frame's real Thomson,
        p_axis = e * ne_peak * Te_peak * (1 + Ti/Te) * pressure_jitter,
    and is imposed with FreeGSNKE's ConstrainPaxisIp profile (batch 1 set beta_p from radial
    force balance instead, which left Te unconstrained).
  * Synthetic Thomson density: core density = real ne_peak * ne_jitter; Te then follows from the
    solved pressure as Te = p / (e ne (1 + Ti/Te)) + T_edge.
  * Only pool frames with a usable real Thomson profile within 15 ms are eligible.
"peak" = 95th percentile over the core channels of the profile nearest the EFIT time.
Rule 3: all calibration uses MAST input signals (coil currents, Ip, TF, Thomson). No MAST EFIT
reconstruction is read. Output layout matches batch 1 (raw/, parquet/, consolidated .npz).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mast as gm  # noqa: E402
import mast_machine as mm  # noqa: E402
import synth_common as sc  # noqa: E402

ROOT = gm.ROOT
OUT = os.path.join(ROOT, "mast_synthetic_v2")
POOL_TS = os.path.join(ROOT, "cache", "mast_input_pool_thomson.npz")
N_POINTS = 4096
N_KEEP = 500
SEED = 2027
DIMS = [
    ("frame", 0.0, 1.0),               # quantile of the Ip-sorted pool of Thomson-valid frames
    ("alpha_m", 1.0, 2.5),
    ("alpha_n", 0.7, 2.5),
    ("scale_vertical", 0.92, 1.12),
    ("scale_divertor", 0.90, 1.10),
    ("scale_solenoid", 0.85, 1.15),
    ("scale_ip", 0.95, 1.05),
    ("ti_te", 0.7, 1.5),
    ("pressure_jitter", 0.8, 1.2),
    ("ne_jitter", 0.9, 1.1),
    ("ne_edge_frac", 0.10, 0.40),
    ("ne_a", 1.5, 4.0),
    ("ne_b", 0.5, 1.5),
    ("te_edge", 10.0, 40.0),
    ("sol_width", 0.03, 0.12),
]
DIM_NAMES = [d[0] for d in DIMS]


def sobol_design():
    u = qmc.Sobol(d=len(DIMS), scramble=True, seed=SEED).random(N_POINTS)
    lo = np.array([d[1] for d in DIMS])
    hi = np.array([d[2] for d in DIMS])
    return lo + u * (hi - lo)


# ---------------------------------------------------------------------------------------------
def _peaks_one(args):
    shot, ts = args
    row = pq.read_table(os.path.join(gm.DATA, shot), columns=[
        "thomson_core_times", "thomson_core_Te", "thomson_core_ne"]).to_pylist()[0]
    tt = np.asarray(row["thomson_core_times"], float)
    if tt.size == 0:
        return [(np.nan, np.nan)] * len(ts)
    te = np.asarray(row["thomson_core_Te"], float)[:, 1:]  # drop ghost channel 0
    ne = np.asarray(row["thomson_core_ne"], float)[:, 1:]
    out = []
    for t in ts:
        k = int(np.argmin(np.abs(tt - t)))
        if abs(tt[k] - t) > 15 or np.isfinite(te[k]).sum() < 10 or np.isfinite(ne[k]).sum() < 10:
            out.append((np.nan, np.nan))
            continue
        out.append((float(np.nanpercentile(te[k], 95)), float(np.nanpercentile(ne[k], 95))))
    return out


def build_pool_thomson(nproc):
    pool = np.load(gm.POOL_PATH)
    by_shot: dict = {}
    for k, (s, t) in enumerate(zip(pool["shot"], pool["t"])):
        by_shot.setdefault(str(s), []).append((k, float(t)))
    keys = list(by_shot)
    with Pool(nproc) as p:
        res = p.map(_peaks_one, [(s, [t for _, t in by_shot[s]]) for s in keys], chunksize=4)
    te = np.full(len(pool["t"]), np.nan)
    ne = np.full(len(pool["t"]), np.nan)
    for s, vals in zip(keys, res):
        for (k, _), (a, b) in zip(by_shot[s], vals):
            te[k], ne[k] = a, b
    ok = np.isfinite(te) & np.isfinite(ne) & (te > 50) & (te < 5000) & (ne > 2e18) & (ne < 1.5e20)
    np.savez(POOL_TS, te_peak=te, ne_peak=ne, valid=ok)
    print(f"pool Thomson peaks: {ok.sum()} of {len(ok)} frames usable -> {POOL_TS}")


# ---------------------------------------------------------------------------------------------
class SolverV2(gm.Solver):
    def run_paxis(self, inputs, p, paxis):
        from freegsnke.jtor_update import ConstrainPaxisIp
        eq = self.eq
        currents = mm.coil_currents_A(inputs, gm.SOL_FACTOR)
        Ip = inputs["magnetics_plasma_current"] * 1e3
        fvac = mm.fvac_from_tf(inputs["magnetics_tf_current"])
        for c, v in currents.items():
            eq.tokamak.set_coil_current(c, v)
        eq.plasma_psi = sc.gaussian_psi_guess(eq, 0.9, 0.0, 0.4, 1.6, Ip)
        prof = ConstrainPaxisIp(eq=eq, paxis=paxis, Ip=Ip, fvac=fvac,
                                alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])
        with sc.quiet():
            self.gs.solve(eq=eq, profiles=prof, constrain=None,
                          target_relative_tolerance=gm.TOL, max_solving_iterations=60)
        rel = float(self.gs.relative_change)
        if rel > 10 * gm.TOL:
            return "not_converged", None
        if prof.flag_limiter:
            return "limited", None
        lab = sc.equilibrium_labels(eq, prof, fvac)
        if lab is None:
            return "no_lcfs", None
        if not (0.45 < lab["r_axis"] < 1.3 and abs(lab["z_axis"]) < 0.5):
            return "axis_out_of_range", None
        if not (1.0 < lab["volume"] < 20.0):
            return "volume_out_of_range", None
        if not (np.isfinite(lab["q95"]) and 1.5 < lab["q95"] < 40):
            return "bad_q95", None
        lab["betap_target"] = lab["beta_p"]
        lab["paxis"] = paxis
        lab["fvac"] = fvac
        lab["ip"] = Ip
        lab["rel_change"] = rel
        return "ok", (prof, lab)


def run_worker(worker, n_workers, out_dir, limit=None):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    pool = np.load(gm.POOL_PATH)
    pts = np.load(POOL_TS)
    valid = np.where(pts["valid"])[0]  # pool is Ip-sorted, so valid stays Ip-sorted
    X = sobol_design()
    solver = SolverV2()
    chord = pq.read_table(sorted(glob.glob(os.path.join(gm.DATA, "*.parquet")))[0],
                          columns=["thomson_chord_R", "thomson_chord_Z"]).to_pylist()[0]
    chord_R = np.asarray(chord["thomson_chord_R"], float)
    chord_Z = np.asarray(chord["thomson_chord_Z"], float)
    idx = range(worker, N_POINTS if limit is None else limit, n_workers)
    for i in idx:
        fn = os.path.join(raw, f"point_{i:05d}.npz")
        if os.path.exists(fn):
            continue
        p = dict(zip(DIM_NAMES, X[i]))
        k = int(valid[min(int(p["frame"] * len(valid)), len(valid) - 1)])
        base = dict(zip(gm.SIG_COLUMNS, pool["values"][k]))
        inputs = gm.scaled_inputs(base, p)
        te_pk, ne_pk = float(pts["te_peak"][k]), float(pts["ne_peak"][k])
        paxis = sc.E_CHARGE * ne_pk * te_pk * (1 + p["ti_te"]) * p["pressure_jitter"]
        t0 = time.time()
        try:
            status, res = solver.run_paxis(inputs, p, paxis)
        except Exception:
            status, res = "error", None
        rec = dict(index=i, status=status, pool_index=k, shot=str(pool["shot"][k]),
                   t=float(pool["t"][k]), params=X[i], sec=time.time() - t0,
                   real_te_peak=te_pk, real_ne_peak=ne_pk)
        if status == "ok":
            prof, lab = res
            rng = np.random.default_rng([SEED, i])
            ne0 = ne_pk * p["ne_jitter"]
            ne, Te = thomson_v2(solver.eq, prof, chord_R, chord_Z, ne0, p, rng)
            rec.update(psi=solver.eq.psi().T.copy(), ne=ne, Te=Te, ne0=ne0,
                       ne0_source="thomson_peak", lcfs=lab.pop("lcfs"), labels=json.dumps(lab),
                       inputs=json.dumps(inputs))
        np.savez(fn, **rec)
        print(f"[w{worker}] point {i}: {status} ({rec['sec']:.1f}s)", flush=True)


def thomson_v2(eq, prof, R, Z, ne0, p, rng):
    """As synth_common.thomson_profiles, with Te = p / (e ne (1 + Ti/Te)) + T_edge."""
    ne, te = sc.thomson_profiles(eq, prof, R, Z, ne0, p["ne_edge_frac"], p["ne_a"], p["ne_b"],
                                 p["te_edge"], p["sol_width"], rng, noise=0.0)
    # thomson_profiles assumes Ti = Te (factor 2); rescale the pressure-driven part
    te_p = np.clip(te - p["te_edge"], 0, None) * 2.0 / (1.0 + p["ti_te"])
    te = np.where(te > p["te_edge"], te_p + p["te_edge"], te)
    ne = ne * (1 + 0.05 * rng.standard_normal(ne.shape))
    te = np.abs(te * (1 + 0.05 * rng.standard_normal(te.shape)))
    return ne, te


def assemble(out_dir):
    """Reuse batch 1's assembler with this batch's design."""
    gm.DIMS, gm.DIM_NAMES, gm.N_POINTS, gm.N_KEEP, gm.SEED = DIMS, DIM_NAMES, N_POINTS, N_KEEP, SEED
    gm.assemble(out_dir)
    s = json.load(open(os.path.join(out_dir, "design_summary.json")))
    s["batch"] = "v2: TF unscaled, pressure and density calibrated to real Thomson peaks"
    json.dump(s, open(os.path.join(out_dir, "design_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--build-pool-thomson", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only points below this index")
    ap.add_argument("--nproc", type=int, default=14)
    a = ap.parse_args()
    if a.build_pool_thomson:
        build_pool_thomson(a.nproc)
    elif a.assemble:
        assemble(a.out)
    else:
        try:
            run_worker(a.worker, a.n_workers, a.out, a.limit)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
