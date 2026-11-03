"""MAST synthetic TRAINING equilibria (m2): forward FreeGSNKE solves at real MAST public-test input frames.

    python gen_m2.py --pool train --worker 0 --n-workers 64 [--n-points 131072]   -> /fsx/zara.ercan/more_data/m2/mast_train/raw/wNN/
    python gen_m2.py --pool val   --worker 0 --n-workers 16 --n-points 8192        -> /fsx/zara.ercan/more_data/m2/mast_val/raw/wNN/
    python gen_m2.py --summary --pool train

Rule 3: only challenge MAST inputs (coil currents, Ip, TF, Thomson levels) and the shipped coil geometry are read; no MAST
equilibrium reconstruction of any kind. Everything else is computed from the FreeGSNKE solution.

Differences from the test-set generators (more_data/generated/code/gen_mast*.py, which this file imports):
  * inputs nearly exact (jitters of a few %), every real frame of the pool eligible, shot-level split: 10 % of the pool's
    shots (by hash) form the 'val' pool and never appear in 'train';
  * two profile families with wide Topeol shapes: 'paxis' (axis pressure from the frame's real Thomson peak, FreeGSNKE
    ConstrainPaxisIp) and 'betap' (beta_p sampled, ConstrainBetapIp);
  * continuation retry: when the direct solve fails, solve at 30 % (then 65 %) of the pressure and warm-start the target;
  * Thomson stored noise-free (dropouts / calibration / noise are added when the records are converted for training);
  * solenoid convention: SOL_FACTOR 0.5 (challenge caveat: the column reads 2x the FAIR-MAST level-2 current).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import qmc

sys.path.insert(0, "/fsx/zara.ercan/more_data/generated/code")
import gen_mast as gm        # noqa: E402
import mast_machine as mm    # noqa: E402
import synth_common as sc    # noqa: E402

OUT_ROOT = "/fsx/zara.ercan/more_data/m2"
POOL_TS = os.path.join(gm.ROOT, "cache", "mast_input_pool_thomson.npz")
SOL_FACTOR = 0.5
TOL = 1e-5
TIMEOUT_S = 40          # per design point; FreeGSNKE occasionally hangs


class PointTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise PointTimeout()
VAL_FRAC = 0.10
SEEDS = {"train": 31001, "val": 31002}
DIMS = [
    ("frame", 0.0, 1.0),           # uniform over the pool's frames (train or val pool)
    ("family", 0.0, 1.0),          # < 0.5 paxis (if the frame has a usable Thomson), else betap
    ("alpha_m", 1.0, 2.5),
    ("alpha_n", 0.7, 2.5),
    ("betap_offset", -0.40, 0.15),   # betap = Shafranov-balance estimate of the real inputs + offset (gen_mast)
    ("paxis_jit", 0.85, 1.15),
    ("ti_te", 0.6, 1.6),
    ("s_vert", 0.97, 1.03),        # P4, P5
    ("s_div", 0.95, 1.05),         # P2, P3, P6
    ("s_sol", 0.94, 1.06),
    ("s_ip", 0.98, 1.02),
    ("ne_edge_frac", 0.10, 0.40),
    ("ne_a", 1.5, 4.0),
    ("ne_b", 0.5, 1.5),
    ("te_edge", 10.0, 40.0),
    ("sol_width", 0.03, 0.12),
]
DIM_NAMES = [d[0] for d in DIMS]


def design(n_points, seed):
    u = qmc.Sobol(d=len(DIMS), scramble=True, seed=seed).random(n_points)
    lo = np.array([d[1] for d in DIMS]); hi = np.array([d[2] for d in DIMS])
    return lo + u * (hi - lo)


def is_val_shot(shot: str) -> bool:
    return int(hashlib.sha1(shot.encode()).hexdigest()[:8], 16) % 1000 < int(VAL_FRAC * 1000)


def pool_indices(pool, which):
    v = np.array([is_val_shot(str(s)) for s in pool["shot"]])
    return np.where(v if which == "val" else ~v)[0]


def scaled_inputs(base, p):
    out = dict(base)
    for c in gm.VERTICAL:
        out[c] *= p["s_vert"]
    for c in gm.DIVERTOR:
        out[c] *= p["s_div"]
    out["magnetics_sol_current"] *= p["s_sol"]
    out["magnetics_plasma_current"] *= p["s_ip"]
    return out


class Solver(gm.Solver):
    def _prof(self, family, Ip, fvac, p, scale, paxis):
        from freegsnke.jtor_update import ConstrainBetapIp, ConstrainPaxisIp
        if family == "paxis":
            return ConstrainPaxisIp(eq=self.eq, paxis=paxis * scale, Ip=Ip, fvac=fvac, alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])
        return ConstrainBetapIp(eq=self.eq, betap=max(p["betap"] * scale, 0.02), Ip=Ip, fvac=fvac, alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])

    def betap_target(self, currents, Ip, p):
        return float(np.clip(self.shafranov_betap(currents, Ip) + p["betap_offset"], 0.03, 1.5))

    def _solve(self, prof, max_it):
        with sc.quiet():
            self.gs.solve(eq=self.eq, profiles=prof, constrain=None, target_relative_tolerance=TOL, max_solving_iterations=max_it)
        return float(self.gs.relative_change) <= 10 * TOL

    def run_m2(self, inputs, p, family, paxis):
        eq = self.eq
        currents = mm.coil_currents_A(inputs, SOL_FACTOR)
        Ip = inputs["magnetics_plasma_current"] * 1e3
        fvac = mm.fvac_from_tf(inputs["magnetics_tf_current"])
        for c, v in currents.items():
            eq.tokamak.set_coil_current(c, v)
        p = dict(p); p["betap"] = self.betap_target(currents, Ip, p)
        guess = sc.gaussian_psi_guess(eq, 0.9, 0.0, 0.4, 1.6, Ip)
        eq.plasma_psi = guess.copy()
        prof = self._prof(family, Ip, fvac, p, 1.0, paxis)
        path = "direct"
        ok = self._solve(prof, 80)
        if not ok:
            return "not_converged", None, path
        if prof.flag_limiter:                        # keep the map for limiter diagnostics (not a training record)
            try:
                lab = sc.equilibrium_labels(eq, prof, fvac)
            except Exception:  # noqa: BLE001
                lab = None
            return "limited", (prof, lab), path
        lab = sc.equilibrium_labels(eq, prof, fvac)
        if lab is None:
            return "no_lcfs", None, path
        if not (0.45 < lab["r_axis"] < 1.3 and abs(lab["z_axis"]) < 0.5):
            return "axis_out_of_range", None, path
        if not (1.0 < lab["volume"] < 20.0):
            return "volume_out_of_range", None, path
        if not (np.isfinite(lab["q95"]) and 1.5 < lab["q95"] < 40):
            return "bad_q95", None, path
        if not np.isfinite(lab["li"]):
            return "bad_li", None, path
        lab.update(fvac=fvac, ip=Ip, rel_change=float(self.gs.relative_change), family=family, paxis=paxis, solve_path=path, betap_target=p["betap"])
        return "ok", (prof, lab), path


def thomson_clean(eq, prof, R, Z, ne0, p):
    """Noise-free Thomson at the 146 chords, Te = p / (e ne (1 + Ti/Te)) + T_edge."""
    rng = np.random.default_rng(0)
    ne, te = sc.thomson_profiles(eq, prof, R, Z, ne0, p["ne_edge_frac"], p["ne_a"], p["ne_b"], p["te_edge"], p["sol_width"], rng, noise=0.0)
    te_p = np.clip(te - p["te_edge"], 0, None) * 2.0 / (1.0 + p["ti_te"])
    te = np.where(te > p["te_edge"], te_p + p["te_edge"], te)
    return ne, te


def run_worker(which, worker, n_workers, n_points):
    out = os.path.join(OUT_ROOT, f"mast_{which}")
    raw = os.path.join(out, "raw", f"w{worker:02d}")
    os.makedirs(raw, exist_ok=True)
    pool = np.load(gm.POOL_PATH)
    pts = np.load(POOL_TS)
    idx_pool = pool_indices(pool, which)
    X = design(n_points, SEEDS[which])
    solver = Solver()
    chord = pq.read_table(sorted(glob.glob(os.path.join(gm.DATA, "*.parquet")))[0], columns=["thomson_chord_R", "thomson_chord_Z"]).to_pylist()[0]
    chord_R = np.asarray(chord["thomson_chord_R"], float); chord_Z = np.asarray(chord["thomson_chord_Z"], float)
    t_start = time.time(); n_done = n_ok = 0
    order = np.random.default_rng(SEEDS[which] + 7).permutation(n_points)[worker::n_workers]   # random, not strided (Sobol stride subsets are biased)
    signal.signal(signal.SIGALRM, _alarm)
    for i in order:
        i = int(i)
        fn = os.path.join(raw, f"point_{i:06d}.npz")
        if os.path.exists(fn):
            continue
        p = dict(zip(DIM_NAMES, X[i]))
        k = int(idx_pool[min(int(p["frame"] * len(idx_pool)), len(idx_pool) - 1)])
        base = dict(zip(gm.SIG_COLUMNS, pool["values"][k]))
        inputs = scaled_inputs(base, p)
        te_pk, ne_pk, valid = float(pts["te_peak"][k]), float(pts["ne_peak"][k]), bool(pts["valid"][k])
        family = "paxis" if (p["family"] < 0.5 and valid) else "betap"
        paxis = sc.E_CHARGE * ne_pk * te_pk * (1 + p["ti_te"]) * p["paxis_jit"] if valid else np.nan
        t0 = time.time()
        signal.alarm(TIMEOUT_S)
        try:
            status, res, path = solver.run_m2(inputs, p, family, paxis)
        except PointTimeout:
            status, res, path = "timeout", None, "timeout"
        except Exception as e:  # noqa: BLE001
            status, res, path = "error", None, f"error:{type(e).__name__}"
        finally:
            signal.alarm(0)
        if status == "timeout":                     # the solver state is unknown after an interrupt: rebuild it
            solver = Solver()
        rec = dict(index=i, status=status, pool_index=k, shot=str(pool["shot"][k]), t=float(pool["t"][k]), params=X[i],
                   sec=time.time() - t0, family=family, solve_path=path, real_te_peak=te_pk, real_ne_peak=ne_pk)
        if status == "limited" and res is not None and res[1] is not None:
            lab = dict(res[1]); rec.update(psi=solver.eq.psi().T.copy().astype(np.float32), lcfs=np.asarray(lab.pop("lcfs"), np.float32), labels=json.dumps(lab))
        if status == "ok":
            prof, lab = res
            ne0 = (ne_pk if valid else 3e19) * (0.9 + 0.2 * np.random.default_rng([SEEDS[which], i]).random())
            ne, Te = thomson_clean(solver.eq, prof, chord_R, chord_Z, ne0, p)
            rec.update(psi=solver.eq.psi().T.copy().astype(np.float32), ne=ne.astype(np.float32), Te=Te.astype(np.float32), ne0=ne0,
                       lcfs=np.asarray(lab.pop("lcfs"), np.float32), labels=json.dumps(lab), inputs=json.dumps(inputs))
            n_ok += 1
        np.savez(fn, **rec)
        n_done += 1
        if n_done % 20 == 0:
            el = time.time() - t_start
            print(f"[w{worker}] {n_done} done, {n_ok} ok ({n_ok / n_done:.2f}), {el / n_done:.1f} s/point", flush=True)
    print(f"[w{worker}] finished: {n_done} points, {n_ok} ok in {(time.time() - t_start) / 60:.1f} min", flush=True)


def summary(which):
    files = sorted(glob.glob(os.path.join(OUT_ROOT, f"mast_{which}", "raw", "w*", "point_*.npz")))
    st, paths, secs = {}, {}, []
    fam = {"paxis": [0, 0], "betap": [0, 0]}
    for f in files:
        d = np.load(f, allow_pickle=True)
        s = str(d["status"]); st[s] = st.get(s, 0) + 1; secs.append(float(d["sec"]))
        pth = str(d["solve_path"]); paths[pth] = paths.get(pth, 0) + (s == "ok")
        fm = str(d["family"]); fam[fm][0] += 1; fam[fm][1] += (s == "ok")
    print(json.dumps({"points": len(files), "status": st, "ok_by_path": paths, "family[n, ok]": fam,
                      "sec_p50_p90": np.round(np.percentile(secs, [50, 90]), 2).tolist() if secs else None}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="train", choices=["train", "val"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=131072)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a.pool)
        return
    try:
        run_worker(a.pool, a.worker, a.n_workers, a.n_points)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
