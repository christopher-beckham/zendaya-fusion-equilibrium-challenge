#!/usr/bin/env python3
"""Audit the synthetic test sets: structure, physics consistency, and end-to-end reproducibility.

    python audit_datasets.py --sets mast d3d sparc iter --resolve 25

Checks, per dataset:
  structure  every parquet readable; for MAST/DIII-D, column names and Arrow types match the
             real challenge reference rows; array shapes consistent (efit_* length T, psi (T,65,65),
             Thomson value arrays vs chord counts, magnetics series vs magnetics_time); no NaN/inf
             outside documented places; Sobol indices unique; parquet vs consolidated npz agree.
  physics    flux sign convention; psi at the axis and on the LCFS match the psi_axis/psi_boundary
             labels; LCFS closed and enclosing the axis; labels finite and plausible; Thomson hot
             inside the plasma and cold outside.
  resolve    re-solves a sample of equilibria from what the parquet row contains (coil columns,
             Ip, TF, profile parameters) and compares the flux with the stored efit_psirz. This
             catches unit, sign or scaling errors in the written input columns.
  loader     runs the C1 DIII-D input loader (my_experiments/c1/features.shot_inputs) on synthetic
             DIII-D rows and compares the feature NaN share with real held-out rows.
Writes ../audit_report.json and prints a summary. Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.abspath(os.path.join(HERE, ".."))
STARTER = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter"
sys.path.insert(0, HERE)
sys.path.insert(0, STARTER)

REF = {"mast": os.path.join(STARTER, "parquet_data/mast_shot_28348.parquet"),
       "d3d": os.path.join(STARTER, "parquet_data/d3d_shot_203702.parquet")}
AXIS_MAX = {"mast": True, "d3d": False, "sparc": True, "iter": True}
PLAUSIBLE = {"efit_q95": (1.5, 40), "efit_beta_n": (0, 8), "efit_li": (0.3, 3.0)}


class Report:
    def __init__(self):
        self.items = []

    def check(self, ds, name, ok, detail=""):
        self.items.append(dict(dataset=ds, check=name, ok=bool(ok), detail=detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def lcfs_of(row, j):
    r = np.asarray(row["efit_lcfs_r"][j], float)
    z = np.asarray(row["efit_lcfs_z"][j], float)
    n = int(row["efit_lcfs_n"][j])
    return r[:n], z[:n]


# ---------------------------------------------------------------------------------------------
def structure_and_physics(ds, rep):
    folder = os.path.join(GEN, f"{ds}_synthetic")
    files = sorted(glob.glob(os.path.join(folder, "parquet", "*.parquet")))
    rep.check(ds, "file count", len(files) == 500, f"{len(files)} parquet files")
    ref_schema = pq.read_schema(REF[ds]) if ds in REF else None
    schema_bad, shape_bad, nonfinite, sign_bad, label_bad, lcfs_bad, thom_bad = [], [], [], [], [], [], []
    psi_ax_err, psi_b_err = [], []
    sobol, psi_hash = [], set()
    npz = np.load(glob.glob(os.path.join(folder, f"{ds}_synthetic_*.npz"))[0])
    npz_idx = {int(i): k for k, i in enumerate(npz["sobol_index"])}
    npz_bad = 0
    for f in files:
        t = pq.read_table(f)
        row = t.to_pylist()[0]
        name = os.path.basename(f)
        if ref_schema is not None:
            for fld in ref_schema:
                if fld.name not in t.schema.names:
                    schema_bad.append(f"{name}: missing {fld.name}")
                elif t.schema.field(fld.name).type != fld.type:
                    schema_bad.append(f"{name}: {fld.name} {t.schema.field(fld.name).type} != {fld.type}")
        T = len(row["efit_times"])
        psi = np.asarray(row["efit_psirz"], float)
        if psi.shape != (T, 65, 65):
            shape_bad.append(f"{name}: psi {psi.shape}")
        for k in ["efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis", "efit_lcfs_n"]:
            if len(row[k]) != T:
                shape_bad.append(f"{name}: {k} len {len(row[k])} != T {T}")
        if len(row["magnetics_time"]) != len(row["magnetics_plasma_current"]) and ds != "d3d":
            shape_bad.append(f"{name}: magnetics length mismatch")
        if ds == "d3d" and len(row["magnetics_plasma_current"]) != len(row["magnetics_plasma_current_times"]):
            shape_bad.append(f"{name}: Ip/Ip-times length mismatch")
        for c in [c for c in row if c.startswith("magnetics_") and c.endswith("_current") or c in ("magnetics_bcoil", "magnetics_ECOILA") or c.startswith("magnetics_F")]:
            if c in ("magnetics_plasma_current",):
                continue
            if isinstance(row[c], list) and len(row[c]) != len(row["magnetics_time"]):
                shape_bad.append(f"{name}: {c} len {len(row[c])} != magnetics_time {len(row['magnetics_time'])}")
        # Thomson arrays vs chords
        names = np.asarray(row["thomson_chord_name"]).astype(str)
        n_core = int(np.char.startswith(names, "TS_core").sum())
        core = np.asarray(row["thomson_core_Te"], float)
        expect_core = n_core + (1 if ds == "mast" else 0)  # MAST keeps the empty channel 0
        if core.shape != (len(row["thomson_core_times"]), expect_core):
            shape_bad.append(f"{name}: core Te {core.shape} vs ({len(row['thomson_core_times'])},{expect_core})")
        if not np.all(np.isfinite(psi)):
            nonfinite.append(f"{name}: psi")
        for k in ["efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"]:
            if not np.all(np.isfinite(row[k])):
                nonfinite.append(f"{name}: {k}")
        cte = core[:, 1:] if ds == "mast" else core
        if not np.all(np.isfinite(cte)):
            nonfinite.append(f"{name}: thomson core Te")
        # physics per frame
        R = np.asarray(row["efit_grid_R"], float)
        Z = np.asarray(row["efit_grid_Z"], float)
        for j in range(T):
            sp = RectBivariateSpline(Z, R, psi[j])
            pa = sp(row["efit_z_axis"][j], row["efit_r_axis"][j])[0, 0]
            lr, lz = lcfs_of(row, j)
            pb = float(np.median(sp(lz, lr, grid=False)))
            pa_lab = row.get("efit_psi_axis", row.get("synth_psi_axis"))
            pb_lab = row.get("efit_psi_boundary", row.get("synth_psi_boundary"))
            pa_lab = pa_lab[j] if isinstance(pa_lab, list) else pa_lab
            pb_lab = pb_lab[j] if isinstance(pb_lab, list) else pb_lab
            span = abs(pa_lab - pb_lab)
            psi_ax_err.append(abs(pa - pa_lab) / span)
            psi_b_err.append(abs(pb - pb_lab) / span)
            if (pa > pb) != AXIS_MAX[ds]:
                sign_bad.append(f"{name}[{j}]")
            closed = np.hypot(lr[0] - lr[-1], lz[0] - lz[-1]) < 1e-6
            inside = Path(np.c_[lr, lz]).contains_point((row["efit_r_axis"][j], row["efit_z_axis"][j]))
            in_grid = lr.min() >= R[0] and lr.max() <= R[-1] and lz.min() >= Z[0] and lz.max() <= Z[-1]
            if not (closed and inside and in_grid and len(lr) >= 32):
                lcfs_bad.append(f"{name}[{j}] closed={closed} inside={inside} in_grid={in_grid} n={len(lr)}")
            for k, (lo, hi) in PLAUSIBLE.items():
                if not lo <= row[k][j] <= hi:
                    label_bad.append(f"{name}[{j}] {k}={row[k][j]:.3g}")
            # Thomson: chords inside the LCFS (by psiN) hotter than those well outside
            cR = np.asarray(row["thomson_chord_R"], float)
            cZ = np.asarray(row["thomson_chord_Z"], float)
            isc = np.char.startswith(names, "TS_core")
            x = (sp(cZ[isc], cR[isc], grid=False) - pa_lab) / (pb_lab - pa_lab)
            te = cte[j]
            if (x < 0.5).sum() >= 3 and (x > 1.2).sum() >= 3:
                if not np.median(te[x < 0.5]) > 3 * np.median(te[x > 1.2]):
                    thom_bad.append(f"{name}[{j}] core-inside {np.median(te[x < 0.5]):.0f} vs outside {np.median(te[x > 1.2]):.0f} eV")
            elif (x < 0.5).sum() >= 3 and not np.median(te[x < 0.5]) > 50:
                thom_bad.append(f"{name}[{j}] inside-plasma Te {np.median(te[x < 0.5]):.0f} eV")
        si = int(row["synth_sobol_index"])
        sobol.append(si)
        h = hash(np.round(psi, 8).tobytes())
        psi_hash.add(h)
        k = npz_idx.get(si)
        if k is None or not np.allclose(npz["psi"][k], psi[0], rtol=1e-5, atol=1e-6 * np.ptp(psi[0])):
            npz_bad += 1
    rep.check(ds, "schema matches challenge reference" if ds in REF else "schema (no reference machine)",
              not schema_bad, f"{len(schema_bad)} issues {schema_bad[:3]}")
    rep.check(ds, "array shapes consistent", not shape_bad, f"{len(shape_bad)} issues {shape_bad[:3]}")
    rep.check(ds, "no NaN/inf in psi, labels, Thomson", not nonfinite, f"{len(nonfinite)} issues {nonfinite[:3]}")
    rep.check(ds, "flux sign convention", not sign_bad, f"axis at {'max' if AXIS_MAX[ds] else 'min'}; {len(sign_bad)} frames wrong")
    rep.check(ds, "psi at axis / LCFS match labels",
              np.percentile(psi_ax_err, 99) < 0.03 and np.percentile(psi_b_err, 99) < 0.05,
              f"99th pct error: axis {np.percentile(psi_ax_err, 99):.4f}, boundary {np.percentile(psi_b_err, 99):.4f} (fraction of axis-boundary flux)")
    rep.check(ds, "LCFS closed, encloses axis, inside grid", not lcfs_bad, f"{len(lcfs_bad)} issues {lcfs_bad[:3]}")
    rep.check(ds, "labels plausible", not label_bad, f"{len(label_bad)} issues {label_bad[:3]}")
    rep.check(ds, "Thomson hot inside plasma, cold outside", not thom_bad, f"{len(thom_bad)} issues {thom_bad[:3]}")
    rep.check(ds, "Sobol indices unique, flux maps distinct",
              len(set(sobol)) == len(sobol) and len(psi_hash) == len(files),
              f"{len(set(sobol))} unique indices, {len(psi_hash)} distinct maps")
    rep.check(ds, "parquet and npz agree", npz_bad == 0, f"{npz_bad} files differ")


# ---------------------------------------------------------------------------------------------
def resolve_sample(ds, n, rep):
    """Re-solve from the parquet row alone; compare with the stored flux."""
    import synth_common as sc
    from freegsnke import GSstaticsolver, equilibrium_update
    from freegsnke.jtor_update import ConstrainBetapIp
    folder = os.path.join(GEN, f"{ds}_synthetic")
    files = sorted(glob.glob(os.path.join(folder, "parquet", "*.parquet")))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]
    if ds == "mast":
        import mast_machine as mm
        tok_dir, grid = os.path.join(GEN, "machines", "MAST_challenge"), (0.06, 2.0, -2.0, 2.0)
        build = lambda: mm.build_tokamak(tok_dir)  # noqa: E731
    elif ds == "d3d":
        import d3d_machine as dm
        from data_fixes import fix_d3d_ip_times
        tok_dir, grid = os.path.join(GEN, "machines", "DIIID_challenge"), (0.84, 2.54, -1.6, 1.6)
        build = lambda: dm.build_tokamak(tok_dir)  # noqa: E731
    else:
        import gen_machine as gmach
        from freegsnke import build_machine
        m = ds.upper()
        grid = gmach.MACHINES[m]["grid"]
        build = lambda: build_machine.tokamak(**gmach.cfg_paths(m))  # noqa: E731
    with sc.quiet():
        tok = build()
        eq = equilibrium_update.Equilibrium(tokamak=tok, Rmin=grid[0], Rmax=grid[1], Zmin=grid[2],
                                            Zmax=grid[3], nx=65, ny=65)
        gs = GSstaticsolver.NKGSsolver(eq)
    r2s, fails = [], []
    for f in pick:
        row = pq.read_table(f).to_pylist()[0]
        p = json.loads(row["synth_params"])
        t = row["efit_times"][0]
        tm = np.asarray(row["magnetics_time"], float)
        at = lambda c: float(np.interp(t, tm, np.asarray(row[c], float)))  # noqa: E731
        if ds == "mast":
            cur = mm.coil_currents_A({c: at(c) for c in mm.COIL_COLUMNS}, row["synth_sol_factor"])
            Ip = at("magnetics_plasma_current") * 1e3
            fvac = mm.fvac_from_tf(at("magnetics_tf_current"))
            betap, R0, a, k = row["synth_betap_target"], 0.9, 0.4, 1.6
            sign = 1.0
        elif ds == "d3d":
            cur = dm.coil_currents_A({c: at(c) for c in dm.COIL_COLUMNS})
            Ip = float(np.interp(t, fix_d3d_ip_times(row), np.asarray(row["magnetics_plasma_current"], float))) * 1e3
            fvac = dm.fvac_from_bcoil(at("magnetics_bcoil"))
            betap, R0, a, k = row["synth_betap_target"], 1.7, 0.4, 1.8
            sign = -1.0  # written in the DIII-D convention
            if Ip < 0:
                cur = {kk: -v for kk, v in cur.items()}
                Ip = -Ip
        else:
            cols = np.asarray(row["coil_input_column"])
            cur = {}
            for c in sorted(set(cols)):
                n_el = int((cols == c).sum())
                cur[c.replace("magnetics_", "").replace("_current", "")] = row[c][0] * 1e3 / n_el
            Ip = row["magnetics_plasma_current"][0] * 1e3
            fvac = sc.MU0 * row["magnetics_tf_current"][0] * 1e3 / (2 * np.pi)
            betap = p["betap"]
            R0, a = p["R0"], p["a"]
            k = p.get("kappa", 1.8)
            sign = 1.0
            a = 0.6 * a
        for c, v in cur.items():
            eq.tokamak.set_coil_current(c, v)
        eq.plasma_psi = sc.gaussian_psi_guess(eq, R0, 0.0, a, k, Ip)
        prof = ConstrainBetapIp(eq=eq, betap=betap, Ip=Ip, fvac=fvac, alpha_m=p["alpha_m"], alpha_n=p["alpha_n"])
        try:
            with sc.quiet():
                gs.solve(eq=eq, profiles=prof, constrain=None, target_relative_tolerance=1e-6,
                         max_solving_iterations=80)
            psi = sign * eq.psi().T
            stored = np.asarray(row["efit_psirz"][0], float)
            r2 = 1 - np.sum((stored - psi) ** 2) / np.sum((stored - stored.mean()) ** 2)
            r2s.append(r2)
            if r2 < 0.995:
                fails.append(f"{os.path.basename(f)} R2={r2:.4f}")
        except Exception as e:
            fails.append(f"{os.path.basename(f)} solve error {str(e)[:60]}")
    r2s = np.array(r2s)
    rep.check(ds, f"re-solve from parquet inputs ({len(pick)} files)",
              len(fails) <= max(1, len(pick) // 10) and (np.median(r2s) > 0.999 if len(r2s) else False),
              f"R2 vs stored psi: median {np.median(r2s) if len(r2s) else float('nan'):.5f}, min {r2s.min() if len(r2s) else float('nan'):.4f}; below 0.995: {fails[:4]}")


# ---------------------------------------------------------------------------------------------
def loader_check(rep):
    sys.path.insert(0, os.path.join(STARTER, "my_experiments", "c1"))
    from features import shot_inputs
    syn = sorted(glob.glob(os.path.join(GEN, "d3d_synthetic", "parquet", "*.parquet")))[::25]
    import gen_d3d as gd
    real = gd.holdout_files()[:20]
    def nan_share(paths):
        shares, errors = [], []
        for f in paths:
            try:
                out = shot_inputs(pq.read_table(f).to_pylist()[0])
                X = np.asarray(out["X"] if "X" in out else out["base"], float)
                shares.append(np.mean(~np.isfinite(X)))
            except Exception as e:
                errors.append(f"{os.path.basename(f)}: {type(e).__name__} {str(e)[:60]}")
        return shares, errors
    s_syn, e_syn = nan_share(syn)
    s_real, e_real = nan_share(real)
    rep.check("d3d", "C1 loader (features.shot_inputs) runs on synthetic rows", not e_syn,
              f"{len(syn) - len(e_syn)}/{len(syn)} ok; feature NaN share synthetic {np.mean(s_syn) if s_syn else float('nan'):.3f} vs real {np.mean(s_real) if s_real else float('nan'):.3f}; errors {e_syn[:2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["mast", "d3d", "sparc", "iter"])
    ap.add_argument("--resolve", type=int, default=25)
    ap.add_argument("--no-loader", action="store_true")
    a = ap.parse_args()
    rep = Report()
    for ds in a.sets:
        print(f"== {ds}")
        structure_and_physics(ds, rep)
        if a.resolve:
            resolve_sample(ds, a.resolve, rep)
    if "d3d" in a.sets and not a.no_loader:
        loader_check(rep)
    out = os.path.join(GEN, "audit_report.json")
    prev = json.load(open(out)) if os.path.exists(out) else []
    keep = [x for x in prev if x["dataset"] not in a.sets]
    json.dump(keep + rep.items, open(out, "w"), indent=1)
    n_fail = sum(not x["ok"] for x in rep.items)
    print(f"\n{len(rep.items) - n_fail} passed, {n_fail} failed -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
