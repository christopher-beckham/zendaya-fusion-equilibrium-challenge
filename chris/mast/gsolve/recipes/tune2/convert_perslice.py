#!/usr/bin/env python3
"""PER-SLICE NOISE COPY of /fsx/chris.beckham/gsolve_data/convert/convert_gsolve.py (the original is untouched).
Difference: every Thomson slice assigned to a solved frame gets its OWN independent noise draw on the frame's noise-free
remapped profile (gen_param.py outputs Te_clean / ne_clean / rel_te / rel_ne / inside), using noise_model.apply with the
recipe's knobs (--knobs). In the original, all slices within 6 ms of a frame carried one identical noisy profile, so the
model's time-interpolated Thomson inputs had single-slice noise, unlike real MAST (two independent slices). Seeds: one
per (file, rank, chunk, slice). Everything else is as in the original:

gsolve synthetic MAST equilibria (examples/generate_synthetic.py .npz) -> gen4-layout case files for synth_cache_seq.py.

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy \
        python convert_gsolve.py --npz pilot30.npz --out gen4_layout --heldout-thr 170 --workers 8

Case = one origin shot, one keep rank (rank 0 = the variant whose solved midplane edges were closest to the Thomson
edges at that frame, rank 1 = the next), up to MAX_FRAMES consecutive solved frames spanning at most MAX_SPAN ms.
Each case file follows Zara's gen4 release layout (gen4_set.write_case):
  * every non-efit column of the origin shot as stored (coil/chord geometry, Thomson radii), efit_grid_R/Z;
  * magnetics_* series cropped to centre +-100 ms, values as stored (solenoid column NOT halved: gsolve applied
    x0.5 internally, our pipeline applies x0.5 at feature time, Zara's files store the column x1 too);
  * Thomson slices (core and edge) at the real times in [first frame - 30 ms, last frame + 30 ms] and centre +-100 ms;
    a slice gets the remapped profile of the nearest solved frame of the case that has one, if that frame is within 6 ms, otherwise
    all channels NaN (gen4: SYN_MAX_DT = 6). Channels missing in the real slice stay missing; the core ghost channel 0
    is copied from the real slice. Te in eV, ne in m^-3 as in the challenge files;
  * labels at the solved frames: efit_psirz (frame, Z, R) as solved (MAST stored convention, axis at the maximum),
    efit_q95 / beta_n / li / r_axis / z_axis / elongation from gsolve.analyse (the npz numbers), efit_psi_axis,
    efit_psi_boundary, efit_geom_axis_r/z and efit_lcfs_* recomputed here with gsolve.diagnostics.analyse on the
    stored map (q95/betaN skipped there); efit_beta_p is not produced.
Cases without a synthetic core Thomson slice of >= 20 channels are dropped (gen4_set.has_thomson).\nSplit: held out if int(sha1(file name)[:8], 16) % 1000 < --heldout-thr (Zara's gen_m2 split uses < 100 for its
held-out pool, so thresholds <= 100 keep every held-out origin outside Zara's training pool).
Reads only input columns of the origin shots (no efit_* label column; the public-test files have none).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_model  # noqa: E402

ORIGIN = Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/mast_public_test")
WINDOW_MS, SYN_DT_CASE, SYN_MAX_DT = 100.0, 30.0, 6.0
MAX_FRAMES, MAX_SPAN = 9, 120.0
VERSION = "gsolve-gen4conv-1-perslice"
_G: dict = {}


def zara_hash(fname: str) -> int:
    return int(hashlib.sha1(fname.encode()).hexdigest()[:8], 16) % 1000


def load_npz(paths):
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    out = {}
    for k in keys:
        vs = [p[k] for p in parts]
        if vs[0].ndim == 2 and k in ("Te", "ne", "R_ts", "Te_clean", "ne_clean", "inside"):        # pad channel axis to the widest shard
            w = max(v.shape[1] for v in vs)
            vs = [np.pad(v.astype(np.float64), ((0, 0), (0, w - v.shape[1])), constant_values=(0 if k == "inside" else np.nan))
                  for v in vs]
        out[k] = np.concatenate(vs)
    return out


def _machine():
    if "m" not in _G:
        import torch
        torch.set_num_threads(1)
        import gsolve
        _G["m"] = gsolve.load_machine("mast")
    return _G["m"]


def shape_numbers(psi, R_ax, Z_ax):
    """psi_axis, psi_boundary, LCFS etc. from gsolve.diagnostics.analyse on a stored map (no profiles needed)."""
    import torch
    from gsolve.diagnostics import analyse

    class _Sol:
        pass
    s = _Sol()
    s.machine = _machine()
    s.psi = torch.as_tensor(psi[None].astype(np.float64))
    s.ip = np.ones(1)
    s.info = {"R_ax": np.array([R_ax]), "Z_ax": np.array([Z_ax])}
    return analyse(s, 0, 0.0)            # TF 0: F_vac = 0, q95 / betaN not evaluated (taken from the npz)


def read_origin(fname):
    import pyarrow.parquet as pq
    path = ORIGIN / fname
    names = [n for n in pq.read_schema(path).names if not n.startswith("efit_") or n in ("efit_grid_R", "efit_grid_Z", "efit_times")]
    t = pq.read_table(path, columns=names)
    return t


def case_jobs(z, rows, heldout_thr):
    """rows of one shot -> list of (rank, row indices of one case)."""
    frame, time = z["frame"][rows], z["time"][rows]
    order = np.lexsort((np.arange(len(rows)), frame))       # stable by frame, keep the generator's rank order
    rank = np.zeros(len(rows), int)
    seen = defaultdict(int)
    for i in order:
        rank[i] = seen[int(frame[i])]
        seen[int(frame[i])] += 1
    cases = []
    for r in sorted(set(rank.tolist())):
        sel = [rows[i] for i in sorted(np.nonzero(rank == r)[0], key=lambda i: time[i])]
        cur = []
        for j in sel:
            if cur and (len(cur) >= MAX_FRAMES or z["time"][j] - z["time"][cur[0]] > MAX_SPAN):
                cases.append((r, cur)); cur = []
            cur.append(j)
        if cur:
            cases.append((r, cur))
    return cases


def write_case(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    fname, rank, k, rows, out_path, split, src_npz = job
    Z = _G["z"]
    tab = read_origin(fname)
    row = {n: tab.column(n)[0].as_py() for n in tab.column_names}
    t_lab = Z["time"][rows].astype(np.float64)
    nf = len(rows)
    tc = float(t_lab[nf // 2])
    tm = np.asarray(row["magnetics_time"], np.float64)
    w = (tm >= tc - WINDOW_MS) & (tm <= tc + WINDOW_MS)
    if not (tm[w].min() <= t_lab.min() - 2.0 and tm[w].max() >= t_lab.max() + 2.0):
        return None
    ipraw = np.asarray(row["magnetics_plasma_current"], np.float64)
    cols = {}
    for n in tab.column_names:
        if n.startswith("efit_") and n not in ("efit_grid_R", "efit_grid_Z"):
            continue
        cols[n] = tab.column(n)
    cols["magnetics_time"] = pa.array([tm[w].tolist()], pa.list_(pa.float32()))
    for n in tab.column_names:
        if n.startswith("magnetics_") and n != "magnetics_time":
            cols[n] = pa.array([np.asarray(row[n], np.float64)[w].tolist()], pa.list_(pa.float64()))
    psi = Z["psi"][rows].astype(np.float64)
    sh = [shape_numbers(psi[q], float(Z["R_ax"][rows[q]]), float(Z["Z_ax"][rows[q]])) for q in range(nf)]
    if any(s["lcfs"] is None or not np.isfinite(s["psi_b"]) for s in sh):
        bad = [q for q, s in enumerate(sh) if s["lcfs"] is None or not np.isfinite(s["psi_b"])]
        keep = [q for q in range(nf) if q not in bad]
        if not keep:
            return None
        rows, sh, psi, t_lab = [rows[q] for q in keep], [sh[q] for q in keep], psi[keep], t_lab[keep]
        nf = len(rows)
    # channel layout of the npz rows (generate_synthetic.read_shot): core radii, then edge radii if the edge exists
    core_R = np.asarray(row["thomson_core_R"], np.float64)
    edge_R = np.asarray(row["thomson_edge_spatial"], np.float64) if row["thomson_edge_spatial"] else np.zeros(0)
    nc = len(core_R) if row["thomson_core_Te"] else 0
    edge_on = bool(row["thomson_edge_Te"]) and np.asarray(row["thomson_edge_Te"][0]).shape[0] == len(edge_R)
    R_ts = Z["R_ts"][rows[0]]
    exp_R = np.concatenate([core_R[:nc], edge_R if edge_on else np.zeros(0)])
    if not np.allclose(R_ts[:len(exp_R)], exp_R, atol=1e-4, equal_nan=True):
        raise RuntimeError(f"{fname}: npz Thomson radii do not match the file's channel layout")
    Te_k, ne_k = Z["Te"][rows].astype(np.float64), Z["ne"][rows].astype(np.float64)
    Tc_k, Nc_k = Z["Te_clean"][rows].astype(np.float64), Z["ne_clean"][rows].astype(np.float64)
    rT_k, rN_k, in_k = Z["rel_te"][rows], Z["rel_ne"][rows], Z["inside"][rows].astype(bool)
    KN = _G["knobs"]
    has_prof = np.isfinite(Te_k).any(1) & np.isfinite(ne_k).any(1)
    side_ch = {"core": (0, nc), "edge": (nc, nc + (len(edge_R) if edge_on else 0))}
    n_syn = {}
    for side in ("core", "edge"):
        tt = np.asarray(row[f"thomson_{side}_times"], np.float64) if row[f"thomson_{side}_times"] else np.zeros(0)
        m = (tt >= t_lab.min() - SYN_DT_CASE) & (tt <= t_lab.max() + SYN_DT_CASE) & (np.abs(tt - tc) <= WINDOW_MS)
        a, b = side_ch[side]
        te_rows, ne_rows, cnt = [], [], 0
        for j in np.nonzero(m)[0]:
            te_r = np.asarray(row[f"thomson_{side}_Te"][j], np.float64)
            ne_r = np.asarray(row[f"thomson_{side}_ne"][j], np.float64)
            off = len(te_r) - (b - a) if side == "core" else 0          # MAST core rows: ghost channel 0
            te_n, ne_n = np.full_like(te_r, np.nan), np.full_like(ne_r, np.nan)
            if off:
                te_n[:off], ne_n[:off] = te_r[:off], ne_r[:off]
            # nearest solved frame that has a remapped profile (54 / 1508 pilot rows have none)
            dt = np.where(has_prof, np.abs(t_lab - tt[j]), np.inf)
            i = int(np.argmin(dt))
            if b > a and dt[i] <= SYN_MAX_DT:
                real_nan = ~(np.isfinite(te_r[off:]) & np.isfinite(ne_r[off:]))
                if not real_nan.all():
                    seed = [zlib.crc32(fname.encode()), int(rank), int(k), {"core": 0, "edge": 1}[side], int(j)]
                    rng = np.random.default_rng(seed)
                    tT, tN = noise_model.apply(Tc_k[i], Nc_k[i], rT_k[i], rN_k[i], KN, in_k[i], rng)
                    tT = np.where(np.isfinite(Tc_k[i]), tT, np.nan); tN = np.where(np.isfinite(Nc_k[i]), tN, np.nan)
                    te_n[off:] = np.where(real_nan, np.nan, tT[a:b] * 1e3)
                    ne_n[off:] = np.where(real_nan, np.nan, tN[a:b] * 1e19)
                    cnt += 1
            te_rows.append(te_n.tolist()); ne_rows.append(ne_n.tolist())
        cols[f"thomson_{side}_times"] = pa.array([tt[m].tolist()], pa.list_(pa.float64()))
        cols[f"thomson_{side}_Te"] = pa.array([te_rows], pa.list_(pa.list_(pa.float64())))
        cols[f"thomson_{side}_ne"] = pa.array([ne_rows], pa.list_(pa.list_(pa.float64())))
        n_syn[side] = [cnt, int(m.sum())]
    core_te = cols["thomson_core_Te"][0].as_py()
    if not any(np.isfinite(np.asarray(x[1:] if nc and len(x) == nc + 1 else x, float)).sum() >= 20 for x in core_te):
        return None                      # gen4_set.has_thomson: a slice with >= 20 core channels (else features fail)
    f64 = lambda v: pa.array([[float(x) for x in v]], pa.list_(pa.float64()))  # noqa: E731
    cols["efit_times"] = pa.array([t_lab.tolist()], pa.list_(pa.float32()))
    cols["efit_psirz"] = pa.array([psi.tolist()], pa.list_(pa.list_(pa.list_(pa.float64()))))
    cols["efit_beta_n"] = f64(Z["betaN"][rows]); cols["efit_li"] = f64(Z["li"][rows]); cols["efit_q95"] = f64(Z["q95"][rows])
    cols["efit_r_axis"] = f64(Z["R_ax"][rows]); cols["efit_z_axis"] = f64(Z["Z_ax"][rows])
    cols["efit_elongation"] = f64(Z["kappa"][rows])
    cols["efit_psi_axis"] = f64([s["psi_ax"] for s in sh]); cols["efit_psi_boundary"] = f64([s["psi_b"] for s in sh])
    cols["efit_geom_axis_r"] = f64([s["R_geo"] for s in sh])
    cols["efit_geom_axis_z"] = f64([0.5 * (s["lcfs"][:, 1].max() + s["lcfs"][:, 1].min()) for s in sh])
    cols["efit_lcfs_n"] = f64([len(s["lcfs"]) for s in sh])
    cols["efit_lcfs_r"] = pa.array([[s["lcfs"][:, 0].tolist() for s in sh]], pa.list_(pa.list_(pa.float64())))
    cols["efit_lcfs_z"] = pa.array([[s["lcfs"][:, 1].tolist() for s in sh]], pa.list_(pa.list_(pa.float64())))
    cols["source"] = pa.array(["MAST_synthetic_gsolve_gen4_case"], pa.string())
    ftmask = np.abs(ipraw) >= 0.8 * np.nanmax(np.abs(ipraw))
    meta = dict(version=VERSION, recipe="gsolve generate_synthetic", source_npz=src_npz,
                case=dict(split=split, file=fname, t=tc, rank=int(rank), chunk=int(k), zara_hash=zara_hash(fname)),
                centre_t=tc, sol_factor=0.5, ip_sign=float(np.sign(np.nanmedian(ipraw[ftmask]))),
                thomson_slices={s: dict(synthetic=v[0], kept=v[1]) for s, v in n_syn.items()},
                rules=dict(window_ms=WINDOW_MS, syn_max_dt=SYN_MAX_DT, syn_dt_case=SYN_DT_CASE))
    cols["synth_meta"] = pa.array([json.dumps(meta)], pa.string())
    gen = [k_ for k_ in Z if k_.startswith("gen_")] + ["variant", "phase", "z_balance", "delta_upper", "delta_lower",
                                                       "volume", "xpt_R", "xpt_Z", "frame"]
    diag = []
    for q, r in enumerate(rows):
        d = {g: (float(Z[g][r]) if Z[g][r].dtype.kind == "f" else int(Z[g][r])) for g in gen if g in Z}
        d.update(t=float(t_lab[q]), status="ok", diverted=bool(sh[q]["diverted"]), a=float(sh[q]["a"]),
                 psi_ax_check=float(sh[q]["psi_ax"]), R_ax_check=float(sh[q]["R_ax"]), Z_ax_check=float(sh[q]["Z_ax"]),
                 kappa_check=float(sh[q]["kappa"]))
        diag.append(d)
    cols["synth_frames"] = pa.array([json.dumps(diag)], pa.string())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return dict(path=str(out_path), split=split, n=nf, ip_sign=meta["ip_sign"],
                ax_err=[float(np.hypot(s["R_ax"] - Z["R_ax"][r], s["Z_ax"] - Z["Z_ax"][r])) for s, r in zip(sh, rows)],
                kappa_err=[float(s["kappa"] - Z["kappa"][r]) for s, r in zip(sh, rows)])


def _init(npz_paths, knobs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G["z"] = load_npz(npz_paths)
    _G["knobs"] = knobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout-thr", type=int, required=True, help="held out if sha1(file)[:8] %% 1000 < thr")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--knobs", required=True, help="the recipe's knob JSON (noise settings)")
    a = ap.parse_args()
    z = load_npz(a.npz)
    shots = np.array([str(s) for s in z["shot"]])
    jobs = []
    for s in sorted(set(shots.tolist())):
        fname = f"mast_shot_{s}.parquet"
        split = "heldout" if zara_hash(fname) < a.heldout_thr else "train"
        rows = np.nonzero(shots == s)[0].tolist()
        for k, (rank, cr) in enumerate(case_jobs(z, rows, a.heldout_thr)):
            out = a.out / split / f"mast_synth_gs_{s}_r{rank}_c{k:03d}.parquet"
            jobs.append((fname, rank, k, cr, str(out), split, ",".join(map(str, a.npz))))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(z['shot'])} equilibria, {len(set(shots.tolist()))} shots -> {len(jobs)} cases", flush=True)
    res = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.npz, json.loads(Path(a.knobs).read_text()))) as ex:
        for i, r in enumerate(ex.map(write_case, jobs, chunksize=4), 1):
            res.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    ok = [r for r in res if r]
    summ = {sp: dict(cases=sum(r["split"] == sp for r in ok), frames=sum(r["n"] for r in ok if r["split"] == sp),
                     shots=len({Path(r["path"]).name.split("_")[3] for r in ok if r["split"] == sp})) for sp in ("train", "heldout")}
    ax = np.concatenate([r["ax_err"] for r in ok]); ke = np.concatenate([r["kappa_err"] for r in ok])
    summ.update(skipped=len(res) - len(ok), ip_signs=sorted({r["ip_sign"] for r in ok}),
                axis_err_m=dict(median=float(np.median(ax)), p99=float(np.percentile(ax, 99)), max=float(ax.max())),
                kappa_err=dict(median=float(np.median(np.abs(ke))), max=float(np.abs(ke).max())))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "convert_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
