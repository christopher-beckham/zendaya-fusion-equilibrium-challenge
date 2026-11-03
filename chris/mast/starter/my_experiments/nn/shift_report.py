"""Do MAST test inputs fall inside the DIII-D training range, feature by feature? CPU only.

Every feature is computed by one code path for both machines. The coil picture ψ_ext is built from
each file's own coil geometry on that machine's grid. Lengths are normalised by the plasma envelope
(fusion_scoring/masks/*_envelope.npz), not by the grid: R0 and Z0 are the envelope centroid, a is
half its radial extent. No EFIT target column is read.

For each feature, the reference range is DIII-D train frames between the 0.5th and 99.5th percentiles.
The report gives the share of DIII-D public-test frames and of MAST frames inside it. A feature
passes at 90% or more of MAST frames inside.

    python my_experiments/nn/shift_report.py --train-shots 200 --d3d-test-shots 200 --workers 3

Writes runs/shift_report/{features_<set>.npz, coverage.csv, report.md}.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    D3D_TF_TURNS, DATA_ROOT, N_SHAPE, REPO, ROBUST_IP_FLOOR_KA, _MU0, _align_channels, _at_times,
    _bilinear, _cell_py, _col, _frame_fields, _group_key, _interp_valid, _ip_denom, _moments_at_time,
    _multipole_row, _profile_system, _rect_profile, _shape_at_time, assign_splits, fix_d3d_ip_times,
    radial_group_key, shaping_ratios, shaping_sides, train_files,
)
from greens import filament_psi, rectangle_psi  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "shift_report"
MASKS = REPO / "fusion_scoring" / "masks"
GRID = 65
# MAST TF: 24 single-turn limbs. Not shipped in the data; an assumption to check against Bt ≈ 0.5 T.
MAST_TF_TURNS = 24
PASS_SHARE = 0.90
Q_LO, Q_HI = 0.5, 99.5
# Filaments per side when averaging a conductor rectangle. DIII-D rectangles are large, MAST's are single turns.
N_FIL = {"d3d": 5, "mast": 2}

FEATURES = (
    ["Ip_kA", "Bz_norm", "decay_index", "hexapole", "updown_flux",
     "null_R", "null_Z", "xpt_R", "xpt_Z",
     "Bt_T", "q_cyl", "V_loop", "dIp_dt", "V_loop_20ms", "dIp_dt_20ms",
     "updown_shaping_over_Ip", "total_shaping_over_Ip"]
    + [f"{name}{suffix}" for suffix in ("", "_win") for name in
       (["log10_p_peak"] + [f"p_shape_{i}" for i in range(1, N_SHAPE)]
        + ["th_R_peak", "th_Z_peak", "th_peaking", "th_width", "th_steep"])]
)
N_THOMSON = 1 + (N_SHAPE - 1) + 5


def machine_of(path: Path) -> str:
    return "mast" if "mast" in path.parent.name else "d3d"


def envelope(machine: str) -> dict:
    """Grid, plasma-region mask, and R0, Z0, a from the region's cells.

    DIII-D: the scorer's envelope. MAST: the region mast_region.py selects (FEC_MAST_MASK; default the envelope).
    """
    if machine == "mast":
        import mast_region
        reg = mast_region.region()
        r, z, inside = reg["r"], reg["z"], reg["mask"]
    else:
        data = np.load(MASKS / f"{machine}_envelope.npz")
        r = np.asarray(data["grid_R"], dtype=np.float64)
        z = np.asarray(data["grid_Z"], dtype=np.float64)
        inside = np.asarray(data["mask_coarse"], dtype=bool)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    r_in = rr[inside]
    return {
        "r": r, "z": z, "inside": inside,
        "R0": float(r_in.mean()), "Z0": float(zz[inside].mean()),
        "a": 0.5 * float(r_in.max() - r_in.min()),
    }


def coil_pictures(path: Path, machine: str, geo: dict) -> dict:
    """ψ_ext per current column, per unit of that column's shipped current, on the machine grid.

    kA·turn columns spread their current over their elements, so the picture is the element mean.
    A kA column whose elements are separate turns (MAST solenoid) carries it in each, so the picture
    is the element sum. DIII-D ECOILA is one rectangle at 1×, as in greens.py.
    """
    cols = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    t = pq.read_table(path, columns=cols)
    names = [str(c) for c in _cell_py(t, "coil_input_column")]
    rect = np.column_stack([np.asarray(_cell_py(t, c), dtype=np.float64) for c in cols[1:]])
    rr, zz = np.meshgrid(geo["r"], geo["z"], indexing="xy")
    R, Z = rr.ravel(), zz.ravel()
    out = {}
    for name in dict.fromkeys(names):
        rows = rect[[n == name for n in names]]
        acc = np.zeros(R.size, dtype=np.float64)
        for rc, zc, w, h in rows:
            acc += rectangle_psi(R, Z, rc, zc, w, h, n=N_FIL[machine])
        per_turn = name == "magnetics_sol_current"
        out[name] = acc if per_turn else acc / len(rows)
    return out


def _currents(path: Path, machine: str, columns: list[str], et: np.ndarray):
    """Currents at the EFIT times and at ±10 ms, Ip and its sign-of-flattop s, TF current."""
    tf_col = "magnetics_bcoil" if machine == "d3d" else "magnetics_tf_current"
    read = [*columns, tf_col, "magnetics_time", "magnetics_plasma_current"]
    if machine == "d3d":
        read.append("magnetics_plasma_current_times")
    t = pq.read_table(path, columns=read)
    clock = _col(t, "magnetics_time").astype(np.float64)
    ip_raw = _col(t, "magnetics_plasma_current").astype(np.float64)
    if machine == "d3d":
        ip_t = fix_d3d_ip_times({
            "magnetics_plasma_current_times": _col(t, "magnetics_plasma_current_times"),
            "magnetics_plasma_current": ip_raw, "magnetics_time": clock,
        })
    else:
        ip_t = clock
    stamps = {}
    for key, q in (("t", et), ("lo", et - 10.0), ("hi", et + 10.0)):
        cur = np.column_stack([_at_times(clock, _col(t, c).astype(np.float64), q) for c in columns])
        stamps[key] = (cur, _at_times(ip_t, ip_raw, q))
    ip = stamps["t"][1]
    peak = np.nanmax(np.abs(ip)) if ip.size else 0.0
    flat = np.abs(ip) >= 0.8 * peak
    s = float(np.sign(np.median(ip[flat]))) if peak > 0 and flat.any() else 1.0
    s = s if s in (-1.0, 1.0) else 1.0
    tf = _at_times(clock, _col(t, tf_col).astype(np.float64), et)
    return stamps, s, tf


def _thomson(path: Path, et: np.ndarray, geo: dict, window: tuple[float, float] | None) -> np.ndarray:
    """(T, N_THOMSON) on the widest radial midplane group, NaN where unusable.

    With `window`, only channels whose (R - R0)/a lies inside it are used.
    """
    cols = ["thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
            "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
            "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne"]
    t = pq.read_table(path, columns=cols)
    names = [str(n) for n in _cell_py(t, "thomson_chord_name")]
    cr = np.asarray(_cell_py(t, "thomson_chord_R"), dtype=np.float64)
    cz = np.asarray(_cell_py(t, "thomson_chord_Z"), dtype=np.float64)
    group = radial_group_key(names, cr, cz)
    chosen = np.array([_group_key(n) == group for n in names])
    system = _profile_system(group)
    te = _rect_profile(_cell_py(t, f"thomson_{system}_Te"))
    ne = _rect_profile(_cell_py(t, f"thomson_{system}_ne"))
    te, ne = _align_channels(te, ne, int(chosen.sum()), path.name)
    radial, vertical = cr[chosen], cz[chosen]
    if window is not None:
        rho = (radial - geo["R0"]) / geo["a"]
        keep = (rho >= window[0]) & (rho <= window[1])
        radial, vertical, te, ne = radial[keep], vertical[keep], te[:, keep], ne[:, keep]
    times = np.asarray(_cell_py(t, f"thomson_{system}_times"), dtype=np.float64)
    rows = np.full((len(te), N_THOMSON), np.nan)
    valid = np.zeros(len(te), dtype=bool)
    for k in range(len(te)):
        shape = _shape_at_time(radial, te[k], ne[k])
        mom = _moments_at_time(radial, vertical, te[k], ne[k], geo["R0"], geo["a"])
        if shape is None or mom is None:
            continue
        r_ratio, z_rel, peaking, width, steep = mom
        rows[k] = [shape[0], *shape[1][1:], (r_ratio * geo["R0"] - geo["R0"]) / geo["a"],
                   z_rel, peaking, width, steep]
        valid[k] = True
    if not valid.any():
        return np.full((len(et), N_THOMSON), np.nan)
    return _interp_valid(times, rows, valid, et)


def shot_features(job) -> dict:
    """(T, len(FEATURES)) for one shot."""
    path, pictures_file, window = Path(job[0]), job[1], job[2]
    machine = machine_of(path)
    geo = envelope(machine)
    pics = np.load(pictures_file)
    columns = list(pics.files)
    green = np.stack([pics[c] for c in columns])
    grid = pq.read_table(path, columns=["efit_grid_R", "efit_grid_Z", "efit_times"])
    if not (np.allclose(_cell_py(grid, "efit_grid_R"), geo["r"], atol=1e-5)
            and np.allclose(_cell_py(grid, "efit_grid_Z"), geo["z"], atol=1e-5)):
        raise RuntimeError(f"{path.name}: grid differs from the {machine} envelope grid")
    et = np.asarray(_cell_py(grid, "efit_times"), dtype=np.float64)
    r, z, R0, Z0, a = geo["r"], geo["z"], geo["R0"], geo["Z0"], geo["a"]
    stamps, s, tf = _currents(path, machine, columns, et)
    cur, ip = stamps["t"][0] * s, stamps["t"][1] * s
    flux = (cur @ green).reshape(-1, GRID, GRID)
    n = len(et)
    out = np.full((n, len(FEATURES)), np.nan)
    inside = geo["inside"]
    off_mid = inside & (np.abs(z - Z0)[:, None] > 0.25 * a)
    center = np.empty(n)
    for k in range(n):
        br, bz, dbz, d2 = _frame_fields(flux[k], r, z)
        row = _multipole_row(flux[k], br, bz, dbz, d2, r, z, R0, a, ip[k], ROBUST_IP_FLOOR_KA)
        bp = np.hypot(br, bz)
        iz, ir = np.unravel_index(np.argmin(np.where(inside, bp, np.inf)), bp.shape)
        jz, jr = np.unravel_index(np.argmin(np.where(off_mid, bp, np.inf)), bp.shape)
        out[k, 1:5] = row[:4]
        out[k, 5:9] = [(r[ir] - R0) / a, (z[iz] - Z0) / a, (r[jr] - R0) / a, (z[jz] - Z0) / a]
        center[k] = _bilinear(flux[k], r, z, R0, Z0)
    out[:, 0] = ip
    ip_a = _ip_denom(ip, ROBUST_IP_FLOOR_KA) * 1000.0
    turns = D3D_TF_TURNS if machine == "d3d" else MAST_TF_TURNS
    bt = _MU0 * turns * tf * 1000.0 / (2.0 * np.pi * R0)
    out[:, 9] = bt
    out[:, 10] = np.abs(2.0 * np.pi * a ** 2 * bt / (_MU0 * R0 * ip_a))
    t_s = et * 1e-3
    if n > 1:
        out[:, 11] = -np.gradient(center, t_s, edge_order=1)
        out[:, 12] = np.gradient(ip, t_s, edge_order=1)
    # The same rates over a fixed 20 ms on both machines, from the currents at t ± 10 ms.
    g0 = np.array([_bilinear(g.reshape(GRID, GRID), r, z, R0, Z0) for g in green])
    lo_cur, lo_ip = stamps["lo"][0] * s, stamps["lo"][1] * s
    hi_cur, hi_ip = stamps["hi"][0] * s, stamps["hi"][1] * s
    out[:, 13] = -((hi_cur - lo_cur) @ g0) / 0.02
    out[:, 14] = (hi_ip - lo_ip) / 0.02
    geom = pq.read_table(path, columns=["coil_input_column", "coil_Z"])
    upper, lower, _ = shaping_sides([str(c) for c in _cell_py(geom, "coil_input_column")],
                                    np.asarray(_cell_py(geom, "coil_Z"), dtype=np.float64))
    up = cur[:, [columns.index(c) for c in upper]]
    lo = cur[:, [columns.index(c) for c in lower]]
    out[:, 15], out[:, 16] = shaping_ratios(up, lo, ip, ROBUST_IP_FLOOR_KA)
    out[:, 17:17 + N_THOMSON] = _thomson(path, et, geo, None)
    out[:, 17 + N_THOMSON:] = _thomson(path, et, geo, window if machine == "mast" else None)
    return {"shot": path.stem, "features": out.astype(np.float32)}


def build_pictures(path: Path, out: Path) -> Path:
    """Write the machine's per-column pictures, with the sign baseline_pictures.npz uses."""
    machine = machine_of(path)
    geo = envelope(machine)
    pics = coil_pictures(path, machine, geo)
    # baseline_pictures.npz flipped the filament convention so ψ_ext correlates with DIII-D ψ.
    # The same flip applies to every machine: it is a convention, not a fit per machine.
    base = np.load(HERE / "baseline_pictures.npz")
    d3d_geo = envelope("d3d")
    ref = filament_psi(*[m.ravel() for m in np.meshgrid(d3d_geo["r"], d3d_geo["z"], indexing="xy")],
                       float(base["centroid_r"]), float(base["centroid_z"]))
    sign = float(np.sign(np.dot(ref, base["ring"].astype(np.float64))))
    np.savez(out, **{k: (v * sign).astype(np.float64) for k, v in pics.items()})
    return out


def check_d3d_pictures(pictures_file: Path) -> float:
    """Largest relative difference between these F-coil and ECOILA pictures and baseline_pictures.npz."""
    base = np.load(HERE / "baseline_pictures.npz")
    pics = np.load(pictures_file)
    from common import F_COILS
    mine = np.stack([pics[f"magnetics_{c}"] for c in F_COILS] + [pics["magnetics_ECOILA"]])
    ref = np.vstack([base["green"], base["ecoil"][None]]).astype(np.float64)
    return float(np.abs(mine - ref).max() / np.abs(ref).max())


def coverage(ref: np.ndarray, other: np.ndarray) -> dict:
    """Share of `other`'s finite values inside ref's [Q_LO, Q_HI] percentile range, plus shift."""
    ref = ref[np.isfinite(ref)]
    fin = np.isfinite(other)
    vals = other[fin]
    if ref.size == 0 or vals.size == 0:
        return {"inside": np.nan, "lo": np.nan, "hi": np.nan, "shift_sd": np.nan,
                "beyond_clip": np.nan, "missing": 1.0 - fin.mean() if other.size else np.nan}
    lo, hi = np.percentile(ref, [Q_LO, Q_HI])
    mu, sd = ref.mean(), ref.std()
    z = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
    return {
        "inside": float(((vals >= lo) & (vals <= hi)).mean()),
        "lo": float(lo), "hi": float(hi),
        "shift_sd": float((np.median(vals) - np.median(ref)) / sd) if sd > 0 else np.nan,
        "beyond_clip": float((np.abs(z) > 5.0).mean()),
        "missing": float(1.0 - fin.mean()),
    }


def run_set(label: str, paths: list[Path], pictures_file: Path, window, workers: int) -> np.ndarray:
    jobs = [(str(p), str(pictures_file), window) for p in paths]
    rows, shots, failed = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (job, res) in enumerate(zip(jobs, ex.map(_safe, jobs, chunksize=4))):
            if isinstance(res, str):
                failed.append((Path(job[0]).name, res))
            else:
                rows.append(res["features"])
                shots.append(np.full(len(res["features"]), res["shot"]))
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  {label}: {i + 1}/{len(jobs)} shots", flush=True)
    for name, err in failed[:10]:
        print(f"  {label}: {name} failed: {err}", flush=True)
    feats = np.concatenate(rows)
    np.savez_compressed(OUT / f"features_{label}.npz", features=feats, shot=np.concatenate(shots),
                        names=np.array(FEATURES), failed=np.array([f[0] for f in failed]))
    print(f"  {label}: {len(rows)} shots, {len(feats)} frames, {len(failed)} failed", flush=True)
    return feats


def _safe(job):
    try:
        return shot_features(job)
    except Exception as exc:  # noqa: BLE001 - one bad shot is reported, not fatal
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--train-shots", type=int, default=200)
    ap.add_argument("--d3d-test-shots", type=int, default=200)
    ap.add_argument("--mast-shots", type=int, default=0, help="0 uses every MAST test shot")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = train_files(args.data_root)
    train = [f for f, s in zip(files, assign_splits(len(files))) if s == "train"]
    train = [train[i] for i in sorted(rng.choice(len(train), min(args.train_shots, len(train)), replace=False))]
    d3d_test = sorted((args.data_root / "data" / "diii_d_public_test").glob("*.parquet"))
    d3d_test = [d3d_test[i] for i in sorted(rng.choice(len(d3d_test), min(args.d3d_test_shots, len(d3d_test)),
                                                      replace=False))]
    mast = sorted((args.data_root / "data" / "mast_public_test").glob("*.parquet"))
    if args.mast_shots:
        mast = mast[:args.mast_shots]

    d3d_pics = build_pictures(train[0], OUT / "pictures_d3d.npz")
    mast_pics = build_pictures(mast[0], OUT / "pictures_mast.npz")
    err = check_d3d_pictures(d3d_pics)
    print(f"DIII-D pictures vs baseline_pictures.npz: max relative difference {err:.2e}", flush=True)
    d3d_geo, mast_geo = envelope("d3d"), envelope("mast")
    for name, g in (("DIII-D", d3d_geo), ("MAST", mast_geo)):
        print(f"{name} envelope: R0 {g['R0']:.3f} m  Z0 {g['Z0']:.3f} m  a {g['a']:.3f} m", flush=True)

    # The DIII-D radial Thomson chord's span in (R - R0)/a, from the train sample's chord positions.
    spans = []
    for p in train[:50]:
        t = pq.read_table(p, columns=["thomson_chord_name", "thomson_chord_R", "thomson_chord_Z"])
        names = [str(n) for n in _cell_py(t, "thomson_chord_name")]
        cr = np.asarray(_cell_py(t, "thomson_chord_R"), dtype=np.float64)
        cz = np.asarray(_cell_py(t, "thomson_chord_Z"), dtype=np.float64)
        g = radial_group_key(names, cr, cz)
        rho = (cr[[_group_key(n) == g for n in names]] - d3d_geo["R0"]) / d3d_geo["a"]
        spans.append((rho.min(), rho.max()))
    window = (float(np.median([s[0] for s in spans])), float(np.median([s[1] for s in spans])))
    print(f"DIII-D radial Thomson span in (R - R0)/a: {window[0]:.3f} to {window[1]:.3f}", flush=True)

    ref = run_set("d3d_train", train, d3d_pics, window, args.workers)
    test = run_set("d3d_test", d3d_test, d3d_pics, window, args.workers)
    mst = run_set("mast_test", mast, mast_pics, window, args.workers)

    import csv
    lines = []
    with open(OUT / "coverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "d3d_lo", "d3d_hi", "d3d_test_inside", "mast_inside", "mast_shift_sd",
                    "mast_beyond_clip", "d3d_missing", "mast_missing", "verdict"])
        for j, name in enumerate(FEATURES):
            ct = coverage(ref[:, j], test[:, j])
            cm = coverage(ref[:, j], mst[:, j])
            miss_ref = float(1.0 - np.isfinite(ref[:, j]).mean())
            verdict = "pass" if cm["inside"] >= PASS_SHARE else "fail"
            w.writerow([name, cm["lo"], cm["hi"], ct["inside"], cm["inside"], cm["shift_sd"],
                        cm["beyond_clip"], miss_ref, cm["missing"], verdict])
            lines.append(f"| {name} | {cm['lo']:.3g} … {cm['hi']:.3g} | {ct['inside']:.1%} | {cm['inside']:.1%} | "
                         f"{cm['shift_sd']:+.2f} | {cm['beyond_clip']:.1%} | {cm['missing']:.1%} | {verdict} |")
    header = [
        "# MAST input coverage",
        "",
        f"DIII-D train: {len(ref)} frames. DIII-D public test: {len(test)} frames. MAST test: {len(mst)} frames.",
        f"Range: DIII-D train {Q_LO}–{Q_HI} percentiles. Pass: at least {PASS_SHARE:.0%} of MAST frames inside.",
        f"`_win` features restrict MAST Thomson to (R − R0)/a in {window[0]:.3f} … {window[1]:.3f}, "
        "the DIII-D chord's span.",
        "",
        "| feature | DIII-D range | DIII-D test inside | MAST inside | MAST median shift (DIII-D sd) | "
        "MAST beyond ±5 sd | MAST missing | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    (OUT / "report.md").write_text("\n".join(header + lines) + "\n")
    print("\n".join(header + lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
