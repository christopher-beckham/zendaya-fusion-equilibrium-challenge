"""One pass over every shot -> per-shot cached summaries (resumable).

Usage:  uv run python my_experiments/data_stats.py [--root hf_dataset] [--splits diii_d_train diii_d_public_test]
Outputs (results/data_stats/):
  cache/<split>/<shot>.pkl   per-shot result (skipped on rerun)
  frames_<split>.parquet     one row per EFIT frame: 21 coils interpolated to efit_times (+ Ip phase)
  raw_<split>.parquet        one row per (shot, channel): raw-signal n / NaN / inf / zero / min / max
  shots_<split>.parquet      one row per shot: lengths, time ranges, Ip fix, psi + scalar NaN checks
"""
import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_fixes import fix_d3d_ip_times  # noqa: E402

SIGS = [
    "ECOILA", "bcoil",
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B",
    "F6A", "F6B", "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
    "plasma_current",
]
SCALARS = ["efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"]
OUT = REPO / "results" / "data_stats"


def process(path: Path, split: str) -> dict:
    schema = pq.read_schema(path).names
    cols = ["efit_times", "magnetics_time", "magnetics_plasma_current_times"]
    cols += [f"magnetics_{s}" for s in SIGS]
    cols += [c for c in SCALARS + ["efit_psirz"] if c in schema]
    tbl = pq.read_table(path, columns=cols)
    col = lambda c: tbl.column(c).combine_chunks()[0].values.to_numpy(zero_copy_only=False)  # noqa: E731

    shot = path.stem
    et = col("efit_times").astype(np.float64)
    mt = col("magnetics_time").astype(np.float64)
    row = {c: col(c) for c in ["magnetics_plasma_current_times", "magnetics_plasma_current", "magnetics_time"]}
    ip_t_raw = row["magnetics_plasma_current_times"].astype(np.float64)
    ip_t = fix_d3d_ip_times(row)

    frames = {"shot": np.full(len(et), shot), "t": et}
    raw_rows = []
    for s in SIGS:
        v = col(f"magnetics_{s}").astype(np.float64)
        t = ip_t if s == "plasma_current" else mt
        n = min(len(t), len(v))
        t, v = t[:n], v[:n]
        fin = np.isfinite(v)
        raw_rows.append({
            "shot": shot, "channel": s, "n": n, "n_nan": int(np.isnan(v).sum()),
            "n_inf": int(np.isinf(v).sum()), "n_zero": int((v == 0).sum()),
            "min": v[fin].min() if fin.any() else np.nan, "max": v[fin].max() if fin.any() else np.nan,
            "t0": t[0] if n else np.nan, "t1": t[-1] if n else np.nan,
            "dt_ms": np.median(np.diff(t)) if n > 1 else np.nan,
        })
        # linear interp like the baseline (interp1d extrapolate); np.interp clamps, so flag out-of-range
        frames[s] = np.interp(et, t[fin], v[fin]) if fin.sum() > 1 else np.full(len(et), np.nan)
        if s == "plasma_current":
            frames["out_of_range_ip"] = (et < t[0]) | (et > t[-1])
        elif s == "F1A":
            frames["out_of_range_mag"] = (et < t[0]) | (et > t[-1])

    ip = frames["plasma_current"]
    peak = np.nanmax(np.abs(ip)) if np.isfinite(ip).any() else np.nan
    flat = np.abs(ip) >= 0.8 * peak
    idx = np.arange(len(et))
    first, last = (idx[flat].min(), idx[flat].max()) if flat.any() else (len(et), -1)
    frames["phase"] = np.where(idx < first, "rampup", np.where(idx > last, "rampdown", "flattop"))

    meta = {
        "shot": shot, "split": split, "T": len(et),
        "efit_t0": et[0] if len(et) else np.nan, "efit_t1": et[-1] if len(et) else np.nan,
        "efit_dt_ms": np.median(np.diff(et)) if len(et) > 1 else np.nan,
        "efit_times_nonmonotonic": bool(np.any(np.diff(et) <= 0)),
        "mag_n": len(mt), "mag_dt_ms": np.median(np.diff(mt)) if len(mt) > 1 else np.nan,
        "ip_axis_fixed": not np.array_equal(ip_t, ip_t_raw),
        "ip_peak_abs": peak, "ip_flattop_median": np.nanmedian(ip[flat]) if flat.any() else np.nan,
        "bcoil_median": np.nanmedian(frames["bcoil"]),
        "n_flattop": int(flat.sum()),
    }
    if "efit_psirz" in schema:
        psi = tbl.column("efit_psirz").combine_chunks()[0].values
        while hasattr(psi, "flatten"):  # (T, 65, 65) nested lists -> flat values
            psi = psi.flatten()
        psi = psi.to_numpy(zero_copy_only=False)
        psi = psi.astype(np.float64).reshape(len(et), -1) if psi.size == len(et) * 4225 else psi
        meta.update({
            "psi_ok_shape": psi.ndim == 2, "psi_n_nan": int(np.isnan(psi).sum()),
            "psi_n_inf": int(np.isinf(psi).sum()), "psi_frames_all_zero": int(np.all(psi == 0, axis=-1).sum()) if psi.ndim == 2 else -1,
            "psi_min": np.nanmin(psi), "psi_max": np.nanmax(psi),
        })
        for s in SCALARS:
            if s in schema:
                a = col(s).astype(np.float64)
                meta[f"{s}_n_nan"] = int(np.isnan(a).sum())
                meta[f"{s}_len_mismatch"] = len(a) != len(et)
                frames[s] = a[: len(et)] if len(a) >= len(et) else np.concatenate([a, np.full(len(et) - len(a), np.nan)])
    return {"frames": pd.DataFrame(frames), "raw": pd.DataFrame(raw_rows), "meta": meta}


def run_one(args):
    path, split = args
    out = OUT / "cache" / split / f"{path.stem}.pkl"
    if out.exists():
        return
    try:
        res = process(path, split)
    except Exception as e:  # keep going; record the failure
        res = {"frames": pd.DataFrame(), "raw": pd.DataFrame(),
               "meta": {"shot": path.stem, "split": split, "error": repr(e)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "hf_dataset"))
    ap.add_argument("--splits", nargs="+", default=["diii_d_train", "diii_d_public_test"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for split in a.splits:
        files = sorted((Path(a.root) / "data" / split).glob("*.parquet"))
        jobs = [(f, split) for f in files]
        print(f"{split}: {len(files)} files", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(as_completed([ex.submit(run_one, j) for j in jobs])):
                if (i + 1) % 250 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
        res = []
        for f in files:
            p = OUT / "cache" / split / f"{f.stem}.pkl"
            if p.exists():
                with open(p, "rb") as fh:
                    res.append(pickle.load(fh))
        pd.concat([r["frames"] for r in res], ignore_index=True).to_parquet(OUT / f"frames_{split}.parquet")
        pd.concat([r["raw"] for r in res], ignore_index=True).to_parquet(OUT / f"raw_{split}.parquet")
        pd.DataFrame([r["meta"] for r in res]).to_parquet(OUT / f"shots_{split}.parquet")
        print(f"  wrote {len(res)} shots for {split}", flush=True)


if __name__ == "__main__":
    main()
