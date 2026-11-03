"""Turn a synthetic MAST sequence set (gen4 layout: several labelled EFIT times per case) into a training cache.

usage: CUDA_VISIBLE_DEVICES="" python synth_cache_seq.py --src DIR --tag NAME [--heldout-src DIR2] [--workers 12] [--mirror]
       FEC_MAST_MASK=limiter ... --tag NAME_limiter     (features with the limiter region's Z0; mast_region.py)

Each source file is one case: 3–9 labelled frames at consecutive EFIT times, ±100 ms of input signals, and a synthetic
Thomson profile at every real Thomson time near the labelled frames (README of the set). Every labelled frame becomes
one example in <cache>/synth/<tag>/arrays.npz, in the same layout as synth_cache.py, so synth_data.SynthSet reads it
unchanged:
  feats (N, 9, F)   features_v5 at the frame's time ± k·5 ms (k = −4…4; the middle is the labelled frame)
  cur, ip           decoder currents and Ip (kA) on the same window
  psi (N, 4225)     label flux × −1 (MAST stores the axis at the maximum; our maps put it at the minimum)
  scal (N, 5)       q95, betaN, li, R_axis, Z_axis
  te, ne, tr, tz    raw Thomson at the labelled frame
  stem, origin, split, mid, case, frame
Splits come from the source folders: files under --src are "train", files under --heldout-src are "heldout" (the set
draws its held-out cases from a separate pool of origin shots). Unlike synth_cache.py, nothing is held or expanded
beyond adding the window times: Thomson stays at its own times.

--mirror builds the up-down mirror image of every case (MAST is up-down symmetric: grid, limiter, coil pairs P2–P6 within
~5 mm, Thomson at Z = 0): each upper coil gets its lower partner's current and vice versa, the label flux is flipped
top to bottom and Z_axis changes sign. Everything else (Ip, solenoid, TF, Thomson) is unchanged.

Reads only the synthetic files and, for dIp/dt and loop-voltage normalisation, the origin shot's Ip input trace; no
efit_* column of any real file is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as nc  # noqa: E402
import features_v2 as fv  # noqa: E402
import features_v5 as f5  # noqa: E402
from build_thomson_raw import shot as thomson_shot  # noqa: E402
from synth_cache import PICS, REAL, ORIGIN_DIR, _sha, t50_ipmax  # noqa: E402

CACHE = nc.CACHE
# MAST EFIT spacing (ms) and window half-width (frames); FEC_SYNTH_HALF widens the window for longer history inputs
# (train_mm --history-frames K needs HALF >= 4K on MAST, synth_data.HISTORY_STEP).
STEP, HALF = 5.0, int(__import__("os").environ.get("FEC_SYNTH_HALF", "4"))
_P: dict = {}
MIRROR = {"flag": False}
PAIRS = [(f"magnetics_p{k}u_current", f"magnetics_p{k}l_current") for k in (2, 3, 4, 5, 6)]


def _mirror_table(t: pa.Table) -> pa.Table:
    """Swap the upper and lower coil currents (one value column per coil circuit)."""
    cols = {n: t.column(n) for n in t.column_names}
    for u, l in PAIRS:
        iu, il = t.schema.get_field_index(u), t.schema.get_field_index(l)
        t = t.set_column(iu, t.schema.field(iu), cols[l]).set_column(il, t.schema.field(il), cols[u])
    return t


def expanded_copy(path: Path, out_dir: Path, split: str) -> tuple[Path, list[float], list[list[int]]]:
    """Copy of `path` whose efit_times are every labelled time ± k·STEP (k ≤ HALF), deduplicated and sorted.
    Returns the copy, the labelled times, and for each labelled frame the indices of its 2·HALF+1 window times."""
    t = pq.read_table(path)
    lab = [float(v) for v in t.column("efit_times")[0].as_py()]
    grid = sorted({round(tl + k * STEP, 3) for tl in lab for k in range(-HALF, HALF + 1)})
    pos = {v: i for i, v in enumerate(grid)}
    windows = [[pos[round(tl + k * STEP, 3)] for k in range(-HALF, HALF + 1)] for tl in lab]
    out = out_dir / f"{split}_{'mirror_' if MIRROR['flag'] else ''}{path.name}"   # train and held-out reuse file names
    if not out.exists():
        i = t.schema.get_field_index("efit_times")
        t = t.set_column(i, t.schema.field(i), pa.array([grid], type=t.schema.field(i).type))
        if MIRROR["flag"]:
            t = _mirror_table(t)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        pq.write_table(t, tmp)
        tmp.replace(out)
    return out, lab, windows


def one(job):
    path, exp_dir, split = Path(job[0]), Path(job[1]), job[2]
    if "mast" not in _P:
        _P["mast"] = fv.load_pictures(CACHE / PICS["mast"])
    pics = _P["mast"]
    exp, lab, windows = expanded_copy(path, exp_dir, split)
    row = pq.read_table(path, columns=["efit_psirz", *nc.SCALAR_COLS, "synth_meta"]).to_pylist()[0]
    meta = json.loads(row["synth_meta"])
    origin = str(meta["case"]["file"]).replace(".parquet", "")
    psi = np.asarray(row["efit_psirz"], np.float32).reshape(len(lab), -1)
    if psi.shape[1] != nc.N_PSI:
        raise ValueError(f"{path.name}: flux has {psi.shape[1]} values per frame, expected {nc.N_PSI}")
    scal = np.stack([np.asarray(row[c], np.float64).ravel()[:len(lab)] for c in nc.SCALAR_COLS], 1).astype(np.float32)
    if MIRROR["flag"]:                                  # grid rows are Z, symmetric about 0 (checked in main)
        n = int(round(np.sqrt(psi.shape[1])))
        psi = np.ascontiguousarray(psi.reshape(len(lab), n, n)[:, ::-1, :].reshape(len(lab), -1))
        scal[:, nc.SCALAR_COLS.index("efit_z_axis")] *= -1
    base, extra = fv.shot_features(exp, pics, return_inputs=True, thomson_window=f5.THOMSON_WINDOW)
    feats = f5.assemble(base, extra, pics)
    t50_w, ipmax_w = t50_ipmax(*fv._pulse(exp, "mast"))
    t50_o, ipmax_o = t50_ipmax(*fv._pulse(REAL / ORIGIN_DIR["mast"] / f"{origin}.parquet", "mast"))
    ipmax_o *= float(meta.get("jitter", {}).get("ip", 1.0))
    jd, jv = f5.FEATURES.index("dIp_dt_norm"), f5.FEATURES.index("V_loop_norm")
    feats[:, jd] *= np.float32((t50_o / t50_w) * (ipmax_w / ipmax_o))
    feats[:, jv] *= np.float32(t50_o / t50_w)
    cur, ip = np.asarray(extra["cur"], np.float32), np.asarray(extra["ip"], np.float32)
    th = {}
    for q in ("Te", "ne"):
        v, r, z = thomson_shot((str(exp), None, q))
        th[q] = v
    out = []
    for j, w in enumerate(windows):
        mid = w[HALF]
        mtag = "mirror_" if MIRROR["flag"] else ""
        out.append({"stem": f"{split}_{mtag}{path.stem}_f{j}", "origin": origin, "split": split,
                    "case": f"{split}_{mtag}{path.stem}",
                    "frame": j,
                    "feats": feats[w], "cur": cur[w], "ip": ip[w], "psi": -psi[j], "scal": scal[j],
                    "te": th["Te"][mid], "ne": th["ne"][mid], "tr": r[mid], "tz": z[mid]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="training cases (read only)")
    ap.add_argument("--heldout-src", type=Path, default=None, help="held-out cases (read only)")
    ap.add_argument("--tag", required=True, help="output name: <cache>/synth/<tag>/")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="per folder, for smoke tests")
    ap.add_argument("--mirror", action="store_true", help="build the up-down mirror image of every case")
    a = ap.parse_args()
    MIRROR["flag"] = a.mirror
    import mast_region
    if mast_region.suffix() and not a.tag.endswith(mast_region.suffix()):
        raise SystemExit(f"{mast_region.ENV_VAR}={mast_region.name()}: the tag must end in {mast_region.suffix()} "
                         f"(MAST features depend on the region; envelope caches keep their names)")
    if not mast_region.suffix() and a.tag.endswith("_limiter"):
        raise SystemExit("a *_limiter tag needs FEC_MAST_MASK=limiter")
    jobs = []
    out = CACHE / "synth" / a.tag
    exp_dir = out / "expanded_mast"            # features_v2 reads the machine from the folder name
    exp_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, src in (("train", a.src), ("heldout", a.heldout_src)):
        if src is None:
            continue
        fs = sorted(Path(src).glob("mast_synth*.parquet"))
        if a.limit:
            fs = fs[:a.limit]
        files.update({f: split for f in fs})
        jobs += [(str(f), str(exp_dir), split) for f in fs]
    if not jobs:
        raise SystemExit("no parquet files found")
    geo = fv.geometry("mast")
    g = pq.read_table(next(iter(files)), columns=["efit_grid_R", "efit_grid_Z"]).to_pylist()[0]
    if not (np.allclose(np.ravel(g["efit_grid_R"]), geo["r"], atol=1e-4)
            and np.allclose(np.ravel(g["efit_grid_Z"]), geo["z"], atol=1e-4)):
        raise SystemExit("synthetic grid does not match features_v2.geometry; resample before use")
    if a.mirror and not (np.allclose(geo["z"], -geo["z"][::-1], atol=1e-6) and np.array_equal(
            *(lambda m: (m, m[::-1]))(__import__("mast_region").mask()))):
        raise SystemExit("--mirror needs a grid and MAST region symmetric about Z = 0")
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2), 1):
            res += r
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} cases", flush=True)
    width = max(len(r["te"]) for r in res)
    pad = lambda k: np.stack([np.r_[np.asarray(r[k], np.float32), np.full(width - len(r[k]), np.nan, np.float32)]  # noqa: E731
                              for r in res])
    arrays = {k: np.stack([r[k] for r in res]) for k in ("feats", "cur", "ip", "psi", "scal")}
    split = np.array([r["split"] for r in res])
    np.savez(out / "arrays.npz", stem=np.array([r["stem"] for r in res]), origin=np.array([r["origin"] for r in res]),
             case=np.array([r["case"] for r in res]), frame=np.array([r["frame"] for r in res]), split=split,
             mid=np.int64(HALF), te=pad("te"), ne=pad("ne"), tr=pad("tr"), tz=pad("tz"), **arrays)
    manifest = {
        "machine": "mast", "layout": "sequence (gen4)", "source": str(a.src), "heldout_source": str(a.heldout_src),
        "n_cases": len(jobs), "n_frames": len(res), "label_sign": -1.0, "pictures": PICS["mast"],
        "mast_mask": mast_region.name(), "mirror": bool(a.mirror),
        "window": {"spacing_ms": STEP, "half_frames": HALF},
        "split_rule": "train = --src folder, heldout = --heldout-src folder",
        "split_counts": {s: int((split == s).sum()) for s in ("train", "heldout")},
        "arrays_sha256": _sha(out / "arrays.npz"),
        "sources_sha256": {f"{s}/{f.name}": _sha(f) for f, s in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    f = arrays["feats"][:, HALF]
    print(f"wrote {out}: {len(jobs)} cases, {len(res)} frames ({manifest['split_counts']}), feats {arrays['feats'].shape}, "
          f"thomson_missing {np.nanmean(f[:, f5.FEATURES.index('thomson_missing')]):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
