"""Turn a folder of synthetic equilibria (FreeGSNKE, challenge-format parquet) into a labelled training cache. CPU only.

usage: CUDA_VISIBLE_DEVICES="" python synth_cache.py --machine mast|d3d --src DIR --tag NAME [--workers 5] [--limit N]

DIR holds <machine>_synth_XXXXX.parquet files with exactly the challenge's columns, one labelled EFIT time each and
±100 ms of input signals around it, plus synth_* columns (generator parameters, origin shot and time). The source
folder is only read. Output: <cache>/synth/<tag>/ (expanded copies in its expanded_<machine>/ subfolder)
  arrays.npz      per case (N cases, W = 2n+1 window frames, middle = the labelled time):
                    feats (N, W, F) features_v5; MAST with pictures_mast_sol0.5 (solenoid ×0.5, as our MAST inputs)
                    cur (N, W, C), ip (N, W)  mirrored decoder currents and Ip (kA) from features_v2
                    stamp (N, W, 21)          DIII-D only: mirrored coil currents as in coils.npy (decoder picture)
                    psi (N, 4225)             label flux in our convention: DIII-D as stored; MAST × −1 (MAST stores
                                              the axis at the maximum, our maps and DIII-D at the minimum)
                    scal (N, 5)               q95, betaN, li, R_axis, Z_axis (common.SCALAR_NAMES order)
                    te, ne, tr, tz (N, K)     raw Thomson at the labelled time (NaN padded)
                    stem, origin, split, mid  file stem, origin shot, "train" / "heldout", middle index
  manifest.json   source folder, file count, SHA256 of every source file and of arrays.npz, split rule, settings

Features need a time window: the file's single EFIT time is expanded to t + k·spacing (k = −n…n at the machine's
EFIT spacing; MAST 5 ms, n = 4; DIII-D 20 ms, n = 1) on a copy, and the single Thomson profile is held at t ± 24 ms.
dIp_dt_norm and V_loop_norm are rescaled to the origin shot's t50 and Ip_max (Ip_max × the case's Ip scale), read
from the origin shot's Ip input trace. Splits are by origin shot (shot-level): MAST origin shots are hashed into
`--heldout-share`; DIII-D cases whose origin shot is in our val or test split are all held out, so DIII-D validation
inputs never reach training.

Reads input columns of real files (origin shots: Ip trace only) and the synthetic files' labels; no efit_* label of
any real file is read. Synthetic Thomson Te runs hot (README); it is stored raw, uncalibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as nc  # noqa: E402
import features_v2 as fv  # noqa: E402
import features_v5 as f5  # noqa: E402
from build_thomson_raw import shot as thomson_shot  # noqa: E402

CACHE = nc.CACHE
REAL = nc.DATA_ROOT / "data"
SPACING = {"mast": (5.0, 4), "d3d": (20.0, 1)}          # EFIT spacing (ms), half-width in frames
THOMSON_HOLD = 24.0                                       # ms
PICS = {"mast": "features_v2/pictures_mast_sol0.5.npz", "d3d": "features_v2/pictures_d3d.npz"}
ORIGIN_DIR = {"mast": "mast_public_test", "d3d": "diii_d_train"}
LABEL_SIGN = {"mast": -1.0, "d3d": 1.0}
_P: dict = {}


def expanded_copy(path: Path, machine: str, out_dir: Path) -> tuple[Path, int]:
    """Copy of `path` with efit_times t + k·spacing and the Thomson profile held at t ± THOMSON_HOLD."""
    step, n = SPACING[machine]
    out = out_dir / path.name
    if out.exists():
        return out, n
    t = pq.read_table(path)
    t0 = float(t.column("efit_times")[0].as_py()[0])
    i = t.schema.get_field_index("efit_times")
    t = t.set_column(i, t.schema.field(i), pa.array([[t0 + k * step for k in range(-n, n + 1)]],
                                                    type=t.schema.field(i).type))
    for s in ("core", "edge"):
        tc = f"thomson_{s}_times"
        if tc not in t.column_names or len(t.column(tc)[0].as_py()) != 1:
            continue
        tt = t.column(tc)[0].as_py()[0]
        j = t.schema.get_field_index(tc)
        t = t.set_column(j, t.schema.field(j), pa.array([[tt - THOMSON_HOLD, tt + THOMSON_HOLD]],
                                                        type=t.schema.field(j).type))
        for q in ("Te", "ne"):
            c = f"thomson_{s}_{q}"
            rows = t.column(c)[0].as_py()
            j = t.schema.get_field_index(c)
            t = t.set_column(j, t.schema.field(j), pa.array([[rows[0], rows[0]]], type=t.schema.field(j).type))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    pq.write_table(t, tmp)
    tmp.replace(out)
    return out, n


def t50_ipmax(ip_t: np.ndarray, ip: np.ndarray) -> tuple[float, float]:
    """As features_v2.shot_features: t50 (s, |Ip| above half its peak) and Ip_max (kA)."""
    ip_max = float(np.max(np.abs(ip))) if ip.size else np.nan
    above = ip_t[np.abs(ip) > 0.5 * ip_max]
    return ((above.max() - above.min()) * 1e-3 if above.size > 1 else np.nan), ip_max


def _labels(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    cols = ["efit_psirz", *nc.SCALAR_COLS, "synth_params", "synth_origin_shot"]
    row = pq.read_table(path, columns=cols).to_pylist()[0]
    psi = np.asarray(row["efit_psirz"], np.float32).reshape(-1)
    if psi.size != nc.N_PSI:
        raise ValueError(f"{path.name}: flux has {psi.size} values, expected {nc.N_PSI}")
    scal = np.array([np.asarray(row[c], np.float64).ravel()[0] for c in nc.SCALAR_COLS], np.float32)
    return psi, scal, {"params": json.loads(row["synth_params"]),
                       "origin": str(row["synth_origin_shot"]).replace(".parquet", "")}


def one(job):
    path, machine, exp_dir = Path(job[0]), job[1], Path(job[2])
    if machine not in _P:
        _P[machine] = fv.load_pictures(CACHE / PICS[machine])
    pics = _P[machine]
    exp, mid = expanded_copy(path, machine, exp_dir)
    psi, scal, meta = _labels(path)
    base, extra = fv.shot_features(exp, pics, return_inputs=True, thomson_window=f5.THOMSON_WINDOW)
    feats = f5.assemble(base, extra, pics)
    t50_w, ipmax_w = t50_ipmax(*fv._pulse(exp, machine))
    t50_o, ipmax_o = t50_ipmax(*fv._pulse(REAL / ORIGIN_DIR[machine] / f"{meta['origin']}.parquet", machine))
    ipmax_o *= float(meta["params"].get("scale_ip", 1.0))
    jd, jv = f5.FEATURES.index("dIp_dt_norm"), f5.FEATURES.index("V_loop_norm")
    feats[:, jd] *= np.float32((t50_o / t50_w) * (ipmax_w / ipmax_o))
    feats[:, jv] *= np.float32(t50_o / t50_w)
    out = {"stem": path.stem, "origin": meta["origin"], "feats": feats, "mid": mid,
           "cur": np.asarray(extra["cur"], np.float32), "ip": np.asarray(extra["ip"], np.float32),
           "psi": psi * np.float32(LABEL_SIGN[machine]), "scal": scal}
    if machine == "d3d":
        feat = nc.featurize(exp)
        out["stamp"] = nc.mirror(feat["coils"], feat["s"]).astype(np.float32)
    th = {}
    for q in ("Te", "ne"):
        v, r, z = thomson_shot((str(exp), None, q))
        th[q] = v[mid]
    out.update(te=th["Te"], ne=th["ne"], tr=r[mid], tz=z[mid])
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _split(machine: str, origins: list[str], heldout_share: float) -> np.ndarray:
    if machine == "d3d":
        sh = pd.read_parquet(CACHE / "shots.parquet")
        not_train = set(sh.loc[sh.split != "train", "shot"].astype(str))
        return np.array(["heldout" if o in not_train else "train" for o in origins])
    # MAST: by origin shot, a stable hash so the split does not change when more cases are added.
    h = [int(hashlib.sha1(o.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for o in origins]
    return np.array(["heldout" if v < heldout_share else "train" for v in h])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["mast", "d3d"], required=True)
    ap.add_argument("--src", type=Path, required=True, help="folder of <machine>_synth_*.parquet (read only)")
    ap.add_argument("--tag", required=True, help="output name: <cache>/synth/<tag>/")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heldout-share", type=float, default=0.15, help="MAST: share of origin shots held out")
    a = ap.parse_args()
    import mast_region
    if a.machine == "mast" and mast_region.suffix() and not a.tag.endswith(mast_region.suffix()):
        raise SystemExit(f"{mast_region.ENV_VAR}={mast_region.name()}: the tag must end in {mast_region.suffix()} "
                         f"(MAST features depend on the region; envelope caches keep their names)")
    if a.machine == "mast" and not mast_region.suffix() and a.tag.endswith("_limiter"):
        raise SystemExit("a *_limiter tag needs FEC_MAST_MASK=limiter")
    files = sorted(a.src.glob("*.parquet"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no parquet files in {a.src}")
    out = CACHE / "synth" / a.tag
    exp_dir = out / f"expanded_{a.machine}"       # features_v2 reads the machine from the folder name
    exp_dir.mkdir(parents=True, exist_ok=True)
    geo = fv.geometry(a.machine)
    g = pq.read_table(files[0], columns=["efit_grid_R", "efit_grid_Z"]).to_pylist()[0]
    if not (np.allclose(np.ravel(g["efit_grid_R"]), geo["r"], atol=1e-4)
            and np.allclose(np.ravel(g["efit_grid_Z"]), geo["z"], atol=1e-4)):
        raise SystemExit("synthetic grid does not match features_v2.geometry; resample before use")
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, [(str(f), a.machine, str(exp_dir)) for f in files], chunksize=4), 1):
            res.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    width = max(len(r["te"]) for r in res)
    pad = lambda k: np.stack([np.r_[np.asarray(r[k], np.float32), np.full(width - len(r[k]), np.nan, np.float32)]  # noqa: E731
                              for r in res])
    arrays = {k: np.stack([r[k] for r in res]) for k in ("feats", "cur", "ip", "psi", "scal")}
    if a.machine == "d3d":
        arrays["stamp"] = np.stack([r["stamp"] for r in res])
    origins = [r["origin"] for r in res]
    split = _split(a.machine, origins, a.heldout_share)
    np.savez(out / "arrays.npz", stem=np.array([r["stem"] for r in res]), origin=np.array(origins), split=split,
             mid=np.int64(res[0]["mid"]), te=pad("te"), ne=pad("ne"), tr=pad("tr"), tz=pad("tz"), **arrays)
    manifest = {
        "machine": a.machine, "source": str(a.src), "n_cases": len(res), "label_sign": LABEL_SIGN[a.machine],
        "pictures": PICS[a.machine], **({"mast_mask": mast_region.name()} if a.machine == "mast" else {}),
        "window": {"spacing_ms": SPACING[a.machine][0], "half_frames": SPACING[a.machine][1]},
        "split_rule": "d3d: origin shot in our val/test split -> heldout" if a.machine == "d3d"
                      else f"mast: sha1(origin shot) < {a.heldout_share} -> heldout",
        "split_counts": {s: int((split == s).sum()) for s in ("train", "heldout")},
        "arrays_sha256": _sha(out / "arrays.npz"),
        "sources_sha256": {f.name: _sha(f) for f in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    f = arrays["feats"][:, res[0]["mid"]]
    print(f"wrote {out}: {len(res)} cases ({manifest['split_counts']}), feats {arrays['feats'].shape}, "
          f"rate_missing {np.nanmean(f[:, f5.FEATURES.index('rate_missing')]):.2f}, "
          f"thomson_missing {np.nanmean(f[:, f5.FEATURES.index('thomson_missing')]):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
