"""Replace the βN of a built MAST submission by variant V2 or L2g. Label-free; nothing is uploaded.

    FEC_MAST_MASK=limiter CUDA_VISIBLE_DEVICES= python apply_bn_variant.py SRC_DIR OUT_DIR --variant V2 [--match-level]

Per frame, from the submitted ψ and MAST inputs only (bn_lib.frame_feats, as postproc_qb / apply_betaN):
  e_geo = 100 · 2μ0⟨p_e⟩/B0² · a · B0 / Ip  with the map's own a (half-width of its boundary) and B0 = F / R_geo
          (F = R0·B0 from the q_ITER feature), ⟨p_e⟩ = Thomson electron pressure averaged over the map's volume
  map_p = the same with the total pressure recovered from the map's current density
  V2  = 2.6088 · e_geo                     (scale fitted on v4 TRAINING frames, true maps; out/dd_constants.json)
  L2g = 2.0908 · e_geo + 0.2250 · map_p    (fitted on v4 HELD-OUT frames with Z7's predicted maps)
Frames where the variant is undefined or outside (0, 10] keep SRC's βN. ψ and q95 are copied; the DIII-D file is
hard-linked. --match-level rescales the variant so its median equals SRC's median βN (keeps SRC's level).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NN = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(NN)); sys.path.insert(0, str(NN / "mast_diag"))
C = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
SPLIT = os.environ.get("FEC_MAST_SPLIT", "mast_public_test")   # FEC_MAST_SPLIT: the MAST test set to process
SSFX = "" if SPLIT == "mast_public_test" else f"_{SPLIT}"   # file-name suffix for caches without the split in their name
MAST = Path(os.environ.get("FEC_DATA_ROOT", "/fsx/hackathon_offsite/fusion_equilibrium_challenge")) / "data" / SPLIT
_S = {}


def constants(variant):
    k = json.loads((HERE / "out" / "dd_constants.json").read_text())
    if variant == "V2":
        return [k["V2 scale·e_geo"]["train"], 0.0]
    if variant == "L2g":
        return list(k["L2g b·e_geo+c·map_p"]["ho"])
    raise SystemExit(f"unknown variant {variant}")


def _init(w):
    import bn_lib
    import features_v5 as f5
    import mast_region
    import physics_suite as ps
    _S.update(bn=bn_lib, ps=ps, m=ps.Machine("MAST"), w=w,
              feat=np.load(C / f"features_v5_{SPLIT}_sol0.5{mast_region.suffix()}.npy", mmap_mode="r"),
              ip=np.load(C / f"mast_inputs{SSFX}.npz")["ip"], fq=f5.FEATURES.index("q_ITER"))


def _shot(job):
    i, start, psi = job
    bn, ps, m, (we, wg) = _S["bn"], _S["ps"], _S["m"], _S["w"]
    rows = start + np.arange(len(psi))
    ip = _S["ip"][rows]
    bt = ps.bt_from_qiter(np.asarray(_S["feat"][rows])[:, _S["fq"]], ip, m.r0, m.a)
    out = np.full(len(psi), np.nan)
    for k in range(len(psi)):
        try:
            o = bn.frame_feats(m, psi[k], m.te[rows[k]], m.ne[rows[k]], m.cr[rows[k]], m.cz[rows[k]], ip[k], bt[k])
        except Exception:
            o = None
        if not o or not np.isfinite(o.get("pe_avg", np.nan)):
            continue
        b0g = bt[k] * m.r0 / o["rgeo"]
        v = we * bn.betan(o["pe_avg"], o["a_map"], b0g, ip[k])
        if wg:
            g = bn.betan(o.get("p_gs_avg", np.nan), o["a_map"], b0g, ip[k])
            v = v + wg * g if (np.isfinite(g) and 0 < g < 10) else np.nan
        out[k] = v
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--variant", required=True, choices=["V2", "L2g"])
    ap.add_argument("--match-level", action="store_true"); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    w = constants(a.variant)
    a.out.mkdir(parents=True, exist_ok=True)
    d = a.out / "diii_d_public_test.npz"
    if (a.src / "diii_d_public_test.npz").exists() and not d.exists():
        os.link(a.src / "diii_d_public_test.npz", d)
    z = np.load(a.src / f"{SPLIT}.npz")
    files = sorted(MAST.glob("*.parquet"))
    index = pd.read_parquet(C / f"features_v5_{SPLIT}_sol0.5.parquet").set_index("filename")
    jobs = [(i, int(index.loc[f.name, "start"]), z[f"shot_{i:04d}_psirz"]) for i, f in enumerate(files)]
    new = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(w,)) as ex:
        for i, b in ex.map(_shot, jobs, chunksize=4):
            new[i] = b
    old = np.concatenate([z[f"shot_{i:04d}_betaN"] for i in range(len(files))]).astype(np.float64)
    nb = np.concatenate([new[i] for i in range(len(files))])
    ok = np.isfinite(nb) & (nb > 0) & (nb <= 10)
    scale = float(np.median(old[ok]) / np.median(nb[ok])) if a.match_level else 1.0
    arrays = {k: z[k] for k in z.files}
    for i in range(len(files)):
        b = new[i] * scale
        keep = ~(np.isfinite(b) & (b > 0) & (b <= 10))
        b[keep] = z[f"shot_{i:04d}_betaN"][keep]
        arrays[f"shot_{i:04d}_betaN"] = b.astype(np.float32)
    np.savez(a.out / f"{SPLIT}.npz", **arrays)
    fin = np.concatenate([arrays[f"shot_{i:04d}_betaN"] for i in range(len(files))]).astype(np.float64)
    changed = np.abs(fin / old - 1) > 0.01
    print(f"{a.variant} weights {np.round(w, 4).tolist()}; defined on {ok.mean():.1%} of {len(ok)} frames; level scale {scale:.3f}")
    print(f"source βN 10/50/90 {np.percentile(old, [10, 50, 90]).round(3)}; new {np.percentile(fin, [10, 50, 90]).round(3)}; "
          f"median new/source {np.median(fin / old):.3f}; frames changed >1% {changed.mean():.1%}; "
          f"Spearman {pd.Series(fin).corr(pd.Series(old), method='spearman'):.3f}; Pearson {np.corrcoef(fin, old)[0, 1]:.3f}")
    print(f"dtypes {sorted({str(v.dtype) for v in arrays.values()})}; wrote {a.out}")


if __name__ == "__main__":
    main()
