"""Synthetic-MAST S of one checkpoint or an ensemble: the mean of the checkpoints' psi maps and q95/betaN heads, then
the same post-processing and official scorer as run_all.py.  All members must use the same MAST region.

usage: ensemble_synth.py [--out JSON] [--sets g4_5000,g4_v2] NAME CKPT [CKPT ...]
Prints one RESULT line per eval set and fold and appends to logs/ensembles.jsonl; --out also writes
{"name", "members", "S": {"<set>" or "<set>_noV1origins": S}, "result": {set: scorer output}} (train_mm.py --synthS-every).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from paths import PY, SCR
from run_all import LOGS, Q95_CAP, SETS, base_env
from paths import SETS as ALL_SETS

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("ckpts", type=Path, nargs="+")
ap.add_argument("--out", type=Path)
ap.add_argument("--sets", default=",".join(SETS))
ap.add_argument("--threads", type=int, default=6, help="predict_synth threads")
ap.add_argument("--workers", type=int, default=8, help="post-processing and scorer workers")
ap.add_argument("--q95-from", type=Path, nargs="+", default=None,
                help="take the submitted q95 from the mean map of these checkpoints instead (as splice_q95.py)")
ap.add_argument("--bn-blend", type=float, default=0.0, help="submitted betaN = w * mean betaN head + (1 - w) * the"
                " Thomson-based physics estimate (0 = as submitted so far, 1 = head only)")
ap.add_argument("--tsmooth-psi", type=float, default=0.0, help="Gaussian smoothing of the mean map over each case's frames"
                " in time, sigma in ms (frame times from the held-out parquet); 0 = off")
ap.add_argument("--tsmooth-qb", type=float, default=0.0, help="the same for the submitted q95 and betaN, sigma in ms")
ap.add_argument("--tta", action="store_true", help="also predict on the up-down mirrored inputs (set <set>m), flip back"
                " and average; sets without a mirrored version are scored as usual")
a = ap.parse_args()
if a.out:                            # train_mm.py jobs: the 20,000-set held-out needs more scorer processes to keep up,
    a.workers = max(a.workers, 10)   # and only the first (main-metric) set is scored; others are scored afterwards
    a.sets = a.sets.split(",")[0]
name, cks, sets = a.name, a.ckpts, a.sets.split(",")
regions = {str(json.loads((ck.parent / "config.json").read_text()).get("mast_mask", "envelope")) for ck in cks}
assert len(regions) == 1, f"members use different MAST regions: {regions}"
region = regions.pop()


def frame_times(set_name, case, frame):
    """efit_times (ms) of every row, from the held-out parquet of its case."""
    import pyarrow.parquet as pq
    src, out, cache = ALL_SETS[set_name]["parquet"], np.zeros(len(case)), {}
    for i, (c, f) in enumerate(zip(case.astype(str), frame)):
        if c not in cache:
            stem = c.removeprefix("heldout_").replace("_mirror_", "_")
            cache[c] = np.asarray(pq.read_table(src / f"{stem}.parquet", columns=["efit_times"]).column(0)[0].as_py(), float)
        out[i] = cache[c][int(f)]
    return out


def tsmooth(x, case, t, sigma):
    """Gaussian-weighted mean over the frames of the same case (rows of x), weights exp(-dt^2 / 2 sigma^2)."""
    y = np.empty_like(x)
    for c in np.unique(case):
        r = np.nonzero(case == c)[0]
        w = np.exp(-0.5 * ((t[r][:, None] - t[r][None, :]) / sigma) ** 2)
        w /= w.sum(1, keepdims=True)
        y[r] = np.tensordot(w, x[r], axes=(1, 0)).astype(x.dtype)
    return y
LOGS.mkdir(exist_ok=True)
(SCR / "pred").mkdir(parents=True, exist_ok=True)
log = open(LOGS / f"ens_{name}.log", "w")


def sh(cmd, env):
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log.write(out.stdout + out.stderr); log.flush()
    if out.returncode:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    return out.stdout


summary = {"name": name, "members": [str(c) for c in cks], "S": {}, "result": {}}
for s in sets:
    preds = [SCR / "pred" / f"ens_{name}_{i}_{s}.npz" for i in range(len(cks))]
    mean, pp = SCR / "pred" / f"ens_{name}_{s}.npz", SCR / "pred" / f"ens_{name}_{s}_pp.npz"
    try:
        for ck, p in zip(cks, preds):
            sh([PY, "-u", "predict_synth.py", str(ck), s, str(p), "--threads", str(a.threads)], base_env(region))
        zs = [dict(np.load(p)) for p in preds]
        if a.tta and f"{s}m" in ALL_SETS:
            for ck, p, z in zip(cks, preds, zs):
                pm = p.with_name(p.stem + "_m.npz")
                try:
                    sh([PY, "-u", "predict_synth.py", str(ck), f"{s}m", str(pm), "--threads", str(a.threads)],
                       base_env(region))
                    zm = np.load(pm)
                finally:
                    pm.unlink(missing_ok=True)
                assert np.array_equal(np.char.replace(zm["case"].astype(str), "_mirror_", "_"), z["case"].astype(str))
                assert np.array_equal(zm["frame"], z["frame"])
                n = int(round(np.sqrt(z["psi"].shape[1])))
                back = zm["psi"].reshape(len(zm["psi"]), n, n)[:, ::-1, :].reshape(len(zm["psi"]), -1)
                z["psi"] = 0.5 * (z["psi"] + back)
                for k in ("q95_head", "betaN_head"):
                    z[k] = 0.5 * (z[k] + zm[k])
        for z in zs[1:]:
            assert np.array_equal(z["case"], zs[0]["case"]) and np.array_equal(z["frame"], zs[0]["frame"])
        np.savez(mean, **{k: np.mean([z[k] for z in zs], axis=0) for k in ("psi", "q95_head", "betaN_head")},
                 case=zs[0]["case"], frame=zs[0]["frame"])
        if a.tsmooth_psi or a.tsmooth_qb:
            zz = dict(np.load(mean)); tt = frame_times(s, zz["case"], zz["frame"])
            if a.tsmooth_psi:
                zz["psi"] = tsmooth(zz["psi"], zz["case"], tt, a.tsmooth_psi)
                np.savez(mean, **zz)
        sh([PY, "-u", "postproc_synth.py", s, str(mean), str(pp), "--workers", str(a.workers)], base_env(region))
        if a.bn_blend or a.tsmooth_qb:
            zp, zh = dict(np.load(pp)), np.load(mean)
            assert np.array_equal(zp["case"], zh["case"]) and np.array_equal(zp["frame"], zh["frame"])
            if a.bn_blend:
                zp["betaN"] = (a.bn_blend * zh["betaN_head"] + (1 - a.bn_blend) * zp["betaN"]).astype(zp["betaN"].dtype)
            if a.tsmooth_qb:
                for k in ("q95", "betaN"):
                    zp[k] = tsmooth(zp[k], zp["case"], tt, a.tsmooth_qb)
            np.savez(pp, **zp)
        if a.q95_from:
            qp = [SCR / "pred" / f"ens_{name}_q{i}_{s}.npz" for i in range(len(a.q95_from))]
            qm, qpp = SCR / "pred" / f"ens_{name}_q_{s}.npz", SCR / "pred" / f"ens_{name}_q_{s}_pp.npz"
            try:
                for ck, p in zip(a.q95_from, qp):
                    sh([PY, "-u", "predict_synth.py", str(ck), s, str(p), "--threads", str(a.threads)], base_env(region))
                qz = [np.load(p) for p in qp]
                np.savez(qm, **{k: np.mean([z[k] for z in qz], axis=0) for k in ("psi", "q95_head", "betaN_head")},
                         case=qz[0]["case"], frame=qz[0]["frame"])
                sh([PY, "-u", "postproc_synth.py", s, str(qm), str(qpp), "--workers", str(a.workers)], base_env(region))
                zm, zq = dict(np.load(pp)), np.load(qpp)
                assert np.array_equal(zm["case"], zq["case"]) and np.array_equal(zm["frame"], zq["frame"])
                zm["q95"] = zq["q95"]
                np.savez(pp, **zm)
            finally:
                for f in [*qp, qm, qpp]:
                    f.unlink(missing_ok=True)
        env = base_env(None); env["SYNTHS_WORKERS"] = str(a.workers)
        if a.out:                    # jobs from train_mm.py --synthS-every: fast scoring (scorer.py SYNTHS_FAST)
            env["SYNTHS_FAST"] = "1"
        out = sh([PY, "-u", "scorer.py", "score", s, str(pp), "--q95cap", str(Q95_CAP)], env)
    finally:
        for f in [*preds, mean, pp]:
            f.unlink(missing_ok=True)
    res = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
    for fold in ("all", "noV1"):
        print(f"RESULT {name} {s} {fold} S {res['as_is'][fold]['S']:.4f}", flush=True)
        summary["S"][s if fold == "all" else f"{s}_noV1origins"] = res["as_is"][fold]["S"]
    summary["result"][s] = res
    if a.out:
        summary["fast"] = True
    with open(LOGS / "ensembles.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "set": s, "members": summary["members"], "result": res}) + "\n")
if a.out:
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary))
    tmp.replace(a.out)
