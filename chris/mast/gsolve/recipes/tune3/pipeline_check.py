"""Check the critic's per-slice emulation against the real conversion path on the test panel.

Generates the test panel (every 4th frame, with flux maps) with the given knobs, converts it with convert_perslice.py,
computes the exact model inputs with synth_cache_seq.one (as the training cache does), and fits the frozen critic on the
model-input block (coil + 13 Thomson inputs + their +-20 ms changes) against the real twins. Prints the held-out log loss
/ AUC next to the emulated value for the same block.

    python pipeline_check.py KNOBS_JSON TAG [--seed 10]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["FEC_MAST_MASK"] = "limiter"
os.environ.setdefault("FEC_SYNTH_HALF", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import critic as cr  # noqa: E402
import tune  # noqa: E402
import tune_lib as tl  # noqa: E402

NN = "/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn"


def _feat(job):
    sys.path.insert(0, NN)
    import synth_cache_seq as scs
    path = job[0]
    out = scs.one(job)
    et = np.asarray(pq.read_table(path, columns=["efit_times"]).column(0)[0].as_py(), float)
    return [(o["origin"], et[o["frame"]], o["feats"][4], o["feats"][0], o["feats"][8]) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("knobs")
    ap.add_argument("tag")
    ap.add_argument("--seed", type=int, default=10)
    a = ap.parse_args()
    kn = tune.defaults() | json.loads(Path(a.knobs).read_text())
    d = tl.RUNS / a.tag
    files, split = tune.panel_files("test")
    outs = tl.generate(kn, files, d, seed=a.seed, every=4, shards=4, threads=4, extra_args=())
    clean = []
    for o in outs:
        c = o.with_name(o.stem + "_c.npz")
        if not c.exists():
            z = dict(np.load(o))
            np.savez(c, **{k: v for k, v in z.items() if v.ndim > 0})
        clean.append(str(c))
    (d / "knobs_full.json").write_text(json.dumps(kn))
    lay = d / "layout"
    if not (lay / "convert_summary.json").exists():
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH="/fsx/chris.beckham/gsolve_copy")
        subprocess.run([tl.PY, str(tl.HERE / "convert_perslice.py"), "--npz", *clean, "--out", str(lay), "--heldout-thr", "0",
                        "--workers", "16", "--knobs", str(d / "knobs_full.json")], env=env, check=True)
    exp = d / "expanded_mast"
    exp.mkdir(exist_ok=True)
    jobs = [(str(p), str(exp), "train") for p in sorted((lay / "train").glob("mast_synth*.parquet"))]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(_feat, jobs, chunksize=2) for r in part]
    feats = json.load(open("/fsx/chris.beckham/fusion_cache/features_v5.json"))["features"]
    jj = [feats.index(c) for c in cr.COIL + cr.MODEL_TH]
    T = tl.real_table().reset_index(drop=True)
    T["shot_t"] = list(zip(T.shot, np.round(T.t, 1)))
    idx = {k: i for i, k in enumerate(T.shot_t)}
    S, R = [], []
    for origin, t, now, prev, nxt in rows:
        shot = origin.replace("mast_shot_", "")
        i = idx.get((shot, round(float(t), 1)))
        if i is None:
            continue
        r = T.iloc[i]
        if not (abs(r.lag_prev - 20) < .6 and abs(r.lag_next - 20) < .6 and r["m:thomson_missing"] == 0
                and r["prev:m:thomson_missing"] == 0 and r["next:m:thomson_missing"] == 0):
            continue
        s = {c: now[j] for c, j in zip(cr.COIL + cr.MODEL_TH, jj)}
        for c in cr.MODEL_TH:
            j = feats.index(c)
            s[f"dprev:{c}"] = now[j] - prev[j]
            s[f"dnext:{c}"] = nxt[j] - now[j]
        q = {c: r[f"m:{c}"] for c in cr.COIL + cr.MODEL_TH}
        for c in cr.MODEL_TH:
            q[f"dprev:{c}"] = r[f"m:{c}"] - r[f"prev:m:{c}"]
            q[f"dnext:{c}"] = r[f"next:m:{c}"] - r[f"m:{c}"]
        S.append(s | {"shot": shot}); R.append(q | {"shot": shot})
    P = pd.concat([pd.DataFrame(R).assign(y=0), pd.DataFrame(S).assign(y=1)], ignore_index=True)
    P.to_parquet(d / "pipeline_pairs.parquet")
    cols = tl.BLOCKS["model_only"]
    nr = len(R)
    if nr > tl.N_FIX:
        keep = np.sort(np.random.default_rng(0).choice(nr, tl.N_FIX, replace=False))
        P = P.iloc[np.r_[keep, keep + nr]].reset_index(drop=True)
    sp = P.shot.map(split).values
    tr, va = sp == "train", sp == "val"
    X, y, g = P[cols].values.astype(np.float32), P.y.values, P.shot.values
    r = cr.heldout_eval(X[tr], y[tr], g[tr], X[va], y[va], g[va], tl.critic_params(), perm_feats=cols)
    out = dict(tag=a.tag, n_pairs=nr, ll=r["ll"], auc=r["auc"], auc_lo=r["auc_lo"], auc_hi=r["auc_hi"],
               top=dict(list(r["imp"].items())[:6]))
    (d / "pipeline_check.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
