"""One evaluation of a knob vector: generate a batch on a shot panel, build critic features, fit the critic, log.

eval_knobs(knobs, panel_files, tag, seed) -> dict of metrics; artefacts under runs/<tag>/.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import critic as cr
import noise_model
import zlib

HERE = Path(__file__).resolve().parent
PY = str(HERE / "venv" / "bin" / "python")
RUNS = HERE / "runs"
REAL_DIR = Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/mast_public_test")
_REAL = {}
LABELS = ["q95", "betaN", "li", "kappa", "R_ax", "Z_ax", "delta_upper", "delta_lower", "gen_bv", "gen_K", "gen_an", "gen_am", "gen_w"]


def real_table():
    if "T" not in _REAL:
        T = pd.read_parquet(HERE / "real" / "real_frames.parquet")
        _REAL["T"] = T.set_index(["shot", "frame"], drop=False)
    return _REAL["T"]


# ---------------------------------------------------------------- generation

def generate(knobs, files, out_dir: Path, seed=0, every=4, shards=4, threads=8, extra_args=(), log_prefix="gen"):
    out_dir.mkdir(parents=True, exist_ok=True)
    kf = out_dir / "knobs.json"
    kf.write_text(json.dumps(knobs))
    procs, outs = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONPATH="/fsx/chris.beckham/gsolve_copy")
    for s in range(shards):
        fs = files[s::shards]
        if not fs:
            continue
        lst = out_dir / f"files_{s}.txt"
        lst.write_text("\n".join(map(str, fs)) + "\n")
        o = out_dir / f"{log_prefix}_{s}.npz"
        outs.append(o)
        if o.exists():
            continue
        cmd = [PY, str(HERE / "gen_param2.py"), "--files-from", str(lst), "--out", str(o), "--knobs", str(kf),
               "--every", str(every), "--seed", str(seed * 1000 + s), *extra_args]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out_dir / f"{log_prefix}_{s}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    bad = [o for o in outs if not o.exists()]
    if bad:
        raise RuntimeError(f"generation failed: {bad}")
    return outs


def load_npz(paths):
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    out, meta = {}, dict(n_frames=0, n_has=0, n_frames_kept=0)
    for p in parts:
        for k in meta:
            meta[k] += int(p[k])
    keys = [k for k in parts[0] if parts[0][k].ndim >= 1 and k != "knobs"]
    for k in keys:
        vs = [p[k] for p in parts]
        if vs[0].ndim == 2 and k in ("Te", "ne", "R_ts", "Te_clean", "ne_clean", "inside"):
            w = max(v.shape[1] for v in vs)
            vs = [np.pad(v.astype(np.float64), ((0, 0), (0, w - v.shape[1])), constant_values=(0 if k == "inside" else np.nan))
                  for v in vs]
        out[k] = np.concatenate(vs)
    return out, meta


# ---------------------------------------------------------------- synthetic critic table

def _synth_rows(args):
    R, Te, Ne, ip, tf = args[:5]
    ps = args[5] if len(args) > 5 else None
    rows = []
    for k in range(len(ip)):
        if ps is None:
            d = cr.slice_features(R[k], Te[k] * 1e3, Ne[k] * 1e19, ip[k])
        else:
            # per-slice conversion (convert_perslice.py): the bracketing Thomson slices of the frame each carry an
            # independent noise draw on the noise-free profile; the model interpolates the per-slice columns in time
            Tc, Nc, rT, rN, ins, w, seed, kn = (ps[q][k] for q in range(8))
            ds = []
            for j in range(2):
                rng = np.random.default_rng([int(seed), j])
                tT, tN = noise_model.apply(Tc, Nc, rT, rN, kn, ins, rng)
                tT = np.where(np.isfinite(Tc), tT, np.nan); tN = np.where(np.isfinite(Nc), tN, np.nan)
                ds.append(cr.slice_features(R[k], tT * 1e3, tN * 1e19, ip[k]))
            mc = ["pe_pa"] + cr.MODEL_TH[1:]
            okj = [all(np.isfinite(dd[c]) for c in mc) for dd in ds]
            if okj[0] and okj[1]:
                d = {c: (1 - w) * ds[0][c] + w * ds[1][c] for c in ds[0]}
            else:
                d = ds[0] if okj[0] else ds[1]
        d["betaN_e"] = float(cr.betaN_e(d["pe_pa"], ip[k], tf[k]))
        rows.append(d)
    return rows


_TT = {}


def _thomson_times(shot):
    if shot not in _TT:
        import pyarrow.parquet as pq
        t = pq.read_table(REAL_DIR / f"mast_shot_{shot}.parquet", columns=["thomson_core_times"]).column(0)[0].as_py()
        _TT[shot] = np.asarray(t or [], float)
    return _TT[shot]


def bracket_weight(shot, t):
    """Interpolation weight of the later of the two real Thomson slices bracketing time t (0.5 if unknown)."""
    tt = np.sort(_thomson_times(shot))
    if len(tt) < 2 or not (tt[0] <= t <= tt[-1]):
        return 0.5
    j = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
    return float((t - tt[j - 1]) / (tt[j] - tt[j - 1])) if tt[j] > tt[j - 1] else 0.0


def synth_table(z, workers=16, perslice_knobs=None):
    n = len(z["shot"])
    nc = z["n_core"]
    C = z["Te"].shape[1]
    core = np.arange(C)[None] < nc[:, None]
    R = np.where(core, z["R_ts"], np.nan)
    Te = np.where(core, z["Te"], np.nan)
    Ne = np.where(core, z["ne"], np.nan)
    chunks = np.array_split(np.arange(n), max(1, workers * 2))
    if perslice_knobs is not None:
        Tc = np.where(core, z["Te_clean"], np.nan); Nc = np.where(core, z["ne_clean"], np.nan)
        ins = core & z["inside"].astype(bool)
        w = np.array([bracket_weight(s_, t_) for s_, t_ in zip(z["shot"].astype(str), z["time"])])
        seeds = np.array([zlib.crc32(f"{s_}_{f_}_{r_}".encode()) for s_, f_, r_ in zip(z["shot"].astype(str), z["frame"], z["rank"])])
        jobs = [(R[c], Te[c], Ne[c], z["ip"][c], z["tf"][c],
                 (Tc[c], Nc[c], z["rel_te"][c], z["rel_ne"][c], ins[c], w[c], seeds[c], [perslice_knobs] * len(c))) for c in chunks]
    else:
        jobs = [(R[c], Te[c], Ne[c], z["ip"][c], z["tf"][c]) for c in chunks]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_synth_rows, jobs))
    rows = [r for p in parts for r in p]
    d = pd.DataFrame(rows)
    d.insert(0, "rank", z["rank"]); d.insert(0, "frame", z["frame"]); d.insert(0, "shot", z["shot"].astype(str))
    for k in LABELS:
        if k in z:
            d[k] = z[k]
    return d


# ---------------------------------------------------------------- matched critic dataset

def matched(syn: pd.DataFrame):
    """Synthetic rows with both +-20 ms neighbours (same rank) and their real twins (same shot, frame)."""
    T = real_table()
    cols_now = cr.MODEL_TH + cr.REL + cr.NOISE + cr.OUTWIN
    s = syn.set_index(["shot", "rank", "frame"])
    key = list(zip(syn.shot, syn["rank"], syn.frame))
    kp = list(zip(syn.shot, syn["rank"], syn.frame - 4))
    kn = list(zip(syn.shot, syn["rank"], syn.frame + 4))
    idx = set(s.index)
    has_nb = np.array([(a in idx) and (b in idx) for a, b in zip(kp, kn)])
    rk = list(zip(syn.shot, syn.frame))
    in_real = np.array([k in T.index for k in rk])
    sel = np.nonzero(has_nb & in_real)[0]
    S_now = syn.iloc[sel][cols_now].reset_index(drop=True)
    S_prev = s.loc[[kp[i] for i in sel], cols_now].reset_index(drop=True)
    S_next = s.loc[[kn[i] for i in sel], cols_now].reset_index(drop=True)
    Rr = T.loc[[rk[i] for i in sel]].reset_index(drop=True)
    ok_real = (np.abs(Rr.lag_prev - 20) < 0.6) & (np.abs(Rr.lag_next - 20) < 0.6) & (Rr["m:thomson_missing"] == 0) \
        & (Rr["prev:m:thomson_missing"] == 0) & (Rr["next:m:thomson_missing"] == 0)
    syn_ok = np.isfinite(S_now[cr.MODEL_TH].values).all(1) & np.isfinite(S_prev[cr.MODEL_TH].values).all(1) \
        & np.isfinite(S_next[cr.MODEL_TH].values).all(1)
    stats = dict(n_syn=len(syn), n_with_nb=int(has_nb.sum()), n_real_ok=int(ok_real.sum()),
                 syn_th_missing=float(1 - syn_ok[ok_real.values].mean()) if ok_real.any() else np.nan)
    keep = ok_real.values             # synthetic frames with missing Thomson inputs stay in (NaN): a real giveaway
    S_now, S_prev, S_next, Rr = (x[keep].reset_index(drop=True) for x in (S_now, S_prev, S_next, Rr))

    def block(now, prev, nxt):
        D = {}
        for c in cols_now:
            D[c] = now[c].values
            D[f"dprev:{c}"] = now[c].values - prev[c].values
            D[f"dnext:{c}"] = nxt[c].values - now[c].values
        return D
    Ds = block(S_now, S_prev, S_next)
    rn = lambda pre: pd.DataFrame({c: Rr[pre + (f"m:{c}" if c in cr.MODEL_TH else c)].values for c in cols_now})  # noqa: E731
    Dr = block(rn(""), rn("prev:"), rn("next:"))
    for c in cr.COIL:
        Ds[c] = Rr[f"m:{c}"].values
        Dr[c] = Rr[f"m:{c}"].values
    Xs, Xr = pd.DataFrame(Ds), pd.DataFrame(Dr)
    g = Rr.shot.values
    P = pd.concat([Xr, Xs], ignore_index=True)
    P["y"] = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))].astype(int)
    P["shot"] = np.r_[g, g]
    stats["n_pairs"] = int(len(Xs))
    return P, stats


PARAMS_FILE = HERE / "critic_params.json"


def critic_params():
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(cr.DEFAULT_PARAMS)


BLOCKS = {
    "model_only": cr.COIL + cr.MODEL_TH + [f"d{s}:{c}" for s in ("prev", "next") for c in cr.MODEL_TH],
    "now_only": cr.COIL + cr.MODEL_TH + cr.REL,
    "noise": cr.COIL + cr.NOISE,
    "outwin": cr.COIL + cr.OUTWIN,
}


def critic_metrics(P, split, seed=0, perm=True, blocks=True):
    """split: dict shot -> 'train' / 'val'. Critic fitted on train shots (early stopping inside), scored on val shots."""
    if N_FIX and len(P) > 2 * N_FIX:      # the same random pairs (real and synthetic member together)
        nr = len(P) // 2
        keep = np.sort(np.random.default_rng(0).choice(nr, N_FIX, replace=False))
        P = P.iloc[np.r_[keep, keep + nr]].reset_index(drop=True)
    sp = P.shot.map(split).values
    tr, va = sp == "train", sp == "val"
    y, g = P.y.values, P.shot.values
    obj = cr.objective_columns()
    params = critic_params()
    X = P[obj].values.astype(np.float32)
    r = cr.heldout_eval(X[tr], y[tr], g[tr], X[va], y[va], g[va], params, seed=seed, perm_feats=obj if perm else None)
    out = {"ll": r["ll"], "auc": r["auc"], "auc_lo": r["auc_lo"], "auc_hi": r["auc_hi"], "n_iter": r["n_iter"],
           "top_groups": {k: round(v, 4) for k, v in list(r.get("imp", {}).items())[:10]},
           "n_train": int(tr.sum()), "n_val": int(va.sum())}
    sa = cr.single_auc(X[va], y[va], obj)
    out["top_single"] = dict(sorted(sa.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10])
    th = cr.MODEL_TH + cr.REL
    out["edist"] = float(cr.energy_distance(P.loc[(P.y == 1).values & va, th].values, P.loc[(P.y == 0).values & va, th].values))
    if blocks:
        for name, cols in BLOCKS.items():
            Xb = P[cols].values.astype(np.float32)
            rb = cr.heldout_eval(Xb[tr], y[tr], g[tr], Xb[va], y[va], g[va], params, seed=seed, n_boot=2)
            out[f"ll_{name}"], out[f"auc_{name}"] = rb["ll"], rb["auc"]
    return out


N_PAIRS_REF = 2013          # median matched pairs of the three baseline batches on the tuning panel
KEPT_REF = 0.905            # baseline share of frames with at least one kept equilibrium
N_FIX = 1200                # matched pairs given to the critic in every evaluation (equal critic power)


def loss_of(res):
    """Held-out critic gain over a coin flip (nats; the critic always sees N_FIX random pairs, so its power does not
    depend on how many frames survive) + coverage penalties: frames the recipe fails to produce (kept share below the
    baseline's) and, as a hard guard, batches with too few pairs to sample N_FIX from."""
    gain = float(np.log(2) - res["ll"])
    kept = res["frac_kept"] / KEPT_REF
    return gain + max(0.0, 0.97 - kept) + 2.0 * max(0.0, 1.0 - res["n_pairs"] / N_FIX)


def eval_knobs(knobs, files, split, tag, seed=0, every=4, shards=4, threads=8, perm=True, keep_npz=True, perslice=False):
    t0 = time.time()
    d = RUNS / tag
    res_f = d / "result.json"
    if res_f.exists():
        return json.loads(res_f.read_text())
    outs = generate(knobs, files, d, seed=seed, every=every, shards=shards, threads=threads, extra_args=("--no-psi",))
    t_gen = time.time() - t0
    z, meta = load_npz(outs)
    syn = synth_table(z, workers=shards * threads, perslice_knobs=knobs if perslice else None)
    syn.to_parquet(d / "synth_table.parquet")
    P, st = matched(syn)
    P.to_parquet(d / "pairs.parquet")
    m = critic_metrics(P, split, seed=0, perm=perm)
    lab = {k: float(np.nanmedian(syn[k])) for k in LABELS if k in syn}
    lab.update({f"{k}_p10": float(np.nanpercentile(syn[k], 10)) for k in ("li", "kappa", "q95", "betaN", "Z_ax") if k in syn})
    lab.update({f"{k}_p90": float(np.nanpercentile(syn[k], 90)) for k in ("li", "kappa", "q95", "betaN", "Z_ax") if k in syn})
    frac_kept = meta["n_frames_kept"] / max(meta["n_frames"], 1)
    res = dict(tag=tag, seed=seed, knobs=knobs, **{k: v for k, v in m.items()}, **st, frac_kept=frac_kept,
               n_eq=int(len(syn)), labels=lab, perslice=bool(perslice), t_gen=t_gen, t_total=time.time() - t0)
    res["gain"] = float(np.log(2) - res["ll"])        # what the critic gains over a coin flip (nats); 0 = indistinguishable
    res["loss"] = loss_of(res)
    res_f.write_text(json.dumps(res, indent=1, default=float))
    if not keep_npz:
        for o in outs:
            o.unlink(missing_ok=True)
    return res
