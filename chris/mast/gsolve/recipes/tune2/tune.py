"""Closed-loop tuning of the gsolve recipe against a real-vs-synthetic critic.

  tune.py eval   --tag T [--knobs JSON] [--seed S] [--panel tune|test]   one evaluation (logged)
  tune.py params --tag T          choose the critic capacity by inner grouped CV on T's critic-train pairs (writes critic_params.json)
  tune.py null   --tag T [--n 20] permutation null of the held-out log loss / AUC on T's pairs (labels swapped within matched pairs)
  tune.py search [--n-trials N] [--stage NAME]    optuna TPE over the knob space, resumable (journal on /fsx)

Every evaluation appends a row to log.csv. State lives under this folder (runs/, optuna journal, log.csv).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import critic as cr  # noqa: E402
import tune_lib as tl  # noqa: E402

HERE = tl.HERE
LOG = HERE / "log.csv"
PANELS = json.loads((HERE / "panels.json").read_text())


PERSLICE = {"on": False}


def defaults():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp", HERE / "gen_param2.py")
    # gen_param imports torch/gsolve at module level; read DEFAULTS textually instead
    src = (HERE / "gen_param2.py").read_text()
    i = src.index("DEFAULTS = dict(")
    j = src.index(")\n", i)
    return eval(src[i + len("DEFAULTS = "):j + 1])  # noqa: S307


def panel_files(which):
    if which == "tune":
        shots = PANELS["train"] + PANELS["val"]
        split = {s: "train" for s in PANELS["train"]} | {s: "val" for s in PANELS["val"]}
    else:                                   # test panel: its own 60/40 split by shot (critic retrained there)
        sh = PANELS["test"]
        split = {s: ("train" if k < 72 else "val") for k, s in enumerate(sh)}
        shots = sh
    return [REALF(s) for s in shots], split


def REALF(s):
    return tl.REAL_DIR / f"mast_shot_{s}.parquet"


def log_row(res, stage):
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), stage=stage, tag=res["tag"], seed=res["seed"],
               loss=res["loss"], gain=res["gain"], ll=res["ll"], auc=res["auc"], auc_lo=res["auc_lo"], auc_hi=res["auc_hi"],
               n_iter=res["n_iter"], edist=res["edist"], frac_kept=res["frac_kept"], n_pairs=res["n_pairs"],
               syn_th_missing=res["syn_th_missing"],
               **{k: res.get(k) for k in res if k.startswith("auc_") and k not in ("auc_lo", "auc_hi")},
               **{k: res.get(k) for k in res if k.startswith("ll_")},
               **{f"lab_{k}": v for k, v in res["labels"].items()},
               **{f"k_{k}": v for k, v in res["knobs"].items()},
               top_groups=json.dumps(res["top_groups"]), top_single=json.dumps({k: round(v, 3) for k, v in res["top_single"].items()}),
               t_total=res["t_total"])
    df = pd.DataFrame([row])
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def run_eval(knobs, tag, seed=0, panel="tune", stage="manual", perm=True, shards=4, threads=8, perslice=None):
    kn = defaults() | (knobs or {})
    files, split = panel_files(panel)
    (tl.RUNS / tag).mkdir(parents=True, exist_ok=True)
    (tl.RUNS / tag / "split.json").write_text(json.dumps(split))
    fresh = not (tl.RUNS / tag / "result.json").exists()
    if perslice is None:
        perslice = PERSLICE["on"]
    res = tl.eval_knobs(kn, files, split, tag, seed=seed, perm=perm, shards=shards, threads=threads, perslice=perslice)
    if fresh:
        log_row(res, stage)
    print(f"[{tag}] ll {res['ll']:.4f} gain {res['gain']:.4f} auc {res['auc']:.3f} [{res['auc_lo']:.3f}, {res['auc_hi']:.3f}] "
          f"kept {res['frac_kept']:.2f} pairs {res['n_pairs']} li {res['labels'].get('li', np.nan):.2f} "
          f"kappa {res['labels'].get('kappa', np.nan):.2f} ({res['t_total']:.0f}s) top {list(res['top_groups'].items())[:4]}", flush=True)
    return res


def cmd_rescore(tag):
    """Recompute the critic metrics of a finished evaluation from its stored pairs (after the critic is frozen)."""
    d = tl.RUNS / tag
    res = json.loads((d / "result.json").read_text())
    syn = pd.read_parquet(d / "synth_table.parquet")
    P, st = tl.matched(syn)
    res["frac_kept"] = res.get("frac_kept", 0.9)
    P.to_parquet(d / "pairs.parquet")
    res.update(st)
    split = json.loads((d / "split.json").read_text())
    m = tl.critic_metrics(P, split, seed=0, perm=True)
    res.update(m)
    res["gain"] = float(np.log(2) - res["ll"])
    res["loss"] = tl.loss_of(res)
    (d / "result.json").write_text(json.dumps(res, indent=1, default=float))
    log_row(res, "rescore")
    print(tag, {k: res[k] for k in ("ll", "gain", "auc", "auc_lo", "auc_hi", "n_iter")}, list(res["top_groups"].items())[:6])


def cmd_params(tag):
    """Inner grouped 4-fold CV on the critic-train shots only; pick the capacity with the lowest held-out log loss."""
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold
    P = pd.read_parquet(tl.RUNS / tag / "pairs.parquet")
    P = P[P.shot.isin(PANELS["train"])].reset_index(drop=True)
    obj = cr.objective_columns()
    X, y, g = P[obj].values.astype(np.float32), P.y.values, P.shot.values
    grid = [dict(max_leaf_nodes=l, learning_rate=lr, min_samples_leaf=ms, l2_regularization=l2, max_iter=800)
            for l in (7, 15, 31) for lr in (0.03, 0.1) for ms in (20, 60) for l2 in (0.0, 1.0)]
    rows = []
    for prm in grid:
        lls = []
        for tr, te in GroupKFold(4).split(X, y, g):
            m, k = cr.fit_critic(X[tr], y[tr], g[tr], prm, seed=0)
            p = np.clip(cr.predict_at(m, k, X[te]), 1e-4, 1 - 1e-4)
            lls.append(log_loss(y[te], p, labels=[0, 1]))
        rows.append(dict(**prm, ll=float(np.mean(lls))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows).sort_values("ll")
    R.to_csv(HERE / "critic_params_grid.csv", index=False)
    best = R.iloc[0].to_dict()
    best = {k: (int(best[k]) if k in ("max_leaf_nodes", "min_samples_leaf", "max_iter") else float(best[k]))
            for k in ("max_leaf_nodes", "learning_rate", "min_samples_leaf", "l2_regularization", "max_iter")}
    tl.PARAMS_FILE.write_text(json.dumps(best, indent=1))
    print("frozen critic params:", best)


def cmd_null(tag, n):
    """Labels swapped at random within each matched (real, synthetic) pair, critic refitted: null of held-out ll / AUC."""
    P = pd.read_parquet(tl.RUNS / tag / "pairs.parquet")
    split = json.loads((tl.RUNS / tag / "split.json").read_text())
    obj = cr.objective_columns()
    X = P[obj].values.astype(np.float32)
    nr = int((P.y == 0).sum())
    assert (P.y.values[:nr] == 0).all() and len(P) == 2 * nr
    sp = P.shot.map(split).values
    tr, va = sp == "train", sp == "val"
    g = P.shot.values
    out = []
    rng = np.random.default_rng(123)
    for i in range(n):
        sw = rng.random(nr) < 0.5
        y = P.y.values.copy()
        y[:nr][sw], y[nr:][sw] = 1, 0
        r = cr.heldout_eval(X[tr], y[tr], g[tr], X[va], y[va], g[va], tl.critic_params(), seed=i, n_boot=2)
        out.append(dict(ll=r["ll"], auc=r["auc"], n_iter=r["n_iter"]))
        print(out[-1], flush=True)
    D = pd.DataFrame(out)
    D.to_csv(HERE / f"null_{tag}.csv", index=False)
    s = dict(ll_mean=D.ll.mean(), ll_p05=D.ll.quantile(0.05), ll_p01=D.ll.min(), auc_mean=D.auc.mean(), auc_p95=D.auc.quantile(0.95))
    (HERE / f"null_{tag}.json").write_text(json.dumps(s, indent=1))
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--knobs", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", default="tune")
    ap.add_argument("--stage", default="manual")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--study", default="s1")
    ap.add_argument("--screen", default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--perslice", type=int, default=0, help="1: critic sees per-slice-noise conversion (convert_perslice.py)")
    ap.add_argument("--seed-from", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    PERSLICE["on"] = bool(a.perslice)
    if a.cmd == "eval":
        kn = json.loads(Path(a.knobs).read_text()) if a.knobs and Path(a.knobs).exists() else json.loads(a.knobs or "{}")
        files, split = panel_files(a.panel)
        (tl.RUNS / a.tag).mkdir(parents=True, exist_ok=True)
        (tl.RUNS / a.tag / "split.json").write_text(json.dumps(split))
        run_eval(kn, a.tag, a.seed, a.panel, a.stage, shards=a.shards, threads=a.threads)
    elif a.cmd == "params":
        cmd_params(a.tag)
    elif a.cmd == "null":
        cmd_null(a.tag, a.n)
    elif a.cmd == "rescore":
        cmd_rescore(a.tag)
    elif a.cmd == "search":
        cmd_search(a.n, a.study, a.screen, a.patience, a.worker, a.shards, a.threads, a.seed_from)



# ---------------------------------------------------------------- search

def space(trial):
    k = {}
    k["k_lo"] = trial.suggest_float("k_lo", 1.2, 3.0)
    k["k_hi"] = k["k_lo"] + trial.suggest_float("k_span", 0.0, 2.0)
    k["am_lo"] = trial.suggest_float("am_lo", 0.5, 3.0)
    k["am_hi"] = k["am_lo"] + trial.suggest_float("am_span", 0.0, 3.0)
    k["an_lo"] = trial.suggest_float("an_lo", 0.2, 3.0)
    k["an_hi"] = k["an_lo"] + trial.suggest_float("an_span", 0.0, 4.0)
    k["w_te"] = trial.suggest_float("w_te", 0.0, 1.0)
    k["pk_up"] = trial.suggest_float("pk_up", 0.0, 2.0)
    k["pk_down"] = trial.suggest_float("pk_down", 0.0, 2.0)
    k["bv_edge"] = trial.suggest_categorical("bv_edge", [1, 0])
    k["bv_mu"] = trial.suggest_float("bv_mu", -10.0, 25.0)
    k["bv_sd"] = trial.suggest_float("bv_sd", 0.0, 10.0)
    k["edge_bias"] = trial.suggest_float("edge_bias", -0.04, 0.04)
    k["edge_sd"] = trial.suggest_float("edge_sd", 0.0, 0.04)
    k["zc_hi"] = trial.suggest_float("zc_hi", 0.0, 1.5)
    k["zc_sd"] = trial.suggest_float("zc_sd", 0.0, 0.04)
    k["sel_out"] = trial.suggest_float("sel_out", 0.02, 0.08)
    k["sel_in"] = trial.suggest_float("sel_in", 0.03, 0.15)
    k["noise_scale"] = trial.suggest_float("noise_scale", 0.5, 2.5)
    k["out_frac"] = trial.suggest_float("out_frac", 0.0, 0.08)
    k["out_sd"] = trial.suggest_float("out_sd", 0.2, 1.2)
    k["drop_frac"] = trial.suggest_float("drop_frac", 0.0, 0.05)
    k["ne_shift_mu"] = trial.suggest_float("ne_shift_mu", -0.03, 0.06)
    k["ne_shift_sd"] = trial.suggest_float("ne_shift_sd", 0.0, 0.05)
    k["shot_corr"] = trial.suggest_float("shot_corr", 0.0, 0.98)
    k["noise_model"] = 1                    # robust core noise estimate (artefact fixed in the generator, not searched)
    k["remap_k3"] = trial.suggest_categorical("remap_k3", [1, 0])
    k["noise_ne"] = trial.suggest_float("noise_ne", 0.4, 1.5)
    k["sel_mode"] = trial.suggest_categorical("sel_mode", [0, 1])
    k["rg_step"] = trial.suggest_categorical("rg_step", [0.05, 0.025])
    k["remap_side"] = trial.suggest_categorical("remap_side", [0, 1])
    return k


def to_params(kn):
    """knob dict -> optuna parameter dict (for enqueue_trial)."""
    d = defaults() | kn
    p = {q: d[q] for q in ("k_lo", "am_lo", "an_lo", "w_te", "pk_up", "pk_down", "bv_edge", "bv_mu", "bv_sd", "edge_bias",
                           "edge_sd", "zc_hi", "zc_sd", "sel_out", "sel_in", "noise_scale", "out_frac", "out_sd",
                           "drop_frac", "ne_shift_mu", "ne_shift_sd", "shot_corr")}
    p["k_span"], p["am_span"], p["an_span"] = d["k_hi"] - d["k_lo"], d["am_hi"] - d["am_lo"], d["an_hi"] - d["an_lo"]
    p["bv_edge"] = int(p["bv_edge"])
    p["remap_k3"] = int(d.get("remap_k3", 1))
    p["noise_ne"] = float(d.get("noise_ne", 1.0))
    p["sel_mode"] = int(d.get("sel_mode", 0))
    p["rg_step"] = float(d.get("rg_step", 0.05))
    p["remap_side"] = int(d.get("remap_side", 0))
    return p


def repeat_mean(knobs, tag, seeds=(1, 2), shards=4, threads=8):
    vals = [json.loads((tl.RUNS / tag / "result.json").read_text())]
    for s in seeds:
        vals.append(run_eval(knobs, f"{tag}_r{s}", seed=s, stage="repeat", perm=True, shards=shards, threads=threads))
    return vals


def cmd_search(n_trials, study_name, screen_file, patience, worker=0, shards=4, threads=8, seed_from=None):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    st = JournalStorage(JournalFileBackend(str(HERE / f"optuna_{study_name}.journal")))
    sampler = optuna.samplers.TPESampler(seed=1000 + worker, multivariate=True, group=True, n_startup_trials=12,
                                         constant_liar=True)   # per-worker seed + liar: no duplicate proposals
    study = optuna.create_study(study_name=study_name, storage=st, sampler=sampler, direction="minimize", load_if_exists=True)
    if seed_from and len(study.trials) == 0 and worker == 0:
        old = optuna.load_study(study_name=seed_from, storage=JournalStorage(JournalFileBackend(str(HERE / f"optuna_{seed_from}.journal"))))
        for t in old.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                pr = dict(t.params, remap_k3=1)
                dist = dict(t.distributions, remap_k3=optuna.distributions.CategoricalDistribution([1, 0]))
                study.add_trial(optuna.trial.create_trial(params=pr, distributions=dist, value=t.value,
                                                          user_attrs={"from": f"{seed_from}_t{t.number:03d}"}))
    if screen_file and worker == 0 and not any(t.state == optuna.trial.TrialState.WAITING for t in study.trials) \
            and not (HERE / f"screen_done_{study_name}").exists():
        for kn in json.loads(Path(screen_file).read_text()):
            study.enqueue_trial(to_params(kn), skip_if_exists=True)
        (HERE / f"screen_done_{study_name}").write_text("1")
    null = json.loads((HERE / "null_base_s0.json").read_text())
    noise = json.loads((HERE / "noise.json").read_text())        # sd of the loss from repeated baseline batches
    inc_f = HERE / f"incumbent_{study_name}.json"
    if not inc_f.exists() and seed_from and (HERE / f"incumbent_{seed_from}.json").exists():
        inc_f.write_text((HERE / f"incumbent_{seed_from}.json").read_text())
    inc = json.loads(inc_f.read_text()) if inc_f.exists() else None
    since = 0 if inc is None else inc.get("since", 0)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    while done < n_trials:
        trial = study.ask()
        kn = space(trial)
        inc = json.loads(inc_f.read_text()) if inc_f.exists() else inc      # the other worker may have moved it
        tag = f"{study_name}_t{trial.number:03d}"
        try:
            res = run_eval(kn, tag, seed=0, stage=study_name, shards=shards, threads=threads)
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] failed: {e}", flush=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        study.tell(trial, res["loss"])
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        inc = json.loads(inc_f.read_text()) if inc_f.exists() else inc
        since += 1
        # a candidate whose single batch beats the incumbent by more than the noise gets two fresh-batch repeats
        if inc is None or res["loss"] < inc["loss_mean"] - noise["loss_sd"]:
            vals = repeat_mean(kn, tag, shards=shards, threads=threads)
            mean = float(np.mean([v["loss"] for v in vals]))
            inc = json.loads(inc_f.read_text()) if inc_f.exists() else inc      # re-read: the other worker may have moved it
            if inc is None or mean < inc["loss_mean"] - 0.5 * noise["loss_sd"]:
                inc = dict(tag=tag, knobs=kn, loss_mean=mean, loss_sd=float(np.std([v["loss"] for v in vals])),
                           auc_mean=float(np.mean([v["auc"] for v in vals])), ll_mean=float(np.mean([v["ll"] for v in vals])),
                           auc_lo=min(v["auc_lo"] for v in vals), auc_hi=max(v["auc_hi"] for v in vals),
                           top_groups=res["top_groups"], since=0)
                since = 0
                print(f"NEW INCUMBENT {tag}: loss {mean:.4f} auc {inc['auc_mean']:.3f}", flush=True)
        inc["since"] = since
        inc_f.write_text(json.dumps(inc, indent=1))
        # stopping rules
        if inc["ll_mean"] >= null["ll_p05"] or (inc["auc_lo"] <= 0.5):
            print("STOP: incumbent indistinguishable from the permutation null", flush=True)
            break
        if since >= patience:
            print(f"STOP: no improvement beyond noise for {since} evaluations", flush=True)
            break
    print(json.dumps(inc, indent=1))


if __name__ == "__main__":
    main()
