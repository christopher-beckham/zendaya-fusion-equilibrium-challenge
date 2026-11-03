#!/usr/bin/env python3
"""Score every listed checkpoint on the synthetic held-out MAST sets and write synth_S_results.csv incrementally.

    python run_all.py                  # all groups, in priority order; resumes (skips rows already in results.jsonl)
    python run_all.py --sanity         # true maps through the same path (S must be 1), and true maps + physics q95/βN
    python run_all.py --parallel 3     # checkpoints scored at once (each: predict 6 threads, post-process 8, score 8)

Per checkpoint and eval set: predict_synth.py (worktree model code, FEC_MAST_MASK = the checkpoint's region) ->
postproc_synth.py (main-repo post-processing, same region) -> scorer.py (official scorer functions, MAST envelope, no
model code). Temporary prediction arrays live on local scratch and are deleted once scored. Log-derived columns
(DIII-D S from score_epNNN.log, real-MAST checks from mast_checks_epNNN.log, metrics.jsonl's epoch row) are parsed,
not recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HERE, PY, RUNS, SCR  # noqa: E402

RESULTS_JSONL = HERE / "results.jsonl"
RESULTS_CSV = HERE / "synth_S_results.csv"
SANITY_JSON = HERE / "synth_S_sanity.json"
LOGS = HERE / "logs"
SETS = ("g4_5000", "g4_v2")
Q95_CAP = 12.44
SUBMISSIONS = {("scr-U3-vertmast-noise06-hist1", 45): "T", ("scr-P1-U3-tightradialmast", 45): "U",
               ("scr-N1-U3-g4", 45): "W (Wc = q95cap columns)", ("scr-L1-N1-limiter", 45): "Y",
               ("scr-T4-vert-noise-hist1", 45): "R (built, not scored)"}
GROUP2 = ["scr-L0", "scr-L1", "scr-L2", "scr-L3", "scr-Z1", "scr-Z2", "scr-C1", "scr-C2", "scr-C3", "scr-B1", "scr-B2",
          "scr-F1", "scr-F2", "scr-F3", "scr-F4", "scr-K1", "scr-K3", "scr-K2c", "scr-K4c"]
GROUP3 = ["scr-N2", "scr-N3", "scr-N4", "scr-N5", "scr-N6", "scr-N7", "scr-N8", "scr-N9", "scr-W05", "scr-W20",
          "scr-A1", "scr-D20", "scr-L25", "scr-L50", "scr-U5", "scr-P2", "scr-G1", "scr-G2", "scr-G3", "scr-G4"]
_LOCK = threading.Lock()


def ckpts(run: Path) -> list[int]:
    return sorted(int(m.group(1)) for p in run.glob("score_ep*.pt") if (m := re.fullmatch(r"score_ep(\d+)\.pt", p.name)))


def runs_with_prefix(prefix: str) -> list[Path]:
    # "scr-L1" matches scr-L1-..., not scr-L10...; "-crashed" folders and runs still training (scr-G*) are skipped
    out = [p for p in sorted(RUNS.glob(f"{prefix}-*")) if p.is_dir() and not p.name.endswith("-crashed")]
    return out


def job_list() -> list[tuple[str, int, int]]:
    jobs, seen = [], set()

    def add(run: str, ep: int, group: int):
        if (run, ep) not in seen and (RUNS / run / f"score_ep{ep:03d}.pt").exists():
            seen.add((run, ep)); jobs.append((run, ep, group))
    for run, ep in SUBMISSIONS:
        add(run, ep, 1)
    for pre in GROUP2:
        for r in runs_with_prefix(pre):
            for ep in ckpts(r):
                add(r.name, ep, 2)
    for pre in GROUP3:
        for r in runs_with_prefix(pre):
            for ep in ckpts(r):
                add(r.name, ep, 3)
    for run, _ in SUBMISSIONS:                      # the leaderboard runs' other epochs, last
        for ep in ckpts(RUNS / run):
            add(run, ep, 4)
    return jobs


# ----------------------------------------------------------------------------------------------- log parsing

def _f(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def parse_d3d(run: Path, ep: int) -> dict:
    p = run / f"score_ep{ep:03d}.log"
    if not p.exists():
        return {}
    t = p.read_text(errors="replace")
    out = {"d3d_S": _f(r"S = ([\d.]+)\s+\(", t), "d3d_r2_psi": _f(r"R2_psi\s+([-\d.]+)", t),
           "d3d_r2_qb": _f(r"R2_\{q95,betaN\}\s+([-\d.]+)", t), "d3d_1-dlcfs": _f(r"1 - D_LCFS\s+([-\d.]+)", t),
           "d3d_consistency": _f(r"Consistency\s+([-\d.]+)", t), "d3d_S_normal": _f(r"S normal\s+([-\d.]+)", t),
           "d3d_S_reversed": _f(r"S reversed\s+([-\d.]+)", t), "d3d_n_shots": _f(r"\((\d+) shots\)", t, int)}
    for k in ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"):
        out[f"d3d_r2c_{k}"] = _f(rf"(?m)^\s+{k}\s+([-\d.]+)\s*$", "\n".join(t.splitlines()[-20:]))
    return out


def parse_mast_checks(run: Path, ep: int) -> dict:
    p = run / f"mast_checks_ep{ep:03d}.log"
    if not p.exists():
        return {}
    t = p.read_text(errors="replace")
    num = r"([-+]?\d+(?:\.\d+)?)"
    pats = {
        "mast_peakZ_beyond_0.2m_pct": rf"beyond ±0\.2 m {num}%", "mast_axis_minus_thomson_cm": rf"axis − Thomson {num} cm",
        "mast_Zaxis_minus_Z0_cm": rf"Z_axis − Z0 {num} cm", "mast_zigzag_p95": rf"zig-zag p95 {num}",
        "mast_kappa": rf"κ {num}", "mast_tri_top_minus_bot": rf"top − bottom triangularity {num}",
        "mast_volume_m3": rf"volume {num} m", "mast_current_over_ip": rf"current / Ip {num}",
        "mast_current_outside_pct": rf"current outside the boundary {num}%",
        "mast_hot_out": rf"hot_out {num}", "mast_cold_in": rf"cold_in {num}", "mast_order_te": rf"order_te {num}",
        "mast_order_ne": rf"order_ne {num}", "mast_order_p": rf"order_p {num}",
        "mast_mid_in_cm": rf"mid_in_cm {num}", "mast_mid_in_abs_cm": rf"mid_in_cm {num} \(\|\.\| {num}".replace(num, r"[-+]?[\d.]+", 1),
        "mast_mid_out_cm": rf"mid_out_cm {num}", "mast_mid_out_abs_cm": rf"mid_out_cm [-+]?[\d.]+ \(\|\.\| {num}",
        "mast_betaN_th_over_head": rf"betaN_th/betaN median {num}", "mast_q95_map_over_head": rf"q95_map/q95 median {num}",
        "mast_li_median": rf"li median {num}", "mast_li_std": rf"li median [\d.]+ std {num}",
        "mast_li_gt2.5_pct": rf"li > 2\.5 {num}%", "mast_spread_over_a": rf"spread/a median {num}",
    }
    return {k: _f(v, t) for k, v in pats.items()}


def parse_metrics(run: Path, ep: int) -> dict:
    p = run / "metrics.jsonl"
    if not p.exists():
        return {}
    row = None
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("epoch") == ep and d.get("source") != "score":
            row = d                                   # the last training row for that epoch
    if row is None:
        return {}
    return {f"m_{k}": v for k, v in row.items() if k != "epoch" and isinstance(v, (int, float))}


# ----------------------------------------------------------------------------------------------- one checkpoint

def sh(cmd: list[str], env: dict, log: Path) -> str:
    with open(log, "a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n"); fh.flush()
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE))
        fh.write(r.stdout); fh.flush()
    if r.returncode:
        raise RuntimeError(f"{cmd[1]} failed ({r.returncode}); see {log}")
    return r.stdout


def base_env(region: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "FEC_MAST_MASK"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"
    if region == "limiter":
        env["FEC_MAST_MASK"] = "limiter"
    return env


def score_ckpt(run: str, ep: int, group: int) -> list[dict]:
    rdir = RUNS / run
    cfg = json.loads((rdir / "config.json").read_text())
    region = str(cfg.get("mast_mask", "envelope"))
    ck = rdir / f"score_ep{ep:03d}.pt"
    tag = f"{run}_ep{ep:03d}"
    log = LOGS / f"{tag}.log"
    log.write_text("")
    common = {"run": run, "epoch": ep, "group": group, "submission": SUBMISSIONS.get((run, ep), ""),
              "trained_on": ";".join(cfg.get("synth") or []) or "none", "mask_region": region,
              "trained_on_v1_mast": any(s.startswith("zara_v1_mast") for s in cfg.get("synth") or []),
              "checkpoint": str(ck)}
    common.update(parse_d3d(rdir, ep)); common.update(parse_mast_checks(rdir, ep)); common.update(parse_metrics(rdir, ep))
    rows = []
    t0 = time.time()
    for s in SETS:
        pred = SCR / "pred" / f"{tag}_{s}.npz"
        pp = SCR / "pred" / f"{tag}_{s}_pp.npz"
        try:
            ta = time.time()
            sh([PY, "-u", "predict_synth.py", str(ck), s, str(pred), "--threads", "6"], base_env(region), log)
            tb = time.time()
            sh([PY, "-u", "postproc_synth.py", s, str(pred), str(pp), "--workers", "8"], base_env(region), log)
            tc = time.time()
            env = base_env(None); env["SYNTHS_WORKERS"] = "8"
            out = sh([PY, "-u", "scorer.py", "score", s, str(pp), "--q95cap", str(Q95_CAP)], env, log)
            td = time.time()
        finally:
            for f in (pred, pp):
                if f.exists():
                    f.unlink()
        res = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
        from scorer import flat
        for fold in ("all", "noV1"):
            row = dict(common)
            row["eval_set"] = s if fold == "all" else f"{s}_noV1origins"
            row.update(flat(res["as_is"][fold]))
            cap = res["q95cap"][fold]
            row.update({"S_q95cap": cap["S"], "r2_q95_q95cap": cap["r2_each"]["q95"], "r2_qb_q95cap": cap["r2_qb"]})
            row.update({"sec_predict": round(tb - ta, 1), "sec_postproc": round(tc - tb, 1), "sec_score": round(td - tc, 1)})
            rows.append(row)
    for r in rows:
        r["sec_checkpoint_total"] = round(time.time() - t0, 1)
    return rows


def write_csv() -> None:
    import pandas as pd
    rows = [json.loads(l) for l in RESULTS_JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    first = ["run", "epoch", "eval_set", "submission", "group", "trained_on", "trained_on_v1_mast", "mask_region",
             "S", "r2_psi", "r2_q95", "r2_betaN", "r2_qb", "dlcfs", "consistency",
             "r2c_R_axis", "r2c_Z_axis", "r2c_kappa", "r2c_tri_top", "r2c_tri_bot", "r2c_volume", "r2c_li",
             "S_q95cap", "r2_q95_q95cap", "r2_qb_q95cap", "psi_sign", "lcfs_fail_frac", "cons_fail_frac", "n_shots",
             "n_frames"]
    d3d = [c for c in df.columns if c.startswith("d3d_")]
    mast = [c for c in df.columns if c.startswith("mast_")]
    m = sorted(c for c in df.columns if c.startswith("m_"))
    rest = [c for c in df.columns if c not in set(first + d3d + mast + m)]
    df = df[[c for c in first if c in df.columns] + d3d + mast + m + rest]
    tmp = RESULTS_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(RESULTS_CSV)


def done_keys() -> set:
    if not RESULTS_JSONL.exists():
        return set()
    return {(d["run"], d["epoch"]) for d in map(json.loads, RESULTS_JSONL.read_text().splitlines()) if d}


def worker(job):
    run, ep, group = job
    t = time.time()
    try:
        rows = score_ckpt(run, ep, group)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {run} ep{ep}: {e}", flush=True)
        return
    with _LOCK:
        with open(RESULTS_JSONL, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        write_csv()
    main_row = rows[0]
    print(f"[{time.strftime('%H:%M:%S')}] {run} ep{ep} ({main_row['mask_region']}): S g4_5000 {rows[0]['S']:.4f} "
          f"noV1 {rows[1]['S']:.4f} | g4_v2 {rows[2]['S']:.4f} | {time.time() - t:.0f} s", flush=True)


def sanity() -> None:
    """True maps and true scalars through the scorer path (S = 1), and true maps + our physics q95/βN post-processing."""
    import numpy as np
    out = {}
    env = base_env(None); env["SYNTHS_WORKERS"] = "24"
    for s in SETS:
        o = sh([PY, "-u", "scorer.py", "perfect", s], env, LOGS / "sanity.log")
        out[f"{s}/perfect"] = json.loads(next(l for l in o.splitlines() if l.startswith("RESULT "))[7:])
        for region in ("limiter", "envelope"):
            sfx = "_limiter" if region == "limiter" else ""
            z = np.load(SCR / "sets" / f"{s}_ho{sfx}" / "arrays.npz")
            dummy = SCR / "pred" / f"sanity_{s}_{region}.npz"
            dummy.parent.mkdir(parents=True, exist_ok=True)
            nan = np.full(len(z["psi"]), np.nan, np.float32)        # no head fallback: a missing estimate scores as the mean
            np.savez(dummy, psi=z["psi"], q95_head=nan, betaN_head=nan, case=z["case"], frame=z["frame"])
            pp = dummy.with_name(dummy.stem + "_pp.npz")
            sh([PY, "-u", "postproc_synth.py", s, str(dummy), str(pp), "--workers", "24"], base_env(region),
               LOGS / "sanity.log")
            o = sh([PY, "-u", "scorer.py", "score", s, str(pp), "--q95cap", str(Q95_CAP)], env, LOGS / "sanity.log")
            out[f"{s}/true_maps+physics_qb_{region}"] = json.loads(
                next(l for l in o.splitlines() if l.startswith("RESULT "))[7:])
            dummy.unlink(); pp.unlink()
    SANITY_JSON.write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for f, r in vv.items():
                    print(f"{k} {kk}/{f}: S {r['S']:.4f} R2psi {r['r2_psi']:.4f} R2q95 {r['r2_each']['q95']:.4f} "
                          f"R2betaN {r['r2_each']['betaN']:.4f} D_LCFS {r['dlcfs']:.4f} cons {r['consistency']:.4f}",
                          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--groups", default="1,2,3,4")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    (SCR / "pred").mkdir(parents=True, exist_ok=True)
    if a.sanity:
        sanity()
        return 0
    groups = {int(g) for g in a.groups.split(",")}
    done = done_keys()
    jobs = [j for j in job_list() if j[2] in groups and (j[0], j[1]) not in done]
    print(f"{len(jobs)} checkpoints to score ({len(done)} done)", flush=True)
    if a.list:
        for j in jobs:
            print(*j)
        return 0
    # in priority order; at most `parallel` at once
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(worker, jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
