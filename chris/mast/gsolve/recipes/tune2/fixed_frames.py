"""Report-only check of the selection effect: score runs on a FIXED frame set, the real frames for which the baseline
recipe (runs/base_s0) kept an equilibrium; otherwise the frozen protocol (N_FIX pairs, frozen critic, same split).
    fixed_frames.py TAG [TAG ...]   -> fixed_frames.csv (appends)"""
import sys, json, numpy as np, pandas as pd
import tune_lib as tl
base = pd.read_parquet("runs/base_s0/synth_table.parquet")
bset = set(zip(base.shot, base.frame))
T_full = tl.real_table()
tl.real_table = lambda: T_full[[k in bset for k in T_full.index]]
rows = []
for t in sys.argv[1:]:
    syn = pd.read_parquet(f"runs/{t}/synth_table.parquet")
    res = json.load(open(f"runs/{t}/result.json"))
    P, st = tl.matched(syn)
    m = tl.critic_metrics(P, json.load(open(f"runs/{t}/split.json")), seed=0, perm=False, blocks=False)
    r = dict(tag=t, ll_fixed=m["ll"], gain_fixed=float(np.log(2) - m["ll"]), auc_fixed=m["auc"], n_pairs_fixed=st["n_pairs"],
             gain_protocol=res["gain"], loss_protocol=res["loss"])
    print(r, flush=True); rows.append(r)
pd.DataFrame(rows).to_csv("fixed_frames.csv", mode="a", index=False, header=not __import__("os").path.exists("fixed_frames.csv"))
