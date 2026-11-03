"""Smoke test of feat2: feature coverage / ranges per machine, and the zero-padded init reproduces the checkpoint's outputs."""
import sys
from pathlib import Path

import torch
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed as E                                   # noqa: E402
import feat2 as F2                                  # noqa: E402
import shots as S                                   # noqa: E402
from build_cache import machine_for                 # noqa: E402
from data import FrameSet, holdout_split            # noqa: E402
from model import PegsNet, forward, net_from_args   # noqa: E402
from physics import GSLayer                         # noqa: E402

dev = torch.device("cuda")
E.P_SHARE = 0.7
_, ho = holdout_split()
sets = [("DIII-D", "train", ho[:16], "_p7", 1.0), ("MAST", "test", S.stems("MAST", "test")[:40], "_p7s5", 0.5),
        ("MASTSYN", "synth", [s for s in S.stems("MASTSYN", "synth")[::50] if s.startswith("mastsyn_")], "_p7", 0.5), ("MASTVAL", "synth", S.stems("MASTVAL", "synth")[::4], "_p7", 0.5),
        ("JET", "synth", S.stems("JET", "synth")[::4], "_p7", 1.0), ("NSTX_FWD", "synth", S.stems("NSTX_FWD", "synth"), "_p7", 1.0)]
ck = torch.load(f"{R1_ROOT}/fusion-equilibrium-challenge-starter/my_experiments/m2/upegs_fz/runs/m2_b/model.pt", map_location=dev)
targs = ck["args"]
machines = {k[6:]: v.shape[0] for k, v in ck["net"].items() if k.startswith("gains.")}
net0 = net_from_args(targs, machines).to(dev); net0.load_state_dict(ck["net"]); net0.eval()
t2 = dict(targs); t2["feat2"] = 1
net1 = net_from_args(t2, machines).to(dev)
sd = dict(ck["net"]); own = net1.state_dict()
w_old, w_new = sd["scal.0.weight"], own["scal.0.weight"]
n_old, n_new = w_old.shape[1] // 2, w_new.shape[1] // 2
w = torch.zeros_like(w_new); w[:, :n_old] = w_old[:, :n_old]; w[:, n_new:n_new + n_old] = w_old[:, n_old:]
sd["scal.0.weight"] = w; sd["in_par"] = own["in_par"]
net1.load_state_dict(sd); net1.eval()
for mname, split, stems, tag, sol in sets:
    m = machine_for(mname, sol)
    gs = GSLayer(m, dev)
    fs = FrameSet(mname, split, stems, dev, labels=False, tag=tag)
    X0 = fs.d["X"].clone()
    F2.extend(fs, gs)
    X2 = fs.d["X"][:, -len(F2.NAMES2):]
    print(f"== {mname}: {len(fs)} frames; X {tuple(X0.shape)} -> {tuple(fs.d['X'].shape)}")
    print("   " + F2.describe(X2).replace("; ", "\n   "))
    idx = torch.arange(0, min(len(fs), 256), device=dev)
    b = fs.batch(idx)
    b0 = dict(b); b0["X"] = b["X"][:, :X0.shape[1]]
    with torch.no_grad():
        s0 = forward(net0, gs, b0, n_steps=4, harm=False)
        s1 = forward(net1, gs, b, n_steps=4, harm=False)
    print(f"   padded-init check: max |psi1 - psi0| = {float((s1['psi'] - s0['psi']).abs().max()):.2e} (psi scale {float(s0['psi'].abs().max()):.2f})")
