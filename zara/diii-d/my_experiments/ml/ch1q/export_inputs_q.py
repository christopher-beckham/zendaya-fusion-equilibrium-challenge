"""Input export of one inputs-only test split for the ml/ch1q scalar networks (ml/ch1s export_inputs.py, test part only, with
the split name as an argument so it runs on a new fold, e.g. the Final-phase private test).

    python export_inputs_q.py --split diii_d_public_test [--ridge $FEQ_CACHE/nn/ridge_full.npz]
reads $FEQ_CACHE/nn/<split> (c1/nn per-shot cache) and $FEQ_CACHE/nn/px/<split> (Group A cache) ->
$FEQ_CACHE/ml/ch1q/inp_rfull_<split>.npz (Z, XP, FB, ip, bt, t, shot, k, T, stems; same layout as ml/ch1s inp_*_test).
"""
from __future__ import annotations
FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1s"))
sys.path.insert(0, str(HERE.parents[1] / "c1" / "nn"))           # firstpass.py (read-only)
import train_c1_copy as tc  # noqa: E402  (ml/ch1s copy of the ml/ch1 trainer, read-only import)

NN = Path(f"{FEQ_CACHE}/nn")
OUT = Path(f"{FEQ_CACHE}/ml/ch1q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ridge", type=Path, default=NN / "ridge_full.npz")
    a = ap.parse_args()
    dev = torch.device("cuda")
    meta = json.loads((NN / "diii_d_train" / "meta.json").read_text())
    names = meta["names"]
    basis = dict(np.load(NN / "diii_d_train" / "basis.npz"))
    ridge = dict(np.load(a.ridge))
    assert list(ridge["names"]) == names
    phys = tc.Physics(basis, ridge, meta["cols"], dev)
    from firstpass import FirstPass
    fpm = FirstPass(tc.GRID_R, tc.GRID_Z, tc.ENVELOPE, dev)
    ibz = names.index("bz_over_bip")
    core = torch.tensor([n.startswith(("coreTe", "coreNe")) for n in names], device=dev)
    tan = torch.tensor([n.startswith(("tanTe", "tanNe")) for n in names], device=dev)
    cache, px = NN / a.split, NN / "px" / a.split
    stems = sorted(p.stem for p in cache.glob("*.npz") if p.stem not in ("basis", "meta"))
    t0 = time.time()
    d, off, T = tc.load_shots(cache, stems, with_psi=False, px=px)
    N = len(d["ip"])
    Zo = np.empty((N, 296), np.float32); FBo = np.empty((N, len(fpm.names)), np.float32)
    for i in range(0, N, 8192):
        sl = slice(i, min(N, i + 8192))
        X = torch.from_numpy(d["X"][sl]).to(dev); I = torch.from_numpy(d["I"][sl]).to(dev)
        ip = torch.from_numpy(d["ip"][sl]).to(dev); bt = torch.from_numpy(d["bt"][sl]).to(dev)
        z = phys.z(X)
        fin = torch.isfinite(X)
        Zo[sl] = torch.cat([z, torch.stack([(~fin[:, core]).all(1), (~fin[:, tan]).all(1)], 1).float()], 1).cpu().numpy()
        P = phys.ridge_psi(z, I, ip)
        Pv = phys.vacuum(I) + phys.ecoil(I)
        FBo[sl] = fpm(P, ip, bt, X[:, ibz], torch.from_numpy(d["tspos"][sl]).to(dev), torch.from_numpy(d["tste"][sl]).to(dev),
                      torch.from_numpy(d["tsne"][sl]).to(dev), Pv).cpu().numpy()
    k = np.concatenate([np.arange(n) for n in T]); TT = np.repeat(T, T)
    out = OUT / f"inp_rfull_{a.split}.npz"
    np.savez(out, Z=Zo, XP=d["Xp"].astype(np.float32), FB=FBo, ip=d["ip"], bt=d["bt"], t=d["t"], shot=d["shot"],
             k=k, T=TT, stems=np.array(stems), fb_names=np.array(fpm.names))
    print(f"{a.split}: {len(stems)} shots, {N:,} frames, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
