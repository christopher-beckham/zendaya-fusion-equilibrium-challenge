"""Export the pixel-network input blocks of every official training frame and every public-test frame for one Ridge model
(the same transformations as train_c1.py, before the train-split standardisation of Groups A / B):

  Z   (N, 296)  Ridge-standardised, clipped features (294) + the two Thomson-availability flags     (depends on the Ridge file)
  XP  (N, 184)  Group A physics inputs, raw                                                         (Ridge independent)
  FB  (N, 66)   Group B first-pass equilibrium of the Ridge map, raw (NaN where undefined)          (depends on the Ridge file)
  ip, bt, t, shot (index into stems), k (frame index in the shot), T (frames of the shot)

    python export_inputs.py --ridge $FEQ_CACHE/nn/ridge_holdout.npz --tag r0
-> $FEQ_CACHE/ml/ch1s/inp_<tag>_train.npz (all 7,041 training shots, sorted stems) and inp_<tag>_test.npz
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "c1" / "nn"))           # firstpass.py (read-only)
import train_c1_copy as tc  # noqa: E402
from s_common import CACHE, OUTC, TEST_CACHE, all_stems, test_stems  # noqa: E402

PX = Path(f"{FEQ_CACHE}/nn/px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = torch.device("cpu")   # (package) CPU only, run by build_caches.sh; no GPU work in this step
    meta = json.loads((CACHE / "meta.json").read_text())
    names = meta["names"]
    basis = dict(np.load(CACHE / "basis.npz"))
    ridge = dict(np.load(a.ridge))
    assert list(ridge["names"]) == names
    phys = tc.Physics(basis, ridge, meta["cols"], dev)
    from firstpass import FirstPass
    fpm = FirstPass(tc.GRID_R, tc.GRID_Z, tc.ENVELOPE, dev)
    ibz = names.index("bz_over_bip")
    core = torch.tensor([n.startswith(("coreTe", "coreNe")) for n in names], device=dev)
    tan = torch.tensor([n.startswith(("tanTe", "tanNe")) for n in names], device=dev)
    for part, cache, stems, px in (("train", CACHE, all_stems(), PX / "diii_d_train"),
                                   ("test", TEST_CACHE, test_stems(), PX / "diii_d_public_test")):
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
        out = OUTC / f"inp_{a.tag}_{part}.npz"
        np.savez(out, Z=Zo, XP=d["Xp"].astype(np.float32), FB=FBo, ip=d["ip"], bt=d["bt"], t=d["t"], shot=d["shot"],
                 k=k, T=TT, stems=np.array(stems), fb_names=np.array(fpm.names))
        print(f"{part}: {len(stems)} shots, {N:,} frames, {time.time() - t0:.0f}s -> {out}", flush=True)
        del d


if __name__ == "__main__":
    main()
