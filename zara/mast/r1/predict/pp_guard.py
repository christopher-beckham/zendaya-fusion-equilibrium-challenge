"""Post-processing intrusion guard: frames whose post-processed map crosses the centre column (scorer-style LCFS, ch2/scal eqgeom)
while the map before post-processing does not are restored from the pre-post-processing folder (psi, q95, betaN). Label-free.
    python pp_guard.py <pre dir> <post dir> <out dir> [--workers 8]"""
import argparse
import math
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter" / "my_experiments" / "ch2" / "scal"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter" / "my_experiments" / "ch2" / "syn" / "post"))
import eqgeom as G                               # noqa: E402
from postproc import grid                        # noqa: E402

CACHE = Path(__import__("os").environ.get("MAST_SPLIT_CACHE", f"{R1_ROOT}/cache/mast/mast_public_test"))   # copy for the Final: input cache of the split
MU0 = 4e-7 * math.pi
_G = {}


def _init():
    R, Z, M = grid("MAST"); _G["g"] = G.Grid(R, Z, M)


def crosses(psi, itf, ip):
    r = G.frame_physics(psi.astype(np.float64), _G["g"], MU0 * 24 * itf / (2 * math.pi), ip, None, levels_q=(0.95,), with_li=False, limiter=(0.2, 1.0))
    return bool(r.get("ok", 0)) and (r.get("limited", 0) == 1 or r.get("R_mid_in", 1.0) < 0.2)


def one(args):
    stem, pre, post, out = args
    a, b = dict(np.load(pre / f"{stem}.npz")), dict(np.load(post / f"{stem}.npz"))
    c = np.load(CACHE / f"{stem}.npz"); itf, ip = np.abs(c["itf"].astype(float)), np.abs(c["ip"].astype(float))
    n_fix = n_bad = 0
    for k in range(len(b["psi"])):
        if crosses(b["psi"][k], itf[k], ip[k]):
            n_bad += 1
            if not crosses(a["psi"][k], itf[k], ip[k]):
                for key in ("psi", "q95", "betaN"):
                    b[key][k] = a[key][k]
                n_fix += 1
    np.savez(out / f"{stem}.npz", **b)
    return n_bad, n_fix, len(b["psi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pre", type=Path); ap.add_argument("post", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sub = lambda d: d / "preds" if (d / "preds").exists() else d
    pre, post = sub(a.pre), sub(a.post)
    (a.out / "preds").mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in post.glob("*.npz"))
    with Pool(a.workers, initializer=_init) as pool:
        r = np.array(pool.map(one, [(s, pre, post, a.out / "preds") for s in stems], chunksize=4))
    print(f"{a.post} -> {a.out}: {r[:, 2].sum()} frames; post-processed maps crossing the column {r[:, 0].sum()}, restored from before "
          f"post-processing {r[:, 1].sum()}")


if __name__ == "__main__":
    main()
