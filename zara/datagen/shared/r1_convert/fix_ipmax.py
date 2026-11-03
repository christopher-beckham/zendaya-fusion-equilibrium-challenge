"""Set the ip_over_max input feature of gen4 caches from the ORIGIN real shot's Ip trace (inputs only).

embed.time_features uses Ip / max(Ip over the record). A gen4 case is a ~45 ms piece of a real shot, so that ratio is ~1 for every
synthetic case, while on a real test shot (whole discharge) ramp-up frames sit at 0.3-0.8. Here the denominator becomes the
maximum Ip of the real shot the case was cut from (real MAST test cache, measured Ip only; no labels).

    python fix_ipmax.py train <machine>            # cache/ch2/g4/caches/<machine>_synth_wd, origin from the record src
    python fix_ipmax.py eval <cache dir> <data dir> # held-out eval cache (MAST split), origin from the parquet synth_meta
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4/fz")
import embed as E  # noqa: E402

REAL = "/fsx/zara.ercan/cache/mast/mast_public_test"
K = E.NAMES.index("ip_over_max")
_IPMAX = {}


def ipmax(origin):
    s = os.path.basename(origin).replace(".parquet", "")
    if s not in _IPMAX:
        ip = np.abs(np.load(os.path.join(REAL, s + ".npz"))["ip"].astype(np.float64))
        _IPMAX[s] = float(np.nanmax(np.clip(ip, 1e4, None)))
    return _IPMAX[s]


def fix(cache_file, origin):
    d = dict(np.load(cache_file))
    ip = np.clip(np.abs(d["ip"].astype(np.float64)), 1e4, None)
    new = np.clip(ip / ipmax(origin), 0.0, 1.2).astype(np.float32)
    old = d["X"][:, K].copy()
    d["X"][:, K] = new
    tmp = cache_file[:-4] + ".fix.npz"
    np.savez(tmp, **d)
    os.replace(tmp, cache_file)
    return old, new


def main():
    mode = sys.argv[1]
    olds, news = [], []
    if mode == "train":
        m = sys.argv[2]
        src = f"/fsx/zara.ercan/cache/ch2/g4/multi/{m}/src"
        cdir = f"/fsx/zara.ercan/cache/ch2/g4/caches/{m.lower()}_synth_wd"
        files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
        for i, f in enumerate(files):
            origin = str(np.load(os.path.join(src, os.path.basename(f)))["origin"])
            o, n = fix(f, origin); olds.append(o); news.append(n)
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    else:
        cdir, data = sys.argv[2], sys.argv[3]
        files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
        for f in files:
            stem = os.path.basename(f)[:-4]
            meta = json.loads(pq.read_table(os.path.join(data, stem + ".parquet"), columns=["synth_meta"]).to_pylist()[0]["synth_meta"])
            o, n = fix(f, meta["case"]["file"]); olds.append(o); news.append(n)
    o, n = np.concatenate(olds), np.concatenate(news)
    q = lambda v: np.round(np.percentile(v, [1, 10, 50, 90, 99]), 3).tolist()
    print(f"{len(files)} files, {len(o)} frames: ip_over_max before {q(o)} after {q(n)}", flush=True)


if __name__ == "__main__":
    main()
