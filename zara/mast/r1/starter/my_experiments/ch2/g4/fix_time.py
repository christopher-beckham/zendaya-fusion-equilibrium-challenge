"""Time-history inputs of gen4 caches from the ORIGIN real shot (inputs only): every embed.TIME_FEATS column of a gen4 frame is
copied from the real test cache (mast_test_iv_full_wdiv: whole-shot history) at the same EFIT time. A gen4 case is a ~45 ms
piece of a shot, so its own time features never see the ramp-up (dlnip_ew out of the real range on ~31 % of real frames).
Frames without a matching real time get ip_over_max from the origin shot's maximum Ip (fix_ipmax.py rule).

    python fix_time.py train <machine>               # cache/ch2/g4/caches/<machine>_synth_wd, origin from the record src
    python fix_time.py eval <cache dir> <data dir>   # held-out cache (MAST split), origin from the parquet synth_meta
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fz"))   # this package's ch2/g4/fz
import embed as E  # noqa: E402

RC = f"{R1_ROOT}/cache/ch2/g4/caches/mast_test_iv_full_wdiv"
REAL = f"{R1_ROOT}/cache/mast/mast_public_test"
COLS = [E.NAMES.index(k) for k in E.TIME_FEATS]
KIP = E.NAMES.index("ip_over_max")
_R = {}


def real(origin):
    s = os.path.basename(origin).replace(".parquet", "")
    if s not in _R:
        c = np.load(os.path.join(RC, s + ".npz"))
        ip = np.abs(np.load(os.path.join(REAL, s + ".npz"))["ip"].astype(np.float64))
        _R[s] = (c["t"].astype(np.float64), c["X"][:, COLS].astype(np.float32), float(np.nanmax(np.clip(ip, 1e4, None))))
    return _R[s]


def fix(cache_file, origin):
    d = dict(np.load(cache_file))
    tr, Xr, ipmax = real(origin)
    t = d["t"].astype(np.float64)
    n = 0
    for k, tk in enumerate(t):
        j = int(np.argmin(np.abs(tr - tk)))
        if abs(tr[j] - tk) <= 0.6:
            d["X"][k, COLS] = Xr[j]; n += 1
        else:
            d["X"][k, KIP] = np.clip(np.clip(abs(float(d["ip"][k])), 1e4, None) / ipmax, 0.0, 1.2)
    tmp = cache_file[:-4] + ".fix.npz"
    np.savez(tmp, **d)
    os.replace(tmp, cache_file)
    return n, len(t)


def main():
    mode = sys.argv[1]
    nm = nt = 0
    if mode == "train":
        m = sys.argv[2]
        src = f"{R1_ROOT}/cache/ch2/g4/multi/{m}/src"
        files = sorted(glob.glob(f"{R1_ROOT}/cache/ch2/g4/caches/{m.lower()}_synth_wd/*.npz"))
        for i, f in enumerate(files):
            a, b = fix(f, str(np.load(os.path.join(src, os.path.basename(f)))["origin"])); nm += a; nt += b
    else:
        cdir, data = sys.argv[2], sys.argv[3]
        files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
        for f in files:
            stem = os.path.basename(f)[:-4]
            meta = json.loads(pq.read_table(os.path.join(data, stem + ".parquet"), columns=["synth_meta"]).to_pylist()[0]["synth_meta"])
            a, b = fix(f, meta["case"]["file"]); nm += a; nt += b
    print(f"{len(files)} files: {nm} / {nt} frames got the real-shot time features", flush=True)


if __name__ == "__main__":
    main()
