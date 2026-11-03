#!/usr/bin/env python3
"""Input features z (zfeat) for every real MAST frame the generator would seed (|Ip| >= 100 kA, finite coils / TF,
Thomson table): input columns only (gen_param2.read_shot). -> mast/real_z.parquet"""
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsolve  # noqa: E402
import gen_param2 as gp  # noqa: E402
import zfeat  # noqa: E402

M = gsolve.load_machine("mast", solenoid_scale=0.5)


def one(f):
    try:
        s = gp.read_shot(f, M)
    except Exception as e:  # noqa: BLE001
        print(f, e, flush=True)
        return None
    aip = np.abs(s["ip"])
    hi = np.nonzero(aip >= 0.9 * np.nanmax(aip))[0]
    phase = np.ones(len(aip), int); phase[:hi[0]] = 0; phase[hi[-1] + 1:] = 2
    sel = np.nonzero((aip >= 100.0) & np.isfinite(s["coils"]).all(1) & np.isfinite(s["tf"]))[0]
    F = dict(ip=s["ip"][sel], ip_frac=aip[sel] / np.nanmax(aip), phase=phase[sel], rr=s["rr"][sel], tf=s["tf"][sel])
    tabs = [gp.ts_table(s["Te"][i], s["ne"][i], s["R_ts"]) for i in sel]
    has = np.array([t is not None for t in tabs])
    z = gp.frame_z(F, tabs, has, M)
    df = pd.DataFrame(z, columns=[f"z_{k}" for k in zfeat.FEATS])
    df.insert(0, "frame", sel); df.insert(0, "shot", s["shot"])
    df["a_th"] = [0.5 * (t[2] - t[1]) if t else np.nan for t in tabs]
    return df[has]


if __name__ == "__main__":
    files = sorted(glob.glob("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/mast_public_test/mast_shot_*.parquet"))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        out = [d for d in ex.map(one, files, chunksize=4) if d is not None]
    Path("mast").mkdir(exist_ok=True)
    D = pd.concat(out, ignore_index=True)
    D.to_parquet("mast/real_z.parquet")
    print(len(D), D.shot.nunique())
