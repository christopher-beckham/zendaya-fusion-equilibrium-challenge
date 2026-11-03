#!/usr/bin/env python3
"""Ip-history features on real MAST frames (inputs only), for the range check against DIII-D. -> mast/real_h.parquet is NOT
written into the shared mast/ link: output mast_h.parquet here."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsolve  # noqa: E402
import gen_param2 as gp  # noqa: E402
import zfeat  # noqa: E402
M = gsolve.load_machine("mast", solenoid_scale=0.5)


def one(f):
    try:
        s = gp.read_shot(f, M)
    except Exception as e:  # noqa: BLE001
        return None
    aip = np.abs(s["ip"])
    hi = np.nonzero(aip >= 0.9 * np.nanmax(aip))[0]
    phase = np.ones(len(aip), int); phase[:hi[0]] = 0; phase[hi[-1] + 1:] = 2
    sel = np.nonzero((aip >= 100.0) & np.isfinite(s["coils"]).all(1) & np.isfinite(s["tf"]))[0][::2]
    F = dict(ip=s["ip"][sel], ip_frac=aip[sel] / np.nanmax(aip), phase=phase[sel], rr=s["rr"][sel], tf=s["tf"][sel],
             time=s["time"][sel], hidx=np.zeros(len(sel), int))
    tabs = [gp.ts_table(s["Te"][i], s["ne"][i], s["R_ts"]) for i in sel]
    has = np.array([t is not None for t in tabs])
    z = gp.frame_z(F, tabs, has, M, [s["ip_trace"]])
    df = pd.DataFrame(z, columns=[f"z_{k}" for k in zfeat.FEATS_ALL])
    df.insert(0, "frame", sel); df.insert(0, "shot", s["shot"])
    return df[has]


if __name__ == "__main__":
    files = [l.strip() for l in open("/fsx/chris.beckham/gsolve_data/s1000/shots.txt") if l.strip()][::3]
    with ProcessPoolExecutor(6) as ex:
        out = [d for d in ex.map(one, files, chunksize=4) if d is not None]
    D = pd.concat(out, ignore_index=True); D.to_parquet("mast_h.parquet")
    T = pd.read_parquet("d3d/train_hist.parquet")
    for k in zfeat.HFEATS:
        lo, hi = np.nanpercentile(T[f"zth_{k}"], [1, 99]); v = D[f"z_{k}"]
        print(f"{k}: DIII-D 1-99% [{lo:.3f}, {hi:.3f}] median {np.nanmedian(T[f'zth_{k}']):.3f} | MAST median {np.nanmedian(v):.3f} "
              f"5-95% {np.nanpercentile(v, [5, 95]).round(3)} outside {np.mean((v < lo) | (v > hi)):.3f}")
    print(len(D), D.shot.nunique())
