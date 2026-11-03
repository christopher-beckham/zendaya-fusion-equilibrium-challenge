#!/usr/bin/env python3
"""Ip-history features (zfeat.hist_features) for the DIII-D tables -> d3d/<split>_hist.parquet (shot, frame, h_* oracle-
edge tau, hth_* Thomson-edge tau)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_fixes import fix_d3d_ip_times  # noqa: E402
import zfeat  # noqa: E402
DATA = Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/diii_d_train")
HERE = Path(__file__).resolve().parent


def one(args):
    stem, rows = args
    r = pq.read_table(DATA / f"{stem}.parquet", columns=["magnetics_plasma_current", "magnetics_plasma_current_times",
                                                         "magnetics_time", "efit_times"]).to_pylist()[0]
    t = fix_d3d_ip_times(r); ip = np.asarray(r["magnetics_plasma_current"], np.float64)
    out = []
    for fr, tm, tav, a, tavt, at in rows:
        h = zfeat.hist_features(tm, t, ip, zfeat.tau_cd(tav, a))
        ht = zfeat.hist_features(tm, t, ip, zfeat.tau_cd(tavt, at)) if np.isfinite(tavt) and np.isfinite(at) else {k: np.nan for k in zfeat.HFEATS}
        out.append(dict(shot=stem, frame=fr, **{f"z_{k}": v for k, v in h.items()}, **{f"zth_{k}": v for k, v in ht.items()}))
    return out


if __name__ == "__main__":
    for sp in sys.argv[1:]:
        d = pd.read_parquet(HERE / "d3d" / f"{sp}3.parquet")
        a_t = 0.5 * (d.R_out_th - d.R_in_th)
        jobs = [(s, list(zip(g.frame, g.time, g.z_tav, g.a, g.zth_tav, a_t[g.index]))) for s, g in d.groupby("shot")]
        with ProcessPoolExecutor(8) as ex:
            res = [x for o in ex.map(one, jobs, chunksize=4) for x in o]
        H = pd.DataFrame(res); H.to_parquet(HERE / "d3d" / f"{sp}_hist.parquet")
        print(sp, len(H), H.describe().round(3).to_string())
