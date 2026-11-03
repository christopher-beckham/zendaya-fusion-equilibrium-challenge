#!/usr/bin/env python3
"""Height rule: z_bal (gsolve balance height from coils + edges, as in gen_d3d) for DIII-D train/val frames, and the
fit EFIT Z_axis ~ s * z_bal on TRAIN.  -> d3d/<split>_zbal.npy, height_fit.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gsolve
HERE = Path(__file__).resolve().parent
COILS = ["F1A", "F1B", "F2A", "F2B", "F3A", "F3B", "F4A", "F4B", "F5A", "F5B", "F6A", "F6B", "F7A", "F7B", "F8A", "F8B",
         "F9A", "F9B", "ECOILA"]
m = gsolve.load_machine("diii-d")
def zbal(df, rin="R_in", rout="R_out"):
    I = df[[f"c_{c}" for c in COILS]].values * df.sign.values[:, None]
    ext = gsolve.coil_flux(m, I).reshape(len(df), -1)
    Rc = 0.5 * (df[rin].values + df[rout].values); a = np.clip(0.5 * (df[rout].values - df[rin].values), 0.4, 0.8)
    z = np.concatenate([gsolve.balance_height(m, ext[j:j + 256], Rc[j:j + 256], a[j:j + 256]) for j in range(0, len(df), 256)])
    return np.nan_to_num(z, nan=0.0)
rin, rout, sfx = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("R_in", "R_out", "")
rep = {}
for sp in ("train", "val"):
    df = pd.read_parquet(HERE / "d3d" / f"{sp}{sfx}.parquet")
    z = zbal(df, rin, rout); np.save(HERE / "d3d" / f"{sp}{sfx}_zbal.npy", z)
    y = df.Z_ax.values; ok = np.isfinite(y)
    s = float(np.sum(z[ok] * y[ok]) / np.sum(z[ok] ** 2))
    A = np.c_[z[ok], np.ones(ok.sum())]; c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    r2 = lambda p: float(1 - np.sum((y[ok] - p) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    rep[sp] = dict(n=int(ok.sum()), slope0=s, r2_slope0=r2(s * z[ok]), slope=float(c[0]), icpt=float(c[1]), r2_lin=r2(A @ c),
                   corr=float(np.corrcoef(z[ok], y[ok])[0, 1]), zbal_sd=float(z.std()), zax_sd=float(y[ok].std()),
                   resid_sd=float(np.std(y[ok] - s * z[ok])))
print(json.dumps(rep, indent=1)); (HERE / f"height_fit{sfx}.json").write_text(json.dumps(rep, indent=1))
