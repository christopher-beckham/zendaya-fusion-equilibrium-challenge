"""Label-free per-frame features for calibration-target rules (same code on holdout and public-test frames).

build(...) takes the scorer functionals / LCFS extremes of the mean map (x0c (N, 7), x0g (N, 6)), of the member maps
(mc (M, N, 7), mg (M, N, 6)), the head outputs (hc (H, N, 7), hg (H, N, 6), NaN where a head has no geo output) and the frame
metadata (shot, k, T, t, ip, bt), and returns a dict with x0, xh, xm, sh, sm (map, head mean, member mean, head sd, member sd
per term; li NaN -> 4.9 = the li guard) and the feature matrix F.
"""
from __future__ import annotations

import numpy as np

LI_GUARD = 4.9


def build(x0c, x0g, mc, mg, hc, hg, shot, k, T, t, ip, bt):
    x0 = np.array(x0c, np.float64, copy=True)
    x0[:, 6] = np.where(np.isfinite(x0[:, 6]), x0[:, 6], LI_GUARD)
    mc = np.array(mc, np.float64, copy=True)
    mc[:, :, 6] = np.where(np.isfinite(mc[:, :, 6]), mc[:, :, 6], LI_GUARD)
    xm, sm = np.nanmean(mc, 0), np.nanstd(mc, 0)
    xm = np.where(np.isfinite(xm), xm, x0)
    xh, sh = hc.mean(0), hc.std(0)
    aip = np.abs(np.asarray(ip, np.float64))
    ipmax = np.zeros_like(aip)
    dip = np.zeros_like(aip)
    tt = np.asarray(t, np.float64)
    for s in np.unique(shot):
        j = np.nonzero(shot == s)[0]
        ipmax[j] = np.maximum.accumulate(aip[j])
        dip[j] = np.gradient(aip[j], tt[j]) if len(j) > 1 else 0.0
    if np.isfinite(hg).any():
        hgm = np.nanmean(hg, 0)
    else:
        hgm = np.full_like(x0g, np.nan)
    with np.errstate(all="ignore"):
        F = np.c_[x0, xh - x0, xm - x0, sh, sm, aip, aip / np.maximum(ipmax, 1e-3), dip, k / T, T, bt, hgm - x0g,
                  np.nanstd(mg, 0), np.nanmean(mg, 0) - x0g]
    return {"x0": x0, "xh": xh, "xm": xm, "sh": sh, "sm": sm, "F": F, "aip": aip, "kT": k / T,
            "hgm": hgm, "gm": np.nanmean(mg, 0), "g0": np.asarray(x0g, np.float64)}


def from_kdata(d, hidx):
    m = d["meta"]
    return build(d["mean"]["cons"], d["mean"]["geo"], d["mem_cons"], d["mem_geo"], d["head_cons"][hidx], d["head_geo"][hidx],
                 m["shot"], m["k"], m["T"], m["t"], m["ip"], m["bt"])
