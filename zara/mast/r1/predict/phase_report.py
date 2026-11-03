"""Label-free real-MAST boundary report by shot phase (the evaluation used for every intrusion-fix candidate).

Per frame (every --every-th): scorer-style LCFS of the predicted map (ch2/scal eqgeom, machine limiter mask) -> centre-column
intrusion flag (flux at R 0.2 m between axis and separatrix flux), inner / outer midplane LCFS radius, axis; Thomson 30 eV
inner / outer edges and the Te-peak radius of the same frame. Summary by fifth of the shot and overall.
    python phase_report.py <preds dir> [--stems file] [--every 2] [--save out.npy]"""
import argparse
import math
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
_G = None
EVERY = 2
TE_EDGE = 30.0


def _init():
    global _G
    R, Z, M = grid("MAST")
    _G = G.Grid(R, Z, M)


def edges(R, te):
    ok = np.isfinite(te) & (te > TE_EDGE)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    pk = R[np.nanargmax(np.where(np.isfinite(te), te, -1))]
    return float(R[ok].min()), float(R[ok].max()), float(pk)


def one(f):
    p = np.load(f)["psi"]; c = np.load(CACHE / f.name)
    itf, ip = np.abs(c["itf"].astype(float)), np.abs(c["ip"].astype(float))
    T = len(p); Rts = c["ts_R"].astype(float); rmin = float(np.nanmin(Rts))
    rows = []
    for k in range(0, T, EVERY):
        r = G.frame_physics(p[k].astype(np.float64), _G, MU0 * 24 * itf[min(k, len(itf) - 1)] / (2 * math.pi),
                            ip[min(k, len(ip) - 1)], None, levels_q=(0.95,), with_li=False, limiter=(0.2, 1.0))
        ok = r.get("ok", 0)
        g = lambda key: float(r.get(key, np.nan)) if ok else np.nan
        tin, tout, tpk = edges(Rts, c["ts_Te"][k].astype(float))
        rows.append([g("limited"), g("R_mid_in"), g("R_mid_out"), g("R_axis"), g("Z_axis"), tin, tout, tpk,
                     float(tin <= rmin + 1e-3) if np.isfinite(tin) else np.nan, k / max(T - 1, 1), float(ok)])
    return rows


def summary(v, label=""):
    lim, rin, rout, rax, zax, tin, tout, tpk, sat, tf, ok = v.T
    def med(a):
        a = a[np.isfinite(a)]
        return (np.median(a), np.percentile(a, 25), np.percentile(a, 75)) if len(a) else (np.nan,) * 3
    din = np.where(sat == 0, rin - tin, np.nan)          # inner edge only where Thomson resolves it (not saturated at the 1st chord)
    dout, dax = rout - tout, rax - tpk
    a, b, cq = med(din); d, e, fq = med(dout); g_, h, i_ = med(dax)
    return (f"{label:10s} n {len(v):6d} | intrusion {np.nanmean(lim):5.1%} | inner-Te edge (resolved {np.mean(sat == 0):4.0%}) "
            f"{100*a:+5.1f} [{100*b:+5.1f},{100*cq:+5.1f}] cm | outer {100*d:+5.1f} [{100*e:+5.1f},{100*fq:+5.1f}] cm | "
            f"axis-Te peak {100*g_:+5.1f} [{100*h:+5.1f},{100*i_:+5.1f}] cm | Z sd {100*np.nanstd(zax):4.1f} cm | LCFS ok {np.mean(ok):.1%}")


def main():
    global EVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path); ap.add_argument("--stems", type=Path, default=None)
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); EVERY = a.every
    d = a.preds / "preds" if (a.preds / "preds").exists() else a.preds
    files = sorted(d.glob("*.npz"))
    if a.stems:
        keep = set(a.stems.read_text().split()); files = [f for f in files if f.stem in keep]
    with Pool(a.workers, initializer=_init) as pool:
        v = np.array([r for rs in pool.map(one, files, chunksize=4) for r in rs], float)
    if a.save:
        np.save(a.save, v)
    print(f"== {d} ({len(files)} shots, every {EVERY})")
    print(summary(v, "all"))
    for i in range(5):
        m = (v[:, 9] >= i / 5) & (v[:, 9] <= (i + 1) / 5 if i == 4 else v[:, 9] < (i + 1) / 5)
        print(summary(v[m], f"fifth {i + 1}"))


if __name__ == "__main__":
    main()
