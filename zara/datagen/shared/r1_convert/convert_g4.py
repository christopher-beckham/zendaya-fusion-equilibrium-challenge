"""gen4 case sets (challenge MAST schema + synthetic efit_* labels) -> upegs multi-frame training records (machine MASTG4).

    python convert_g4.py --set /fsx/zara.ercan/more_data/m2/gen4_set20000 --name mast_g4s20000 --split train --prefix s20k \
        --machine MASTG4 --workers 32

Inputs are processed exactly as the real MAST test shots are (c2/preprocess_mast.py: coil / Ip / TF window means at the
labelled times, core Thomson aligned and taken from the nearest profile within +-6 ms, edge Thomson likewise), so a network
trained on these records sees the same input pipeline at inference on real MAST.
Record /fsx/zara.ercan/cache/ch2/g4/multi/<machine>/src/<prefix>_<case>.npz (format of shots.synth_shot + m2/gen/convert_seq.py):
    t (T,) ms, I (T,13) A, ip (T,) A, itf (T,) A per turn, ts_pts (146,2) (core + edge radii at Z = 0, as shots.mast_shot),
    ts_te (T,146) keV, ts_ne (T,146) 1e19 m^-3, psi (T,65,65) f32 (FreeGSNKE psi, axis = maximum = MAST stored convention),
    q95 / betaN / li / beta_p (T,), lcfs_r / lcfs_z (T,n), cons (T,7) scorer functionals of the true psi (limiter mask), lcfs64,
    origin (origin real file), limited (T,) bool (recipe v3 inner-wall-limited frames).
Only the synthetic labels of these files are read; no MAST EFIT data exists in them (Rule 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/fsx/zara.ercan/fusion-equilibrium-challenge-starter")
sys.path.insert(0, str(ROOT / "my_experiments" / "m2" / "gen"))
sys.path.insert(0, str(ROOT / "my_experiments" / "c2"))
import convert_m2 as C                                     # noqa: E402  (static grid / limiter mask, LS = local scorer)
import preprocess_mast as PM                               # noqa: E402

OUT_ROOT = Path("/fsx/zara.ercan/cache/ch2/g4/multi")
REAL_TS = False


def one(args):
    f, out = args
    if out.exists():
        return {"stem": out.stem, "skip": True}
    st = C.static()
    row = pd.read_parquet(f).iloc[0].to_dict()
    te = np.asarray(row["efit_times"], dtype=np.float64)
    tm = np.asarray(row["magnetics_time"], dtype=np.float64)
    I = np.column_stack([PM.window_mean(tm, np.asarray(row[c], dtype=np.float64), te, 1.0) for c in PM.COLS]) * 1e3
    ip = PM.window_mean(tm, np.asarray(row["magnetics_plasma_current"], dtype=np.float64), te, 1.0) * 1e3
    itf = np.abs(I[:, PM.COLS.index("magnetics_tf_current")])
    R, Te, ne = PM.align_mast_thomson_core(row)
    Te = np.where(Te > 0, Te, np.nan); ne = np.where(ne > 0, ne, np.nan)
    ts_Te, _ = PM._ts_at(row["thomson_core_times"], Te, te, PM.TS_WINDOW_MS)
    ts_ne, _ = PM._ts_at(row["thomson_core_times"], ne, te, PM.TS_WINDOW_MS)
    eR = np.asarray(row["thomson_edge_spatial"], dtype=np.float64)
    try:
        eTe = np.stack([np.asarray(p, dtype=np.float64) for p in row["thomson_edge_Te"]])
        ene = np.stack([np.asarray(p, dtype=np.float64) for p in row["thomson_edge_ne"]])
        eTe = np.where(eTe > 0, eTe, np.nan); ene = np.where(ene > 0, ene, np.nan)
        te_Te, _ = PM._ts_at(row["thomson_edge_times"], eTe, te, PM.TS_WINDOW_MS)
        te_ne, _ = PM._ts_at(row["thomson_edge_times"], ene, te, PM.TS_WINDOW_MS)
    except Exception:  # noqa: BLE001
        te_Te = te_ne = np.full((len(te), len(eR)), np.nan)
    n_real = 0
    if REAL_TS:
        # ch2/g4 --real-thomson: the network input is the ORIGIN real shot's Thomson at the same EFIT time (the generator built the
        # label from that profile); frames without a matching real time keep the synthetic Thomson
        o = json.loads(row["synth_meta"])["case"]["file"].replace(".parquet", ".npz")
        r = np.load(os.path.join("/fsx/zara.ercan/cache/mast/mast_public_test", o))
        tr = r["t"].astype(np.float64)
        ts_Te, ts_ne, te_Te, te_ne = ts_Te.copy(), ts_ne.copy(), np.array(te_Te, dtype=np.float64), np.array(te_ne, dtype=np.float64)
        for k, tk in enumerate(te):
            j = int(np.argmin(np.abs(tr - tk)))
            if abs(tr[j] - tk) <= 0.6:
                ts_Te[k], ts_ne[k] = r["ts_Te"][j], r["ts_ne"][j]
                te_Te[k], te_ne[k] = r["te_Te"][j], r["te_ne"][j]
                n_real += 1
    Rall = np.concatenate([R, eR])
    pts = np.stack([Rall, np.zeros_like(Rall)], 1)
    ts_te = np.concatenate([ts_Te, te_Te], 1) / 1e3
    ts_ne = np.concatenate([ts_ne, te_ne], 1) / 1e19
    psi = np.stack([np.asarray(np.stack(p), dtype=np.float64) for p in row["efit_psirz"]])        # (T, nZ, nR)
    T = len(te)
    assert psi.shape == (T, len(st["gZ"]), len(st["gR"])), psi.shape
    ref = C.LS.build_reference(psi.astype(np.float32), st["gR"], st["gZ"], st["mask"], st["mask_f"])
    cons = np.where(ref[2], ref[1], np.nan)
    c64 = np.full((T, 64, 2), np.nan)
    for k in range(T):
        if ref[0][k] is not None:
            cc = ref[0][k]; c64[k] = cc[np.linspace(0, len(cc), 64, endpoint=False).astype(int)]
    n = max(len(r) for r in row["efit_lcfs_r"])
    lr = np.full((T, n), np.nan); lz = np.full((T, n), np.nan)
    for k in range(T):
        a = np.asarray(row["efit_lcfs_r"][k], float); b = np.asarray(row["efit_lcfs_z"][k], float)
        lr[k, :len(a)] = a; lz[k, :len(b)] = b
    meta = json.loads(row["synth_meta"])
    fr = [g for g in json.loads(row["synth_frames"]) if g.get("status") == "ok"]
    lim = np.array([bool(g.get("limited", False)) for g in fr]) if len(fr) == T else np.zeros(T, bool)
    lab = lambda c: np.asarray(row[c], dtype=np.float64)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, t=te, I=I, ip=ip, itf=itf, ts_pts=pts, ts_te=ts_te, ts_ne=ts_ne, psi=psi.astype(np.float32),
             q95=lab("efit_q95"), betaN=lab("efit_beta_n"), li=lab("efit_li"), beta_p=lab("efit_beta_p"),
             lcfs_r=lr, lcfs_z=lz, cons=cons.astype(np.float32), lcfs64=c64.astype(np.float32),
             origin=np.asarray(str(meta.get("case", {}).get("file", ""))), limited=lim)
    os.replace(tmp, out)
    core_ok = np.isfinite(ts_Te).sum(1) >= 20
    return {"stem": out.stem, "T": T, "n_real_ts": n_real, "origin": str(meta.get("case", {}).get("file", "")), "split": meta.get("case", {}).get("split"), "version": meta.get("version"), "recipe": meta.get("recipe"),
            "cons_ok": float(np.isfinite(cons).all(1).mean()), "ts_ok": float(core_ok.mean()), "limited": int(lim.sum()),
            "ip_med": float(np.nanmedian(ip)), "q95_med": float(np.nanmedian(lab("efit_q95")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--name", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--prefix", required=True); ap.add_argument("--machine", default="MASTG4")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))); ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--real-thomson", action="store_true", help="Thomson input from the origin real shot at the same times")
    a = ap.parse_args()
    global REAL_TS
    REAL_TS = a.real_thomson
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    files = sorted(glob.glob(os.path.join(a.set, f"{a.name}_{a.split}", "*.parquet")))[: a.max or None]
    out = OUT_ROOT / a.machine
    (out / "src").mkdir(parents=True, exist_ok=True)
    st = C.static()
    if not (out / "spec.npz").exists():
        np.savez(out / "spec.npz", name=a.machine, grid_R=st["gR"], grid_Z=st["gZ"], columns=np.array(PM.COLS), wall=st["lim"],
                 tf_turns=24.0, note="spec = machine.multi_spec (mast_spec x0.5) for M2_MACHINES")
    jobs = [(f, out / "src" / f"{a.prefix}_{os.path.basename(f).split('_')[-1][:-8]}.npz") for f in files]
    print(f"{len(jobs)} files -> {out}/src ({a.workers} workers)", flush=True)
    metas = []
    with Pool(a.workers) as p:
        for i, m in enumerate(p.imap_unordered(one, jobs, chunksize=8), 1):
            metas.append(m)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    new = [m for m in metas if not m.get("skip")]
    if new:
        T = np.array([m["T"] for m in new])
        summ = {"set": a.set, "split": a.split, "prefix": a.prefix, "records": len(new), "frames": int(T.sum()),
                "T_hist": np.bincount(T).tolist(), "cons_ok": float(np.mean([m["cons_ok"] for m in new])),
                "ts_ok": float(np.mean([m["ts_ok"] for m in new])), "limited_frames": int(sum(m["limited"] for m in new)),
                "versions": sorted({str(m["version"]) for m in new}), "splits": sorted({str(m["split"]) for m in new}),
                "origins": len({m["origin"] for m in new})}
        (out / f"check_{a.prefix}_{a.split}.json").write_text(json.dumps({"summary": summ, "records": new}, indent=0))
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
