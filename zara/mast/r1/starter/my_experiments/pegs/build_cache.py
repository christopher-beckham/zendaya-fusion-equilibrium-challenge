"""First-pass equilibria + embeddings for every frame of one machine/split -> per-shot npz files.

    python build_cache.py --machine DIII-D --split train [--shard 0 --n-shards 3] [--max-shots 20 --check]
    python build_cache.py --machine MAST --split test
    python build_cache.py --machine DIII-D --split train --subset train --stride 2 --tag _s2   (training shots, every 2nd frame)

Output $R1_ROOT/cache/pegs/<machine>_<split>/<stem>.npz:
    X (T, F) features (embed.NAMES), tok (T, N, 8) f16, tmask (T, N), psin_fp (T, N) f16,
    Jfp (T, K) f16 first-pass cell currents / Ip on the symmetrised domain, fp_* (T,) first-pass geometry,
    I (T, C), ip, itf, t (T,), ts_pts (N, 2), ts_te / ts_ne (T, N) f16 (keV, 1e19 m^-3), fidx (T,) frame index in
    the full shot (stride > 1 keeps every stride-th frame; time features use the kept frames)
--check (DIII-D train) prints first-pass axis / shape errors against EFIT for the processed shots.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed as E                       # noqa: E402
import shots as S                       # noqa: E402
from machine import d3d, mast           # noqa: E402
from physics import GSLayer             # noqa: E402

OUT = Path(f"{R1_ROOT}/cache/pegs")
FP_KEYS = ["R_axis", "Z_axis", "psi_axis", "psi_b", "R_geo", "Z_geo", "a", "kappa", "tri_top", "tri_bot", "lp",
           "volume", "li", "res", "is_x", "xpt_R", "xpt_Z"]
EXTRA_KEYS = ["rung"]          # stored for diagnostics, not loaded as a feature


def machine_for(name, sol_factor=1.0):
    return d3d() if name == "DIII-D" else mast(sol_factor)


def cache_dir(machine, split, tag=""):
    return OUT / f"{machine.replace('-', '').lower()}_{split}{tag}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["DIII-D", "MAST"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--sol-factor", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--subset", default="all", choices=["all", "train", "holdout"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--p-share", type=float, default=E.P_SHARE)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--no-ladder", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    E.P_SHARE = args.p_share
    m = machine_for(args.machine, args.sol_factor)
    gs = GSLayer(m, dev)
    out = cache_dir(args.machine, args.split, args.tag)
    out.mkdir(parents=True, exist_ok=True)
    stems = S.stems(args.machine, args.split)
    if args.subset != "all":
        from data import holdout_split
        tr, ho = holdout_split()
        stems = tr if args.subset == "train" else ho
    stems = stems[: args.max_shots or None]
    stems = stems[args.shard::args.n_shards]
    todo = [s for s in stems if not (out / f"{s}.npz").exists()]
    load0 = (lambda s: S.d3d_shot(s, args.split, labels=False)) if args.machine == "DIII-D" else (lambda s: S.mast_shot(s, args.split))
    def load(s):
        r = load0(s)
        T = len(r["ip"])
        k = np.arange(0, T, args.stride)
        for key, v in list(r.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T and key != "ts_pts":
                r[key] = v[k]
        r["fidx"] = k
        return r
    print(f"{m.name} {args.split}: {len(stems)} shots in shard {args.shard}/{args.n_shards}, {len(todo)} to do; "
          f"K={gs.K} cells", flush=True)
    t0 = time.time()
    chk = []
    group, nfr, done = [], 0, 0
    def flush(group):
        recs = [load(s) for s in group]
        T = [len(r["ip"]) for r in recs]
        nch = max(len(r["ts_pts"]) for r in recs)
        def padp(r):
            p = np.full((nch, 2), np.nan); p[: len(r["ts_pts"])] = r["ts_pts"]; return p
        def padv(v):
            o = np.full((len(v), nch), np.nan); o[:, : v.shape[1]] = v; return o
        cat = lambda f: np.concatenate([f(r) for r in recs])
        tt = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        I = tt(cat(lambda r: r["I"])); ip = tt(cat(lambda r: r["ip"])).clamp(min=1e4); itf = tt(cat(lambda r: r["itf"]))
        pts = tt(cat(lambda r: np.broadcast_to(padp(r), (len(r["ip"]), nch, 2))))
        te = tt(cat(lambda r: padv(r["ts_te"]))); ne = tt(cat(lambda r: padv(r["ts_ne"])))
        res = {}
        for i in range(0, len(ip), args.batch):
            sl = slice(i, i + args.batch)
            fp = E.first_pass(gs, I[sl], ip[sl], n_iter=args.n_iter, omega=args.omega, ladder=not args.no_ladder)
            X, tok, tmask, psin, aux = E.frame_features(gs, fp, ip[sl], itf[sl], pts[sl], te[sl], ne[sl])
            part = {"X": X, "tok": tok, "tmask": tmask, "psin_fp": psin, "Jfp": fp["J"] / ip[sl][:, None],
                    **{f"fp_{k}": fp[k].float() for k in FP_KEYS + EXTRA_KEYS}, **{f"aux_{k}": v for k, v in aux.items()}}
            for k, v in part.items():
                res.setdefault(k, []).append(v.cpu().numpy())
        res = {k: np.concatenate(v) for k, v in res.items()}
        o = 0
        idx = {n: j for j, n in enumerate(E.NAMES)}
        for r, s, n in zip(recs, group, T):
            sl = slice(o, o + n); o += n
            X = res["X"][sl].copy()
            tf = E.time_features(r["t"], {k[4:]: res[k][sl] for k in res if k.startswith("aux_")})
            for k, v in tf.items():
                X[:, idx[k]] = v
            nc = len(r["ts_pts"])
            tok = np.clip(res["tok"][sl][:, :nc], -1e4, 1e4)                  # float16 range
            d = {"X": X.astype(np.float32), "tok": tok.astype(np.float16), "tmask": res["tmask"][sl][:, :nc] & np.isfinite(tok).all(-1),
                 "psin_fp": res["psin_fp"][sl][:, :nc].astype(np.float16), "Jfp": res["Jfp"][sl].astype(np.float16),
                 **{k: res[k][sl].astype(np.float32) for k in res if k.startswith("fp_")},
                 "I": r["I"].astype(np.float32), "ip": r["ip"].astype(np.float32), "itf": r["itf"].astype(np.float32),
                 "t": r["t"].astype(np.float32), "sign": np.float32(r["sign"]), "ts_pts": r["ts_pts"].astype(np.float32),
                 "ts_te": np.clip(r["ts_te"], 0, 1e3).astype(np.float16), "ts_ne": np.clip(r["ts_ne"], 0, 1e4).astype(np.float16),
                 "fidx": r["fidx"].astype(np.int32)}
            tmp = out / f"{s}.tmp.npz"
            np.savez(tmp, **d)
            os.replace(tmp, out / f"{s}.npz")
            if args.check and args.machine == "MAST":
                chk.append(np.stack([d["fp_R_axis"], d["fp_Z_axis"], d["fp_res"], d["fp_a"], d["fp_kappa"], d["fp_rung"]], 1))
            if args.check and args.machine == "DIII-D":
                g = np.load(S.D3D_CACHE[args.split] / f"{s}.npz")
                chk.append(np.stack([d["fp_R_axis"] - g["r_axis"][r["fidx"]], d["fp_Z_axis"] - g["z_axis"][r["fidx"]], d["fp_res"]], 1))
        return len(recs)
    for s in todo:
        group.append(s)
        nfr += 1
        if len(group) >= 8:
            done += flush(group); group = []
            if done % 200 < 8:
                print(f"  {done}/{len(todo)} shots {time.time() - t0:.0f}s", flush=True)
    if group:
        done += flush(group)
    print(f"done {done} shots in {time.time() - t0:.0f}s", flush=True)
    if chk and args.machine == "MAST":
        c = np.concatenate(chk)
        q = lambda v: np.round(np.nanpercentile(v, [5, 50, 95]), 3).tolist()
        print(f"MAST first pass over {len(c)} frames: R_axis {q(c[:, 0])} Z_axis {q(c[:, 1])} a {q(c[:, 3])} kappa {q(c[:, 4])} "
              f"res median {np.median(c[:, 2]):.1e}; res<1e-2 {np.mean(c[:, 2] < 1e-2):.3f}; a<0.2 m {np.mean(c[:, 3] < 0.2):.3f}; "
              f"rungs {[round(float(np.mean(c[:, 5] == r)), 3) for r in range(3)]}", flush=True)
        chk = []
    if chk:
        c = np.concatenate(chk)
        f = np.isfinite(c).all(1)
        c = c[f]
        print(f"first pass vs EFIT over {len(c)} frames: dR_axis mean {1e3 * c[:, 0].mean():.0f} mm rms {1e3 * np.sqrt((c[:, 0] ** 2).mean()):.0f}; "
              f"dZ_axis mean {1e3 * c[:, 1].mean():.0f} rms {1e3 * np.sqrt((c[:, 1] ** 2).mean()):.0f} mm; "
              f"res median {np.median(c[:, 2]):.1e}, frac<1e-3 {(c[:, 2] < 1e-3).mean():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
