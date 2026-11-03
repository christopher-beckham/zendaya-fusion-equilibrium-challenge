"""Predict psi, q95, betaN with a trained pegs model -> <out>/preds/<stem>.npz (stored convention of the machine).

    python predict.py --run runs/v1 --machine DIII-D --split train --subset holdout --out runs/v1/ho
    python predict.py --run runs/v1 --machine MAST --split test --out runs/v1/mast [--tta 30]

Frames are predicted at every EFIT time (stride 1). betaN: kinetic (Thomson x learned ratio) where the core is
covered by Thomson, else the model's own pressure (--betaN kin|mag|auto). --tta N runs N steps of label-free
test-time refinement (tta.py) per batch, warm-started from the network.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
R1_ROOT = __import__("os").environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_cache import machine_for                 # noqa: E402
from data import FrameSet, holdout_split            # noqa: E402
from model import PegsNet, forward, forward_kwargs, pressure_profile, net_from_args, refine_inputs  # noqa: E402
from physics import GSLayer                         # noqa: E402
import shots as S                                   # noqa: E402
import embed as E                                  # noqa: E402

TIME_CONST = {"dlnip_tauR": 0.0, "dlnip_ew": 0.0, "ip_over_max": 1.0, "ddt_bz": 0.0, "ddt_br": 0.0, "ddt_quad": 0.0, "ddt_flux": 0.0}


def neutralize_time(b):
    """Unseen-machine rule: set the time-history embeddings to the values single-frame synthetic records carry (m2)."""
    X = b["X"].clone()
    for k, v in TIME_CONST.items():
        X[:, E.NAMES.index(k)] = v
    b["X"] = X
    return b
from train import model_scalars                     # noqa: E402


Q95_RANGE = (1.0, 30.0)        # physical bounds of the scalar outputs (any machine)
BETAN_RANGE = (0.0, 6.0)


def bound(v, rng):
    """Values outside (lo, hi] or non-finite -> the shot median of the in-range values (or the range midpoint if
    none). Returns the bounded array and the number of values replaced."""
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v) & (v > rng[0]) & (v <= rng[1])
    fill = float(np.median(v[ok])) if ok.any() else 0.5 * (rng[0] + rng[1])
    return np.where(ok, v, fill), int((~ok).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--machine", default="DIII-D")
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", default="all", choices=["all", "holdout", "ho100"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=-1, help="Picard steps (-1 = as trained)")
    ap.add_argument("--extra-steps", type=int, default=0)
    ap.add_argument("--betaN", default="auto", choices=["auto", "kin", "mag", "mean"], help="mean: average of the kinetic and magnetic routes where Thomson covers the core, else magnetic")
    ap.add_argument("--sol-factor", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--tta-lr", type=float, default=0.05)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--no-harm", action="store_true", help="drop the learned vacuum-harmonic correction")
    ap.add_argument("--harm-mode", default=None, choices=[None, "net", "off", "free0"],
                    help="TTA only: harmonics from the net / off / fitted label-free from zero (see tta.py)")
    ap.add_argument("--tta-prior", type=float, default=1.0)
    ap.add_argument("--tta-all", action="store_true", help="TTA also frees the Z-odd head components (default: even only)")
    ap.add_argument("--two-pass", type=int, default=-1, help="1/0: re-embed from the model's own equilibrium before the final pass (-1 = as trained)")
    ap.add_argument("--vstab", action="store_true", help="inference-time vertical stabilisation in every Picard step (rigid shift to zero "
                    "net vertical force from the external field; same rule on every machine)")
    ap.add_argument("--tta-w", default="", help="m2: label-free TTA loss weights, e.g. gs=1,ampere=1,ts=3,tsedge=3,kin=0.3 (unlisted: 1; names from train.PHYS_ALL)")
    ap.add_argument("--vc", action="store_true", help="m2: vertically controlled solve: per batch, the force-balance height z* is found by solve_vc with the network's initial profiles (no grad), then TTA / forward run with the current centroid pinned at z* (implies --vstab)")
    ap.add_argument("--vc-inner", type=int, default=10)
    ap.add_argument("--vc-outer", type=int, default=8)
    ap.add_argument("--vc-width", type=float, default=0.2)
    ap.add_argument("--vstab-safe", action="store_true", help="m2: vstab ignores frames without a force zero in the window and clamps the per-step shift (implies --vstab)")
    ap.add_argument("--no-qcorr", action="store_true", help="m2: drop the learned q95 correction head (pure field-line integral) on unseen machines")
    ap.add_argument("--vstab-rw", action="store_true", help="m2: vstab with the physical R weighting of the vertical force (implies --vstab)")
    ap.add_argument("--no-time", action="store_true", help="m2 rule for unseen machines: time-history embeddings set to the single-frame synthetic constants")
    ap.add_argument("--harm-rule", default="", choices=["", "trained"],
                    help="trained: apply the learned vessel-current (harmonic) correction only on machines the model was "
                         "trained on (those with learned coil gains in the checkpoint); overrides --no-harm")
    ap.add_argument("--passive", type=float, default=0.0, help="ch2/syn: add this multiple of the induced passive-current flux to the vacuum flux")
    ap.add_argument("--passive-dir", default=f"{R1_ROOT}/cache/ch2/syn/passive/MAST")
    ap.add_argument("--tta-sel", default="all", choices=["all", "divert"], help="ch2/syn: divert = TTA only on frames that are limited or violate the diverted condition")
    ap.add_argument("--sel-margin", type=float, default=0.0, help="ch2/syn: margin of the divert selection (fraction of the axis - X-point swing)")
    ap.add_argument("--divert", action="store_true", help="ch2/syn: evaluate the 'divert' label-free term (inner-wall gap) in TTA; weight via --tta-w divert=w")
    args = ap.parse_args()
    if args.divert:
        import train as _T
        _T.DIVERT = True
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(args.run / "model.pt", map_location=dev)
    targs = ck["args"]
    steps = targs["steps"] if args.steps < 0 else args.steps
    fkw = forward_kwargs(targs)
    if args.vstab or args.vstab_rw or args.vstab_safe or args.vc:
        fkw["vstab"] = True
    two_pass = (float(targs.get("two_pass", 0.0)) > 0) if args.two_pass < 0 else bool(args.two_pass)
    print(f"two-pass embedding: {two_pass}", flush=True)
    import embed as E
    E.P_SHARE = float(targs.get("p_share", 0.5))
    if not args.tag and targs.get("val_tag"):
        args.tag = targs["val_tag"] if args.machine == "DIII-D" and args.split == "train" else targs["val_tag"].replace("_s2", "")
    print(f"profile start share {E.P_SHARE}, cache tag '{args.tag}'", flush=True)
    print(f"forward options {fkw}, {steps} Picard steps (+{args.extra_steps})", flush=True)
    m = machine_for(args.machine, args.sol_factor)
    gs = GSLayer(m, dev)
    gs.vstab_rw = bool(args.vstab_rw)
    tta_weights = None
    if args.tta_w:
        from train import PHYS
        tta_weights = {k: 1.0 for k in PHYS}
        for kv in args.tta_w.split(","):
            k, v = kv.split("="); tta_weights[k] = float(v)
        print(f"TTA loss weights {tta_weights}", flush=True)
    gs.vstab_safe = bool(args.vstab_safe)
    if args.vstab_rw:
        print('vstab: R-weighted vertical force root', flush=True)
    net = net_from_args(targs, machines={k: v.shape[0] for k, v in ((k[6:], v) for k, v in ck["net"].items() if k.startswith("gains."))}).to(dev)
    net.load_state_dict(ck["net"]); net.eval()
    if args.harm_rule == "trained":
        args.no_harm = m.name.replace("-", "") not in net.gains
        print(f"harm rule 'trained': trained machines {list(net.gains)}; harmonics {'off' if args.no_harm else 'on'} for {m.name}", flush=True)
    if args.machine == "DIII-D" and args.split == "train":
        _, ho = holdout_split()
        stems = ho if args.subset == "holdout" else ho[:100] if args.subset == "ho100" else S.stems("DIII-D", "train")
    else:
        stems = S.stems(args.machine, args.split)
    stems = stems[: args.max_shots or None]
    od = args.out / "preds"
    od.mkdir(parents=True, exist_ok=True)
    (args.out / "stems.txt").write_text("\n".join(stems))
    t0 = time.time()
    stats = []
    n_fallback = 0
    n_vc_fail = 0
    n_bounded = [0, 0]
    n_sel_tot = [0, 0]
    n_pas_missing = [0]
    chunk = 64
    for c0 in range(0, len(stems), chunk):
        cs = stems[c0:c0 + chunk]
        fs = FrameSet(args.machine, args.split, cs, dev, stride=1, labels=False, tag=args.tag)
        if args.passive:
            # ch2/syn: flux of the induced passive-structure currents (fz/passive.py, from the measured full-rate coil / Ip
            # signals and the public MAST vessel geometry) added to the vacuum flux, x args.passive
            Gp = torch.tensor(np.load(Path(args.passive_dir) / "Gp.npy"), dtype=torch.float32, device=dev)
            Ivs = []
            for si, s_ in enumerate(cs):
                fpv = Path(args.passive_dir) / f"{s_}.npz"
                Iv = np.load(fpv)["Iv"].astype(np.float32) if fpv.exists() else None
                if Iv is None or len(Iv) != fs.T_full[si]:
                    n_pas_missing[0] += 1
                    Iv = np.zeros((fs.T_full[si], Gp.shape[0]), np.float32)
                Ivs.append(Iv[fs.keep[si]])
            fs.d["psi_add"] = args.passive * (torch.tensor(np.concatenate(Ivs), device=dev) @ Gp)
        if int(targs.get("feat2", 0) or 0):
            import feat2 as F2
            F2.extend(fs, gs)
        out = {k: [] for k in ("psi", "q95", "bk", "bm", "has", "gs", "R_axis", "Z_axis", "qcorr")}
        for i in range(0, len(fs), args.batch):
            b = fs.batch(torch.arange(i, min(len(fs), i + args.batch), device=dev))
            if args.no_time:
                b = neutralize_time(b)
            if two_pass:
                with torch.no_grad():
                    st1 = forward(net, gs, b, n_steps=steps, harm=not args.no_harm, **fkw)
                b = refine_inputs(gs, b, st1)
            fkw_b = fkw
            if args.vc:
                with torch.no_grad():
                    st0 = forward(net, gs, b, n_steps=steps, harm=not args.no_harm, **fkw)
                    Jv, Fz, okr = gs.solve_vc(st0['psi_ext'], b['ip'], st0['P'], st0['Q'], R0=st0['R0'], J0=st0['J'], n_inner=args.vc_inner, n_outer=args.vc_outer, n_final=args.vc_inner, omega=0.5, width=args.vc_width)
                    z_star = gs.centroid(Jv)[:, 1]
                    n_vc_fail += int((~okr).sum())
                fkw_b = dict(fkw, z_pin=z_star)
            sel_n = 0
            if args.tta and args.tta_sel != "all":
                # ch2/syn: TTA only on frames whose network solution is limited / violates the diverted-plasma condition
                from tta import refine
                import train as _T
                with torch.no_grad():
                    st = forward(net, gs, b, n_steps=steps, extra_steps=args.extra_steps, harm=not args.no_harm, **fkw_b)
                    p, ff = pressure_profile(st, gs)
                    sc = model_scalars(gs, st, b, p, ff)
                    viol, okd = _T.divert_violation(gs, st, b["fp_a"].clamp(min=0.05), margin=args.sel_margin)
                    sel = (okd & (viol > 0)) | (~st["is_x"])
                idx = torch.nonzero(sel)[:, 0]
                sel_n = len(idx)
                if sel_n:
                    Bn = b["ip"].shape[0]
                    bs = {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == Bn else v) for k, v in b.items()}
                    st2, sc2 = refine(net, gs, bs, steps, args.tta, args.tta_lr, weights=tta_weights, harm=not args.no_harm, harm_mode=args.harm_mode, prior=args.tta_prior, even_only=not args.tta_all, **fkw_b)
                    for k in ("psi", "J_next", "J", "R_axis", "Z_axis", "qcorr"):
                        if k in st and k in st2:
                            st[k] = st[k].detach().clone(); st[k][idx] = st2[k].detach()
                    for k in ("q95", "betaN_kin", "betaN_mag", "has_ts"):
                        sc[k] = sc[k].detach().clone(); sc[k][idx] = sc2[k].detach()
                n_sel_tot[0] += sel_n; n_sel_tot[1] += int(b["ip"].shape[0])
            elif args.tta:
                from tta import refine
                st, sc = refine(net, gs, b, steps, args.tta, args.tta_lr, weights=tta_weights, harm=not args.no_harm, harm_mode=args.harm_mode, prior=args.tta_prior, even_only=not args.tta_all, **fkw_b)
            else:
                with torch.no_grad():
                    st = forward(net, gs, b, n_steps=steps, extra_steps=args.extra_steps, harm=not args.no_harm, **fkw_b)
                    p, ff = pressure_profile(st, gs)
                    sc = model_scalars(gs, st, b, p, ff)
            with torch.no_grad():
                if args.no_qcorr and "qcorr" in st:
                    sc["q95"] = sc["q95"] / torch.exp(st["qcorr"])
                out["psi"].append((m.to_stored(st["psi"])).float().cpu().numpy())
                out["q95"].append(sc["q95"].cpu().numpy()); out["bk"].append(sc["betaN_kin"].cpu().numpy())
                out["bm"].append(sc["betaN_mag"].cpu().numpy()); out["has"].append(sc["has_ts"].cpu().numpy())
                gsr = ((st["J_next"] - st["J"]) ** 2).sum(1) / (st["J"] ** 2).sum(1).clamp(min=1e-12)
                out["gs"].append(gsr.cpu().numpy())
                out["R_axis"].append(st["R_axis"].cpu().numpy()); out["Z_axis"].append(st["Z_axis"].cpu().numpy())
                out["qcorr"].append((st["qcorr"] if "qcorr" in st else torch.zeros_like(st["R_axis"])).cpu().numpy())
        out = {k: np.concatenate(v) for k, v in out.items()}
        # never write non-finite predictions: fall back to the first-pass equilibrium for such frames
        badf = ~np.isfinite(out["psi"]).all(1)
        if badf.any():
            with torch.no_grad():
                bi = torch.tensor(np.nonzero(badf)[0], device=dev)
                bb = fs.batch(bi)
                fp = gs.vacuum(bb["I"]) + gs.plasma(bb["Jfp"] * bb["ip"][:, None])
                out["psi"][badf] = m.to_stored(fp).float().cpu().numpy()
            print(f"  {int(badf.sum())} frames with non-finite psi -> first-pass equilibrium", flush=True)
        for k in ("q95", "bk", "bm"):
            out[k] = np.where(np.isfinite(out[k]), out[k], np.nan)
        n_fallback += int(badf.sum())
        bN = {"kin": out["bk"], "mag": out["bm"], "auto": np.where(out["has"], out["bk"], out["bm"]),
              "mean": np.where(out["has"] & np.isfinite(out["bk"]), 0.5 * (np.nan_to_num(out["bk"]) + out["bm"]), out["bm"])}[args.betaN]
        o = 0
        for si, (s, T) in enumerate(zip(cs, fs.T)):
            sl = slice(o, o + T); o += T
            psi = out["psi"][sl].reshape(T, len(m.grid_Z), len(m.grid_R))
            q95, nq = bound(out["q95"][sl], Q95_RANGE)
            bNs, nb = bound(bN[sl], BETAN_RANGE)
            n_bounded[0] += nq; n_bounded[1] += nb
            assert np.isfinite(psi).all() and np.isfinite(q95).all() and np.isfinite(bNs).all(), s
            np.savez(od / f"{s}.npz", psi=psi.astype(np.float32), q95=q95.astype(np.float32), betaN=bNs.astype(np.float32),
                     gs_res=out["gs"][sl].astype(np.float32), R_axis=out["R_axis"][sl], Z_axis=out["Z_axis"][sl],
                     betaN_kin=out["bk"][sl].astype(np.float32), betaN_mag=out["bm"][sl].astype(np.float32),
                     qcorr=out["qcorr"][sl].astype(np.float32),
                     t=fs.times[si].astype(np.float32))
        stats.append(out["gs"])
        print(f"  {c0 + len(cs)}/{len(stems)} shots {time.time() - t0:.0f}s", flush=True)
    g = np.concatenate(stats)
    if args.vc:
        print(f"vc: no force sign change in {n_vc_fail} frames", flush=True)
    if args.passive:
        print(f"passive flux x{args.passive}: {n_pas_missing[0]} shots without passive currents (zeros)", flush=True)
    if n_sel_tot[1]:
        print(f"TTA selection ({args.tta_sel}): {n_sel_tot[0]} / {n_sel_tot[1]} frames ({n_sel_tot[0] / n_sel_tot[1]:.1%})", flush=True)
    print(f"done {len(stems)} shots in {time.time() - t0:.0f}s; {n_fallback} fallback frames; out-of-range scalars replaced: "
          f"q95 {n_bounded[0]}, betaN {n_bounded[1]}; GS residual median {np.median(g):.2e} p90 {np.quantile(g, 0.9):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
