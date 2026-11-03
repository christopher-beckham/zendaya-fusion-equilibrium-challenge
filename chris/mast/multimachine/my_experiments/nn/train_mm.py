"""Minimal training script for the bottom-up rebuild (PLAN_BOTTOM_UP.md in the main repo).

    python my_experiments/nn/train_mm.py --name NAME [options]

Base recipe: v5 features, physical decoder normalisation, flux MSE + gradient term + the q95/βN head on real
DIII-D. Every addition is a flag, off by default, added only once a screen has earned it:

  --decoder linear            ψ = coil picture + G·J + offset (lindec.py); default conv
  --lindec-ip-floor KA        learned current below |Ip| = KA (linear decoder; 150 earned DIII-D 0.946 -> 0.980)
  --lindec-exact-ip-mast      Σ J = −Ip exactly on MAST (linear decoder)
  --lambda-axis W             Thomson-centre axis pull on DIII-D and MAST batches (physics_loss.AxisPull),
                              with --axis-z-slack and --mast-axis-hi; --axis-vertical-only keeps only the term
                              that holds the axis near the midplane (no pull towards the Thomson peak)
  --synth TAG[:W[:F]]         labelled synthetic set (synth_cache.py), homogeneous batches (synth_data.py); F < 1
                              trains on that share of its training origin shots (nested subsets; held out unchanged)
  --thomson-noise S           Gaussian noise of S standard deviations on the Thomson-derived inputs of every
                              training batch (real and synthetic), so the network cannot lean on them too hard
  --lambda-fluxfn W           flux-function current loss (fluxfn_loss.py), with --fluxfn-pressure for the Thomson
                              pressure term; on DIII-D and, with --lambda-axis, the MAST batch
  --mast-mask {envelope,limiter}
                              MAST plasma region (mast_region.py; sets FEC_MAST_MASK for this run and its checks).
                              Default: FEC_MAST_MASK, else the envelope. "limiter" reads the limiter variants of the
                              MAST caches (lindec_G_mast_limiter.npz, features_*_limiter, synthetic tags *_limiter)

Checkpoint selection for MAST (synthetic-MAST S, the main selection metric; leaderboard rank correlation 0.86):
  --synthS-every K            every K epochs, score that epoch's checkpoint on Zara's held-out synthetic MAST cases
                              with the official scorer (main repo synth_S/ensemble_synth.py, CPU, in the background)
                              and keep the best one whose DIII-D S (score.py, so use K a multiple of --score-every)
                              is at least --d3d-floor as best_mast.pt; best_mast.json lists every scored epoch
  --synth-shape K=W,...       shape losses on labelled synthetic MAST frames (shape_loss.py; terms grad, jtor, axis, bnd,
                              li), added to each MAST set's flux loss; needs shape_targets.npz (synth_shape_targets.py)
  --ema-halflife H           from epoch --ema-start on, keep an exponential moving average of the weights with a
                              half-life of H epochs; evaluation, scoring and saved checkpoints use the averaged
                              weights (training continues on the raw ones; resume.pt keeps both)
  --init CKPT                 start from this checkpoint's weights (same feature set and architecture), e.g. to fine-tune
  --d3d-weight W              weight of the real DIII-D flux, slope and scalar losses (default 1). The MAST model's
                              DIII-D accuracy does not count (submissions take DIII-D from the Challenge 1 model), so a
                              MAST specialist can lower it; 0 trains on the synthetic sets only
  --synthS-metric NAME        column that ranks them (default g4_5000_noV1origins); --synthS-sets which sets to score

Logged every epoch: real DIII-D val (R²ψ, R²(q95, βN), ∇ψ RMSE), each synthetic set's train loss and held-out
R²ψ, and the label-free MAST monitor (mast_monitor.py). Score epochs dump the DIII-D score subset and run score.py;
with --mast-checks-every K the scorer-based MAST checks run in the background too. Checkpoints load with
model.load_checkpoint and predict.py like train.py's.

Reuses train.py's data loading, normalisation, evaluation and saving; the old train.py stays as the reference for
everything this script leaves out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SYNTHS = HERE.parents[2] / "starter" / "my_experiments" / "nn" / "synth_S"
sys.path.insert(0, str(HERE))
from common import CACHE, DATA_ROOT, IP_COL, append_jsonl, history_stats, robust_center_scale, score_subset  # noqa: E402
import features_v2  # noqa: E402
from greens import load_or_build_baseline, DEFAULT_FRAMES  # noqa: E402
from model import Net, grad_mse, precision_context  # noqa: E402
from train import (  # noqa: E402
    _dump_subset, _external, _rows, _save, _with_history, assert_disjoint, cached_stats, evaluate, fit_norm,
    load_split, scheduled_lr, split_key,
)

RUNS = HERE / "runs"
MASKS = HERE.parents[1] / "fusion_scoring" / "masks"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-shot", action="append", default=[],
                    help="drop a DIII-D training shot by id (e.g. d3d_shot_955c35c962: q95 0.17-0.95 and βN up to 14 throughout)")
    # schedule and optimiser
    ap.add_argument("--epochs", type=int, default=60, help="length of the learning-rate schedule")
    ap.add_argument("--stop-epoch", type=int, default=0, help="stop early, keeping the schedule of --epochs")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run from <run>/resume.pt (weights, optimiser, epoch, step), written every epoch")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--no-amp", action="store_true")
    # model
    ap.add_argument("--decoder", choices=["conv", "linear"], default="linear")
    ap.add_argument("--attn-decoder", action="store_true",
                    help="linear decoder: replace the conv decoder by a coordinate-query attention field (lindec.AttnField)")
    ap.add_argument("--output-prior", action="store_true", help="conv decoder only")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=3, help="residual blocks in the trunk")
    ap.add_argument("--seed-channels", type=int, default=256)
    ap.add_argument("--scalar-width", type=int, default=128)
    ap.add_argument("--scalar-depth", type=int, default=2)
    ap.add_argument("--feature-set", default="v5")
    ap.add_argument("--history-frames", type=int, default=0)
    ap.add_argument("--pictures", type=Path, default=HERE / "baseline_pictures_fitgain2.npz")
    ap.add_argument("--lindec-ip-floor", type=float, default=0.0)
    ap.add_argument("--lindec-exact-ip-mast", action="store_true")
    # losses
    ap.add_argument("--lambda-grad", type=float, default=0.3)
    ap.add_argument("--lambda-s", type=float, default=1.0, help="q95/βN head (Huber)")
    ap.add_argument("--lambda-axis", type=float, default=0.0)
    ap.add_argument("--axis-z-slack", type=float, default=0.1)
    ap.add_argument("--mast-axis-hi", type=float, default=0.03)
    ap.add_argument("--mast-axis-lo", type=float, default=-0.20,
                    help="MAST radial band lower edge, (R_axis − R_th) / a; −0.03 with hi 0.03 is a tight two-sided pull")
    ap.add_argument("--axis-vertical-only", action="store_true")
    ap.add_argument("--axis-mast-only", action="store_true",
                    help="apply the axis pull to MAST batches only; DIII-D frames are left to their labels (a vertical "
                         "slack of 0.1 a is tighter than real DIII-D axes, which reach +0.21 a)")
    ap.add_argument("--mast-batch-size", type=int, default=512)
    ap.add_argument("--mast-variant", default="_sol0.5", choices=["", "_sol0.5"])
    ap.add_argument("--mast-mask", choices=["envelope", "limiter"], default=None,
                    help="MAST plasma region (mast_region.py); default: $FEC_MAST_MASK, else envelope")
    ap.add_argument("--synth", action="append", default=[], metavar="TAG[:WEIGHT[:FRAC]]")
    ap.add_argument("--synth-batch", type=int, default=64)
    ap.add_argument("--thomson-noise", type=float, default=0.0)
    ap.add_argument("--thomson-dropout", type=float, default=0.0,
                    help="share of training frames (per history slot) whose Thomson inputs are marked missing")
    ap.add_argument("--lambda-fluxfn", type=float, default=0.0)
    ap.add_argument("--fluxfn-pressure", type=float, default=0.0)
    ap.add_argument("--fluxfn-frames", type=int, default=512)
    # logging
    ap.add_argument("--score-every", type=int, default=5)
    ap.add_argument("--mast-checks-every", type=int, default=0)
    ap.add_argument("--no-mast-monitor", action="store_true")
    ap.add_argument("--synthS-every", type=int, default=0)
    ap.add_argument("--synthS-sets", default="g4_5000")
    ap.add_argument("--synthS-metric", default="g4_5000_noV1origins")
    ap.add_argument("--d3d-floor", type=float, default=0.85)
    ap.add_argument("--ema-halflife", type=float, default=0.0)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--d3d-weight", type=float, default=1.0)
    ap.add_argument("--synth-shape", default="", metavar="TERM=W,...")
    ap.add_argument("--ema-start", type=int, default=10)
    a = ap.parse_args(argv)
    # Fixed choices of the rebuild, kept in the checkpoint's args so model.load_checkpoint and predict.py work.
    a.decoder_norm, a.lindec_erode, a.lindec_act, a.lindec_exact_ip = "physical", 0, "softplus", False
    if a.decoder == "linear" and a.output_prior:
        raise SystemExit("--output-prior is for the conv decoder")
    if (a.lindec_ip_floor or a.lindec_exact_ip_mast or a.lambda_fluxfn) and a.decoder != "linear":
        raise SystemExit("--lindec-* options and --lambda-fluxfn need --decoder linear")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    # The MAST region is fixed before anything reads it (geometry, G, MAST inputs, synthetic sets, losses, monitor).
    import mast_region
    if args.mast_mask is not None:
        if mast_region.explicit() and mast_region.name() != args.mast_mask:
            raise SystemExit(f"--mast-mask {args.mast_mask} contradicts {mast_region.ENV_VAR}={mast_region.name()}")
        mast_region.select(args.mast_mask)
    args.mast_mask = mast_region.name()
    print(f"MAST region: {args.mast_mask}", flush=True)
    linear = args.decoder == "linear"
    device = torch.device(args.device)
    last_epoch = args.stop_epoch or args.epochs
    torch.manual_seed(args.seed)

    # --- real DIII-D: cache, norms, inputs (train.py's loaders) ---
    fv = __import__(f"features_{args.feature_set}")
    shots = pd.read_parquet(args.cache / "shots.parquet")
    assert_disjoint(shots)
    train_rows, val_rows = _rows(shots, "train"), _rows(shots, "val")
    if args.exclude_shot:
        keep = ~train_rows["shot"].astype(str).isin(args.exclude_shot)
        print(f"excluding {int((~keep).sum())} training shot(s): {args.exclude_shot}", flush=True)
        train_rows = train_rows[keep].reset_index(drop=True)
    mm = {k: np.load(args.cache / f"{k}.npy", mmap_mode="r") for k in ("coils", "psi", "scal", "phase")}
    pictures = load_or_build_baseline(args.pictures, args.data_root, args.cache, DEFAULT_FRAMES)
    train_key = split_key(train_rows)
    norm = cached_stats(args.cache / "norm_train.npz", train_key,
                        lambda: fit_norm(mm["coils"], mm["psi"], mm["scal"], train_rows), True)
    norm["psi_std"] = np.float32(norm["psi_std"])
    norm["robust_inputs"], norm["amp"] = False, not args.no_amp
    meta_path = args.cache / f"features_{args.feature_set}.json"
    feat_mm = np.load(args.cache / f"features_{args.feature_set}.npy", mmap_mode="r")
    k = int(args.history_frames)
    cols = list(range(fv.N_FEATURES))

    def inputs(rows):
        idx = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n)) for r in rows.itertuples(index=False)])
        return np.asarray(feat_mm[idx], dtype=np.float32)

    tr_in = inputs(train_rows)
    import hashlib
    feat_key = hashlib.sha256((train_key + meta_path.read_text()).encode()).hexdigest()[:16]
    flags = [fv.FEATURES.index(n) for n in fv.FLAGS]
    feat = cached_stats(args.cache / f"features_{args.feature_set}_norm.npz", feat_key,
                        lambda: dict(zip(("feat_mean", "feat_std"), robust_center_scale(tr_in, flag_cols=flags))), True)
    feat["feat_mean"], feat["feat_std"] = history_stats(np.asarray(feat["feat_mean"]), np.asarray(feat["feat_std"]), k)
    tr_in = _with_history(tr_in, train_rows, k)
    norm = {**norm, "feature_set": args.feature_set, "feature_columns": cols, "history_frames": k,
            "feature_names": list(fv.FEATURES), **feat}
    norm["rax_col"] = k * len(cols) + norm["feature_names"].index("R_ax_th")
    args.n_in = tr_in.shape[1]
    print(f"features {args.feature_set}: {args.n_in} inputs (history {k})", flush=True)
    train, coil_lo, coil_hi = load_split(mm["coils"], mm["psi"], mm["scal"], mm["phase"], train_rows, norm, device,
                                         keep_raw_scalars=False, coil_rows=tr_in)
    del tr_in
    norm = {**norm, "coil_lo": coil_lo, "coil_hi": coil_hi}
    val_in = _with_history(inputs(val_rows), val_rows, k)
    val, _, _ = load_split(mm["coils"], mm["psi"], mm["scal"], mm["phase"], val_rows, norm, device,
                           keep_raw_scalars=True, coil_rows=val_in)
    del val_in
    subset = pd.read_parquet(args.cache / "score_subset.parquet")
    print(f"{train['psi'].shape[0]} train / {val['psi'].shape[0]} val frames on {device}", flush=True)

    green = torch.as_tensor(np.ascontiguousarray(pictures["green"], np.float32), device=device)
    ecoil = torch.as_tensor(np.ascontiguousarray(pictures["ecoil"], np.float32), device=device)
    d3d = features_v2.geometry("d3d")
    norm["decoder_geometry"] = (float(d3d["R0"]), float(d3d["Z0"]), float(d3d["a"]))

    # --- model ---
    extra = {}
    cls = Net
    if linear:
        from lindec import LinearDecoderNet
        cls = LinearDecoderNet
        extra = dict(erode=0, activation="softplus", exact_ip=False, ip_floor=args.lindec_ip_floor,
                     exact_ip_mast=args.lindec_exact_ip_mast, attn_decoder=args.attn_decoder, cache=args.cache,
                     mast_mask=args.mast_mask)
    net = cls(n_in=args.n_in, width=args.width, n_blocks=args.n_blocks, scalar_width=args.scalar_width, scalar_depth=args.scalar_depth,
              seed_channels=args.seed_channels, pixel_mean=norm["pixel_mean"], psi_std=float(norm["psi_std"]),
              decoder_norm="physical", output_prior=args.output_prior, geometry=norm["decoder_geometry"],
              **extra).to(device)

    # --- optional pieces ---
    mast = axis_d = axis_m = None
    if args.lambda_axis > 0:
        from physics_loss import AxisPull, MastInputs
        env_d = np.load(MASKS / "d3d_envelope.npz")["mask_coarse"]
        # --axis-vertical-only: radial bounds so wide the radial term never acts; the vertical term remains.
        radial = dict(hi=1e9, lo=-1e9) if args.axis_vertical_only else {}
        axis_d = AxisPull(d3d["r"], d3d["z"], env_d, d3d["R0"], d3d["a"], device, z0=d3d["Z0"],
                          z_slack=args.axis_z_slack, **radial)
        mast = MastInputs(args.cache, norm, device, variant=args.mast_variant)
        gm = features_v2.geometry("mast")
        axis_m = AxisPull(gm["r"], gm["z"], mast_region.mask(), gm["R0"], gm["a"], device, z0=gm["Z0"],
                          z_slack=args.axis_z_slack, **(radial or {"hi": args.mast_axis_hi, "lo": args.mast_axis_lo}))
    synth_sets = []
    if args.synth:
        from synth_data import SynthSet
        for spec in args.synth:
            tag, _, rest = spec.partition(":")
            w, _, frac = rest.partition(":")
            synth_sets.append(SynthSet(Path(tag), norm, device, mast_variant=args.mast_variant or "_sol0.5",
                                       d3d_picture=lambda st: _external(st, green, ecoil), weight=float(w or 1), frac=float(frac or 1)))
        if args.synth_shape:
            sw = {k: float(v) for k, v in (kv.split("=") for kv in args.synth_shape.split(","))}
            bad = set(sw) - {"grad", "jtor", "axis", "bnd", "li"}
            if bad:
                raise SystemExit(f"--synth-shape: unknown terms {sorted(bad)}")
            on = [ss.tag for ss in synth_sets if ss.enable_shape(sw)]
            if not on:
                raise SystemExit("--synth-shape: no MAST synthetic set has shape_targets.npz")
            print(f"shape losses {sw} on {on}", flush=True)
    monitor = None
    if not args.no_mast_monitor:
        from mast_monitor import MastMonitor
        monitor = MastMonitor(args.cache, norm, device, variant=args.mast_variant or "_sol0.5", mast=mast)

    noise = None
    if args.thomson_noise > 0 or args.thomson_dropout > 0:
        keys = ("Te_", "ne_", "p_rho", "betaN_e", "R_ax_th")
        names, width, slots = norm["feature_names"], len(norm["feature_names"]), 2 * k + 1
        th_cols = [s * width + i for s in range(slots) for i, n in enumerate(names) if any(key in n for key in keys)]
        th_mask = torch.zeros(args.n_in, device=device)
        th_mask[th_cols] = 1.0
        # missing Thomson as the pipeline encodes it: profile inputs NaN -> 0 after standardisation, flag raw 1
        slot_th = torch.zeros(slots, args.n_in, device=device)
        slot_flag = torch.zeros(slots, args.n_in, device=device)
        j = names.index("thomson_missing")
        flag_on = float((1.0 - norm["feat_mean"][j]) / norm["feat_std"][j])
        for s_ in range(slots):
            slot_th[s_, [c for c in th_cols if s_ * width <= c < (s_ + 1) * width]] = 1.0
            slot_flag[s_, s_ * width + j] = 1.0

        def noise(x):
            if args.thomson_noise > 0:
                x = x + args.thomson_noise * torch.randn_like(x) * th_mask
            if args.thomson_dropout > 0:
                drop = (torch.rand(x.shape[0], slots, device=x.device) < args.thomson_dropout).to(x.dtype)
                th, fl = drop @ slot_th, drop @ slot_flag
                x = x * (1 - th) * (1 - fl) + fl * flag_on
            return x
        print(f"Thomson input noise {args.thomson_noise} sd on {len(th_cols)} columns, dropout {args.thomson_dropout} "
              f"per frame (flag value {flag_on:.2f})", flush=True)
    ff_d = ff_m = None
    if args.lambda_fluxfn > 0:
        from fluxfn_loss import FluxFunctionLoss
        from lindec_build import load_g
        th = {"d3d": None, "mast": None}
        if args.fluxfn_pressure > 0:
            tr = np.concatenate([np.arange(int(r.start), int(r.start) + int(r.n)) for r in train_rows.itertuples(index=False)])
            for m, rows in (("d3d", tr), ("mast", None if mast is None else mast.rows)):
                if rows is not None:
                    raw = np.load(args.cache / f"thomson_raw_{m}.npz")
                    th[m] = {q: raw[q][rows] for q in ("te", "r", "z")}
                    th[m]["ne"] = np.load(args.cache / f"thomson_raw_ne_{m}.npz")["te"][rows]
        ffs = {}
        for m in ("d3d", "mast"):
            gm_, gg = features_v2.geometry(m), load_g(m, args.cache)
            ffs[m] = FluxFunctionLoss(gm_["r"], gm_["z"], gg["mask"], gg["idx"], gm_["R0"], device,
                                      thomson=th[m], pressure_weight=args.fluxfn_pressure)
        ff_d, ff_m = ffs["d3d"], (ffs["mast"] if mast is not None else None)

    # --- run ---
    if args.init is not None:
        ck0 = torch.load(args.init, map_location=device, weights_only=False)
        net.load_state_dict(ck0["model"])
        print(f"initialised from {args.init} (epoch {ck0.get('epoch')})", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = train["coils"].shape[0]
    steps_per = max(1, math.ceil(n_train / args.batch_size))
    total, warmup = steps_per * args.epochs, int(args.warmup_epochs * steps_per)
    use_amp = device.type == "cuda" and not args.no_amp
    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k_: str(v) if isinstance(v, Path) else v
                                                     for k_, v in vars(args).items()}, indent=2))
    metrics = run_dir / "metrics.jsonl"
    score = {"proc": None, "done": set()}

    def launch_score(epoch: int, wait: bool) -> None:
        if args.score_every <= 0 or epoch in score["done"]:
            return
        p = score["proc"]
        if p is not None and p.poll() is None:
            if not wait:
                print(f"epoch {epoch}: score.py still running, skip", flush=True)
                return
            p.wait()
        pred = run_dir / f"score_ep{epoch:03d}"
        _dump_subset(net, val, norm, subset["shot"], pred, device, args.batch_size, green, ecoil)
        ck = run_dir / f"score_ep{epoch:03d}.pt"
        shutil.copyfile(run_dir / "last.pt", ck)
        cmd = [sys.executable, str(HERE / "score.py"), "--pred", str(pred), "--split", "score", "--append-metrics",
               str(metrics), "--epoch", str(epoch), "--cache", str(args.cache), "--data-root", str(args.data_root)]
        score["proc"] = subprocess.Popen(cmd, stdout=open(run_dir / f"score_ep{epoch:03d}.log", "w"),
                                         stderr=subprocess.STDOUT, start_new_session=True)
        if args.mast_checks_every > 0 and epoch % args.mast_checks_every == 0:
            diag = HERE / "mast_diag"
            cmds = [[sys.executable, str(diag / f), str(ck)]
                    for f in ("lindec_mast_checks.py", "physics_ckpt.py", "li_ckpt_cli.py")]
            if synth_sets:
                cmds.append([sys.executable, str(diag / "synth_shape_check.py"), str(ck), *[s.tag for s in synth_sets]])
            subprocess.Popen(["bash", "-c", "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3; "
                              + " ; ".join(shlex.join(c) for c in cmds)],
                             stdout=open(run_dir / f"mast_checks_ep{epoch:03d}.log", "w"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        score["done"].add(epoch)
        print(f"epoch {epoch}: scoring -> {pred.name}", flush=True)

    synths = {"queue": [], "proc": None}

    def select_best_mast() -> None:
        """best_mast.pt = the scored epoch with the highest synthetic-MAST S among those with DIII-D S >= floor."""
        d3d = {}
        for line in metrics.read_text().splitlines() if metrics.exists() else []:
            r = json.loads(line)
            if r.get("source") == "score":
                d3d[r["epoch"]] = r["S"]
        rows = []
        for f in sorted(run_dir.glob("synthS_ep*.json")):
            ep = int(f.stem[len("synthS_ep"):])
            sv = json.loads(f.read_text())["S"].get(args.synthS_metric)
            if sv is not None:
                rows.append({"epoch": ep, "synthS": sv, "d3d_S": d3d.get(ep),
                             "eligible": d3d.get(ep) is not None and d3d[ep] >= args.d3d_floor})
        ok = [r for r in rows if r["eligible"]]
        if not ok:
            return
        top = max(ok, key=lambda r: r["synthS"])
        out = run_dir / "best_mast.json"
        prev = json.loads(out.read_text()).get("epoch") if out.exists() else None
        if prev != top["epoch"] or not (run_dir / "best_mast.pt").exists():
            shutil.copyfile(run_dir / f"score_ep{top['epoch']:03d}.pt", run_dir / "best_mast.tmp")
            (run_dir / "best_mast.tmp").replace(run_dir / "best_mast.pt")
            print(f"best_mast.pt <- epoch {top['epoch']} (synthetic S {top['synthS']:.4f}, "
                  f"DIII-D S {top['d3d_S']:.4f})", flush=True)
        out.write_text(json.dumps({**top, "metric": args.synthS_metric, "d3d_floor": args.d3d_floor,
                                   "scored": rows}, indent=1))

    def synths_pump(wait: bool) -> None:
        """Run queued synthetic-S jobs one at a time; with wait, drain the queue."""
        while True:
            p = synths["proc"]
            if p is not None and p.poll() is None:
                if not wait:
                    return
                p.wait()
            if p is not None:
                synths["proc"] = None
                select_best_mast()
            if not synths["queue"]:
                return
            ep = synths["queue"].pop(0)
            cmd = [sys.executable, "-u", str(SYNTHS / "ensemble_synth.py"), "--out",
                   str(run_dir / f"synthS_ep{ep:03d}.json"), "--sets", args.synthS_sets, "--threads", "4",
                   "--workers", "6", f"{args.name}_ep{ep:03d}", str(run_dir / f"score_ep{ep:03d}.pt")]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            synths["proc"] = subprocess.Popen(["nice", "-n", "10", *cmd], cwd=SYNTHS, env=env,
                                              stdout=open(run_dir / f"synthS_ep{ep:03d}.log", "w"),
                                              stderr=subprocess.STDOUT, start_new_session=True)
            print(f"epoch {ep}: synthetic-MAST S scoring started", flush=True)

    def synths_launch(epoch: int) -> None:
        if args.synthS_every <= 0 or (run_dir / f"synthS_ep{epoch:03d}.json").exists() or epoch in synths["queue"]:
            return
        ck = run_dir / f"score_ep{epoch:03d}.pt"
        if not ck.exists():                      # score.py was skipped this epoch; last.pt is this epoch's weights
            shutil.copyfile(run_dir / "last.pt", ck)
        synths["queue"].append(epoch)
        synths_pump(wait=False)

    best, step, first = -float("inf"), 0, 1
    ema = None                                   # averaged weights (--ema-halflife), per-step decay below
    ema_d = 0.5 ** (1.0 / (args.ema_halflife * steps_per)) if args.ema_halflife > 0 else 0.0
    if args.resume:
        rs = torch.load(run_dir / "resume.pt", map_location=device, weights_only=False)
        net.load_state_dict(rs["net"])
        opt.load_state_dict(rs["opt"])
        best, step, first = rs["best"], rs["step"], rs["epoch"] + 1
        score["done"].update(rs["score_done"])
        torch.set_rng_state(rs["rng"])
        ema = rs.get("ema")
        print(f"resumed from epoch {rs['epoch']} (step {step}, best proxy {best:.4f})", flush=True)
        if args.synthS_every:                    # synthetic-S jobs cut off by the crash
            for ep in range(args.synthS_every, first, args.synthS_every):
                if (run_dir / f"score_ep{ep:03d}.pt").exists():
                    synths_launch(ep)
    for epoch in range(first, last_epoch + 1):
        t0 = time.perf_counter()
        net.train()
        perm = torch.randperm(n_train, device=device)
        sums = {"loss": 0.0, "psi": 0.0, "grad": 0.0, "scal": 0.0, "axis_d": 0.0, "axis_m": 0.0, "ff_d": 0.0, "ff_m": 0.0}
        syn_sums = [0.0] * len(synth_sets)
        seen = 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            b = int(idx.shape[0])
            for g in opt.param_groups:
                g["lr"] = scheduled_lr(step, total, warmup, args.lr)
            opt.zero_grad(set_to_none=True)
            stamp = train["stamp"][idx]
            x_in = train["coils"][idx] if noise is None else noise(train["coils"][idx])
            with precision_context(device.type, use_amp):
                out = net(x_in, _external(stamp, green, ecoil), stamp[:, IP_COL], train["rax"][idx],
                          **({"return_current": True} if ff_d is not None else {}))
            psi_hat, scal_hat = out[0], out[1]
            psi_hat, scal_hat = psi_hat.float(), scal_hat.float()
            l_psi = F.mse_loss(psi_hat, train["psi"][idx])
            l_grad = grad_mse(psi_hat, train["psi"][idx], net.z_coord, net.r_coord) if args.lambda_grad else psi_hat.new_zeros(())
            l_scal = F.huber_loss(scal_hat, train["scal"][idx])
            loss = args.d3d_weight * (l_psi + args.lambda_grad * l_grad + args.lambda_s * l_scal)
            if ff_d is not None:
                nf = min(args.fluxfn_frames, b)
                with torch.autocast(device_type=device.type, enabled=False):
                    l_fd, _, _ = ff_d(psi_hat[:nf] * net.psi_std + net.pixel_mean, out[2][:nf].float(),
                                      stamp[:nf, IP_COL], idx[:nf])
                loss = loss + args.lambda_fluxfn * l_fd
                sums["ff_d"] += l_fd.item() * b
            if axis_d is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    l_ad = psi_hat.new_zeros(()) if args.axis_mast_only else \
                        axis_d(psi_hat * net.psi_std + net.pixel_mean, stamp[:, IP_COL], train["rax"][idx])[0]
                    mb = mast.batch(args.mast_batch_size)
                with precision_context(device.type, use_amp):
                    out_m = net(mb["coils"], mb["picture"], mb["ip"], mb["rax"], coord=mast.coord,
                                prior=mb["prior"], r0=mast.r0,
                                **({"machine": "mast", "return_current": ff_m is not None} if linear else {}))
                psi_m = out_m[0]
                with torch.autocast(device_type=device.type, enabled=False):
                    l_am, _ = axis_m(psi_m.float() * net.psi_std + net.pixel_mean, mb["ip"], mb["rax"])
                    if ff_m is not None:
                        l_fm, _, _ = ff_m(psi_m.float() * net.psi_std + net.pixel_mean, out_m[2].float(), mb["ip"],
                                          mb["idx"])
                        loss = loss + args.lambda_fluxfn * l_fm
                        sums["ff_m"] += l_fm.item() * b
                loss = loss + args.lambda_axis * (l_ad + l_am)
                sums["axis_d"] += l_ad.item() * b
                sums["axis_m"] += l_am.item() * b
            for j, ss in enumerate(synth_sets):
                sidx = ss.batch_idx(args.synth_batch)
                if sidx is None:
                    continue
                with precision_context(device.type, use_amp):
                    l_syn = ss.loss(net, sidx, linear, noise)
                syn_sums[j] += l_syn.item() * b
                loss = loss + ss.weight * l_syn
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            if ema_d and epoch >= args.ema_start:
                with torch.no_grad():
                    if ema is None:
                        ema = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
                    else:
                        for k_, v in net.state_dict().items():
                            if v.dtype.is_floating_point:
                                ema[k_].mul_(ema_d).add_(v.detach(), alpha=1 - ema_d)
                            else:
                                ema[k_].copy_(v)
            sums["loss"] += loss.item() * b; sums["psi"] += l_psi.item() * b
            sums["grad"] += l_grad.item() * b; sums["scal"] += l_scal.item() * b
            seen += b
            step += 1
        raw = None
        if ema is not None:                      # evaluate, save and score the averaged weights
            raw = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            net.load_state_dict(ema)
        stats = evaluate(net, args.batch_size, norm, val, device, green, ecoil)
        syn_eval = {ss.tag: ss.evaluate(net, linear) for ss in synth_sets}
        mon = monitor(net, linear) if monitor is not None else {}
        record = {"epoch": epoch, **{f"train_{k_}": v / seen for k_, v in sums.items()},
                  **{f"synth_{ss.tag}_train_loss": syn_sums[j] / seen for j, ss in enumerate(synth_sets)},
                  **{f"synth_{t}_heldout_{k_}": v for t, e in syn_eval.items() for k_, v in e.items()},
                  **{f"synth_{ss.tag}_shape_{k_}": v for ss in synth_sets if ss.shape is not None
                     for k_, v in ss.take_shape_sums().items()},
                  **mon, "seconds": time.perf_counter() - t0, **stats}
        append_jsonl(metrics, record)
        improved = stats["proxy"] > best
        if improved:
            best = stats["proxy"]
            _save(run_dir / "best.pt", net, norm, args, epoch, best, coil_lo, coil_hi)
        _save(run_dir / "last.pt", net, norm, args, epoch, stats["proxy"], coil_lo, coil_hi)
        torch.save({"net": raw if raw is not None else net.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "step": step, "best": best, "score_done": sorted(score["done"]), "rng": torch.get_rng_state(),
                    "ema": ema}, run_dir / "resume.tmp")
        (run_dir / "resume.tmp").replace(run_dir / "resume.pt")
        print(f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  R²ψ {stats['r2_psi']:.4f}  "
              f"R²qb {stats['r2_qb']:.4f}  ∇rmse {stats['rmse_grad']:.4g}  "
              + (f"axis D {record['train_axis_d']:.2e} M {record['train_axis_m']:.2e}  " if axis_d is not None else "")
              + (f"ff D {record['train_ff_d']:.3f} M {record['train_ff_m']:.3f}  " if ff_d is not None else "")
              + (monitor.line(mon) if mon else "")
              + "".join(f"syn {ss.tag} {record[f'synth_{ss.tag}_train_loss']:.3g}/{syn_eval[ss.tag]['loss']:.3g} "
                        f"R²ψ {syn_eval[ss.tag]['r2_psi']:.4f}  " for ss in synth_sets)
              + f"{record['seconds']:.0f}s" + ("  best" if improved else ""), flush=True)
        if args.score_every and epoch % args.score_every == 0:
            launch_score(epoch, wait=False)
        if args.synthS_every and epoch % args.synthS_every == 0:
            synths_launch(epoch)
        synths_pump(wait=False)
        if raw is not None:
            net.load_state_dict(raw)
    if ema is not None:
        net.load_state_dict(ema)                 # the final scoring dump matches last.pt
    launch_score(last_epoch, wait=True)
    if args.synthS_every:
        synths_launch(last_epoch)
        if score["proc"] is not None:
            score["proc"].wait()                 # the last epoch's DIII-D S is needed for the floor
        synths_pump(wait=True)
        select_best_mast()
    print(f"done. best proxy {best:.4f} -> {run_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
