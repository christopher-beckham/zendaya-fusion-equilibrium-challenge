"""Dedicated scalar network on the exported pixel-network inputs (export_inputs.py): q95, betaN and the boundary fraction x_b.

Inputs per frame (same as train_c1.py): Ridge z (294) + 2 Thomson flags, Group A (184) and Group B (66) standardised with the
training-frame statistics (finite mean / sd, NaN -> 0, clamp +-5). Options:
  --pos            position-in-shot inputs: log1p(k), log1p(T-1-k), k == 0, k == T-1, k < 3, T-1-k < 3, 1 / T
  --nbr            the input vectors of the previous and next frame of the same shot (edge frames repeat themselves + flags)
  --targets        comma list from q95, betaN, xb (xb = (psi_EFIT-boundary - psi_axis) / (psi_separatrix - psi_axis), from
                   xb_targets.py on the true maps; frames without a finite target get no loss for it)
Network: residual MLP (LayerNorm, Linear(d, 2d), SiLU, Linear(2d, d)) x depth, Muon on hidden matrices + AdamW elsewhere,
warm-up + cosine. NaN guards: physical label ranges (q95 in (0.5, 60), |betaN| < 15, x_b in [0, 1.05]), float64 statistics,
non-finite gradient steps skipped (counted), finite weights / predictions asserted before writing.

    python train_s.py --tag r0 --seed 0 --out runs/<name> [--holdout 0 --tag rfull]   (holdout 0 = all shots, test predictions)
Writes runs/<name>/preds/<stem>.npz (holdout) and preds_test/<stem>.npz (public test): q95, betaN[, xb] float32, log.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muon_opt  # noqa: E402
from s_common import OUTC, split, write_preds  # noqa: E402

T0 = time.time()
RANGES = {"q95": (0.5, 60.0), "betaN": (-15.0, 15.0), "xb": (0.0, 1.05)}


class Block(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 2 * d); self.fc2 = nn.Linear(2 * d, d); self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.fc2(F.silu(self.fc1(self.ln(x)))))


class Net(nn.Module):
    def __init__(self, n_in, width, depth, n_out, drop=0.0):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[Block(width, drop) for _ in range(depth)])
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_out)

    def forward(self, x):
        return self.head(self.ln(self.blocks(self.inp(x))))


def role(name, p):
    if p.ndim != 2:
        return "other"
    if name.startswith("inp."):
        return "inp"
    if name.startswith("head."):
        return "head"
    return "hidden"


def fit_stats(A):
    A = A.astype(np.float64)
    fin = np.isfinite(A)
    cnt = np.maximum(fin.sum(0), 1)
    mu = np.where(fin, A, 0).sum(0) / cnt
    sd = np.sqrt(np.where(fin, (A - mu) ** 2, 0).sum(0) / cnt)
    sd = np.where(sd > 1e-9, sd, np.inf)
    return mu, sd


def standardize(A, mu, sd):
    return np.clip(np.nan_to_num((A - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0), -5, 5).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r0", help="input export tag (r0 / r42 / rfull)")
    ap.add_argument("--seed", type=int, default=0, help="holdout split seed")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", default="q95,betaN")
    ap.add_argument("--tw", default="", help="loss weights per target (comma), default 1")
    ap.add_argument("--pos", action="store_true")
    ap.add_argument("--nbr", action="store_true")
    ap.add_argument("--ctxw", type=int, default=0, help="add the mean input vector over frames k-K..k+K of the same shot")
    ap.add_argument("--shotmean", action="store_true", help="add the mean input vector of the whole shot")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--huber-delta", type=float, default=1.0, help="in standardised units")
    ap.add_argument("--in-noise", type=float, default=0.0)
    ap.add_argument("--init-seed", type=int, default=1)
    ap.add_argument("--fold", default="", help="K,i: train on the training shots outside fold i of K (shot-grouped), "
                    "predict fold i (written as preds_oof) and the holdout")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--bn-mask-below", type=float, default=None, help="no betaN loss on training frames with betaN label below this")
    ap.add_argument("--bn-clamp-min", type=float, default=None, help="(label study B) training betaN labels below this are set to it")
    ap.add_argument("--bn-train-range", default="", help="(label study D) LO,HI: training frames whose betaN label is outside "
                    "[LO, HI] are removed from the training set entirely")
    ap.add_argument("--softplus", action="store_true", help="(label study E) q95 and betaN outputs pass through softplus in "
                    "physical units (predictions >= 0)")
    ap.add_argument("--huber-targets", default="", help="(label study F) comma list of targets that get the Huber loss with "
                    "--loss huber (default: all targets)")
    ap.add_argument("--load", default="", help="predict only: load this model.pt (same data arguments as its training run), "
                    "skip training, write preds / preds_test")
    ap.add_argument("--xbmix", default="", help="QC_HOLD[,QC_TEST]: x_b = mixture over boundary candidates {1, inner wall "
                    "R = 1.020 / 1.033 / 1.046 m, free value}; wall fractions from the true maps (training frames, xb_all.npz) "
                    "and from the predicted maps' q-curve files (holdout / test); also used as inputs")
    muon_opt.add_args(ap)
    a = ap.parse_args()
    torch.manual_seed(a.init_seed)
    np.random.seed(a.init_seed)
    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    targets = a.targets.split(",")
    tw = torch.tensor([float(v) for v in a.tw.split(",")] if a.tw else [1.0] * len(targets), device=dev)
    hub_mask = torch.tensor([(not a.huber_targets) or (t in a.huber_targets.split(",")) for t in targets], device=dev)

    D = np.load(OUTC / f"inp_{a.tag}_train.npz")
    stems = list(D["stems"])
    tr_st, ho_st = split(a.seed, a.holdout)
    sidx = {s: i for i, s in enumerate(stems)}
    shot = D["shot"]
    tr_mask = np.isin(shot, [sidx[s] for s in tr_st])
    oof_st = []
    if a.fold:
        K, fi = [int(v) for v in a.fold.split(",")]
        perm = np.random.RandomState(777).permutation(len(tr_st))
        fold_of = np.empty(len(tr_st), int); fold_of[perm] = np.arange(len(tr_st)) % K
        oof_st = [s for s, f in zip(tr_st, fold_of) if f == fi]
        keep = [s for s, f in zip(tr_st, fold_of) if f != fi]
        tr_mask = np.isin(shot, [sidx[s] for s in keep])
    ho_mask = np.isin(shot, [sidx[s] for s in ho_st])
    oof_mask = np.isin(shot, [sidx[s] for s in oof_st])

    # labels
    Y = np.full((len(shot), len(targets)), np.nan, np.float64)
    lab = {}
    from s_common import CACHE
    q95 = np.empty(len(shot)); bn = np.empty(len(shot))
    off = np.r_[0, np.cumsum(np.bincount(shot, minlength=len(stems)))]
    for i, s in enumerate(stems):
        with np.load(CACHE / f"{s}.npz") as d:
            q95[off[i]:off[i + 1]] = d["q95"]; bn[off[i]:off[i + 1]] = d["betaN"]
    lab["q95"], lab["betaN"] = q95, bn
    if "xb" in targets:
        X = np.load(OUTC / "xb_all.npz")
        assert list(X["stems"]) == stems
        lab["xb"] = np.clip(X["A"][:, list(X["cols"]).index("x_b")], None, 1.0)
    n_bad = {}
    for j, t in enumerate(targets):
        y = lab[t].astype(np.float64)
        lo, hi = RANGES[t]
        ok = np.isfinite(y) & (y > lo) & (y < hi)
        n_bad[t] = int((~ok & tr_mask).sum())
        Y[:, j] = np.where(ok, y, np.nan)
    print(f"labels outside the physical range / non-finite (training frames, no loss): {n_bad}", flush=True)
    if a.bn_mask_below is not None and "betaN" in targets:   # betaN loss masked on anomalous training frames only
        jb = targets.index("betaN")
        mb = tr_mask & np.isfinite(Y[:, jb]) & (Y[:, jb] < a.bn_mask_below)
        for i_ in np.where(mb)[0]:
            s_ = stems[shot[i_]]
            print(f"  betaN loss masked: {s_} frame {i_ - off[shot[i_]]} label {Y[i_, jb]:.3f}", flush=True)
        Y[mb, jb] = np.nan
        log_bn_mask = int(mb.sum())
        print(f"betaN loss masked on {log_bn_mask} training frames with label < {a.bn_mask_below}", flush=True)
    if a.bn_clamp_min is not None and "betaN" in targets:     # (B) clamp low betaN training labels
        jb = targets.index("betaN")
        mc = tr_mask & np.isfinite(Y[:, jb]) & (Y[:, jb] < a.bn_clamp_min)
        Y[mc, jb] = a.bn_clamp_min
        print(f"betaN training labels below {a.bn_clamp_min} set to {a.bn_clamp_min}: {int(mc.sum())} frames", flush=True)
    if a.bn_train_range:                                     # (D) remove frames outside the betaN operating range
        lo_, hi_ = [float(v) for v in a.bn_train_range.split(",")]
        yb_ = lab["betaN"].astype(np.float64)
        out_ = tr_mask & ~((yb_ >= lo_) & (yb_ <= hi_))
        print(f"training frames removed (betaN label outside [{lo_}, {hi_}] or non-finite): {int(out_.sum())} of {int(tr_mask.sum())} "
              f"(below {lo_}: {int((tr_mask & (yb_ < lo_)).sum())}, above {hi_}: {int((tr_mask & (yb_ > hi_)).sum())})", flush=True)
        tr_mask = tr_mask & ~out_

    # inputs
    mu_p, sd_p = fit_stats(D["XP"][tr_mask]); mu_f, sd_f = fit_stats(D["FB"][tr_mask])

    # (--xbmix) inner-wall boundary fractions: true maps for the training export, predicted maps for holdout / test frames
    Wtr = Wte = None
    if a.xbmix:
        assert "xb" in targets
        qc_paths = a.xbmix.split(",")
        XA = np.load(OUTC / "xb_all.npz")
        assert list(XA["stems"]) == stems
        cols = list(XA["cols"]); A_ = XA["A"]
        pa_, ps_ = A_[:, cols.index("psi_ax")], A_[:, cols.index("psi_sep")]
        Wtr = np.column_stack([(A_[:, cols.index(c)] - pa_) / (ps_ - pa_) for c in ("wall1020", "wall1033", "wall1046")])
        if qc_paths[0]:
            qh = np.load(qc_paths[0])
            assert list(qh["stems"]) == ho_st
            ex = list(qh["extra"])
            Wtr[ho_mask] = qh["E"][:, [ex.index(c) for c in ("x_wall1020", "x_wall1033", "x_wall1046")]]
        else:
            assert not ho_mask.any(), "--xbmix needs the holdout q-curve file when there is a holdout"
        assert Wtr is None or len(qc_paths) > 1 or a.no_test, "--xbmix needs the test q-curve file unless --no-test"
        Wtr = np.where(np.isfinite(Wtr), Wtr, 2.0).astype(np.float32)
        if len(qc_paths) > 1 and qc_paths[1]:   # (package) '--xbmix ,' trains without a test file
            qt_ = np.load(qc_paths[1]); ex = list(qt_["extra"])
            Wte = qt_["E"][:, [ex.index(c) for c in ("x_wall1020", "x_wall1033", "x_wall1046")]]
            Wte = np.where(np.isfinite(Wte), Wte, 2.0).astype(np.float32)
        print(f"xbmix: training frames with a wall fraction < 1: {np.mean((Wtr[tr_mask] < 1).any(1)):.4f}", flush=True)

    def build(DD, W=None):
        parts = [DD["Z"].astype(np.float32), standardize(DD["XP"], mu_p, sd_p), standardize(DD["FB"], mu_f, sd_f)]
        if W is not None:
            parts.append(np.clip((W - 1.0) * 10.0, -5, 5).astype(np.float32))
        if a.pos:
            k, T = DD["k"].astype(np.float64), DD["T"].astype(np.float64)
            e = T - 1 - k
            parts.append(np.column_stack([np.log1p(k), np.log1p(e), k == 0, e == 0, k < 3, e < 3, 1.0 / T]).astype(np.float32))
        return np.concatenate(parts, 1)

    def nbr_index(sh, k, T):
        n = len(sh); i = np.arange(n)
        prev = np.where(k > 0, i - 1, i); nxt = np.where(k < T - 1, i + 1, i)
        return prev, nxt

    Xall = torch.from_numpy(build(D, Wtr)).to(dev)
    Wall_t = torch.from_numpy(Wtr).to(dev) if Wtr is not None else None
    prev, nxt = nbr_index(shot, D["k"], D["T"])
    prev_t, nxt_t = torch.from_numpy(prev).to(dev), torch.from_numpy(nxt).to(dev)
    n_base = Xall.shape[1]
    n_in = n_base * 3 + 2 if a.nbr else n_base
    n_in += n_base * (int(a.ctxw > 0) + int(a.shotmean))

    def context(Xs, sh, k, T):
        """[window mean over k-K..k+K, shot mean] of the input vectors (same shot only), float32 on the GPU."""
        parts = []
        n = len(sh)
        i = np.arange(n)
        if a.ctxw > 0:
            s0, s1 = i - k, i - k + T - 1
            lo = np.maximum(s0, i - a.ctxw); hi = np.minimum(s1, i + a.ctxw)
            cs = torch.cat([torch.zeros(1, Xs.shape[1], dtype=torch.float64, device=dev), torch.cumsum(Xs.double(), 0)])
            lo_t, hi_t = torch.from_numpy(lo).to(dev), torch.from_numpy(hi).to(dev)
            cnt = (hi_t - lo_t + 1).double()[:, None]
            parts.append(((cs[hi_t + 1] - cs[lo_t]) / cnt).float())
            del cs
        if a.shotmean:
            st = torch.from_numpy(sh.astype(np.int64)).to(dev)
            ns = int(st.max()) + 1
            sm = torch.zeros(ns, Xs.shape[1], dtype=torch.float64, device=dev).index_add_(0, st, Xs.double())
            c = torch.zeros(ns, dtype=torch.float64, device=dev).index_add_(0, st, torch.ones(n, dtype=torch.float64, device=dev))
            parts.append((sm / c[:, None])[st].float())
        return torch.cat(parts, 1) if parts else None

    ctx_of = {}
    if a.ctxw > 0 or a.shotmean:
        ctx_of[id(Xall)] = context(Xall, shot, D["k"], D["T"])
        print(f"temporal context inputs: window +-{a.ctxw} frames, shot mean {a.shotmean}; inputs {n_in}", flush=True)

    def feats(Xs, idx, pv, nx):
        x = Xs[idx]
        c = ctx_of.get(id(Xs))
        if a.nbr:
            fl = torch.stack([(pv[idx] == idx).float(), (nx[idx] == idx).float()], 1)
            x = torch.cat([x, Xs[pv[idx]], Xs[nx[idx]], fl], 1)
        if c is not None:
            x = torch.cat([x, c[idx]], 1)
        return x

    Yt = torch.from_numpy(Y).to(dev)
    tr_idx = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
    okY = torch.isfinite(Yt[tr_idx])
    y_mu = torch.stack([Yt[tr_idx][okY[:, j], j].mean() for j in range(len(targets))]).float()
    y_sd = torch.stack([Yt[tr_idx][okY[:, j], j].std() for j in range(len(targets))]).float()
    Ys = ((Yt - y_mu.double()) / y_sd.double()).float()
    print(f"train frames {len(tr_idx):,}, holdout {int(ho_mask.sum()):,}, oof {int(oof_mask.sum()):,}; inputs {n_in}; "
          f"targets {targets} mu {y_mu.tolist()} sd {y_sd.tolist()}", flush=True)

    nt = len(targets)
    net = Net(n_in, a.width, a.depth, nt + (6 if a.xbmix else 0), a.drop).to(dev)
    jx = targets.index("xb") if "xb" in targets else -1
    if a.load:                                               # predict only
        ck = torch.load(a.load, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"])
        assert torch.allclose(ck["y_mu"].to(dev), y_mu) and torch.allclose(ck["y_sd"].to(dev), y_sd), "label statistics differ"
        a.epochs = 0
        print(f"loaded {a.load}; training skipped", flush=True)

    def post(o, W):
        """standardised outputs (B, nt); with --xbmix the x_b column is the candidate mixture."""
        if not a.xbmix and not a.softplus:
            return o
        o2 = o[:, :nt].clone()
        if a.xbmix:
            cand = torch.cat([torch.ones_like(W[:, :1]), W.clamp(max=1.0), 0.75 + 0.25 * torch.sigmoid(o[:, nt + 5:nt + 6])], 1)
            xbh = (torch.softmax(o[:, nt:nt + 5], 1) * cand).sum(1)
            o2[:, jx] = (xbh - y_mu[jx]) / y_sd[jx]
        if a.softplus:                                       # (E) physical-unit softplus on q95 / betaN
            for jj in [targets.index(t_) for t_ in ("q95", "betaN") if t_ in targets]:
                o2[:, jj] = (F.softplus(o[:, jj] * y_sd[jj] + y_mu[jj]) - y_mu[jj]) / y_sd[jj]
        return o2
    if a.opt == "muon":
        mu_par, ad_par, summ = muon_opt.split_params(net.named_parameters(), role, a)
        opts = [muon_opt.make_muon(mu_par, a), torch.optim.AdamW(ad_par, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))]
    else:
        opts = [torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))]
    opt = muon_opt.Combo(opts)
    N = len(tr_idx); spe = N // a.batch; total = max(1, a.epochs * spe); warm = max(1, min(1000, total // 20))
    lam = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in opts]

    @torch.no_grad()
    def predict(Xs, idx_np, pv, nx, Ws=None):
        net.eval()
        out = []
        idx = torch.from_numpy(idx_np).to(dev)
        for i in range(0, len(idx), 16384):
            j = idx[i:i + 16384]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(feats(Xs, j, pv, nx))
            o = post(o.float(), Ws[j] if Ws is not None else None)
            out.append(o * y_sd + y_mu)
        net.train()
        return torch.cat(out).double().cpu().numpy()

    def r2(y, p):
        m = np.isfinite(y)
        return float(1 - np.sum((p[m] - y[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2))

    ho_idx = np.where(ho_mask)[0]
    Yraw = np.column_stack([lab[t] for t in targets])           # scorer: every finite label counts
    log = {"args": {k: str(v) for k, v in vars(a).items()}, "targets": targets, "epochs": [], "n_bad_labels": n_bad}
    step = 0
    for ep in range(a.epochs):
        t0 = time.time()
        perm = tr_idx[torch.randperm(N, device=dev)]
        tot = 0.0; skipped = 0
        for i in range(spe):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            x = feats(Xall, idx, prev_t, nxt_t)
            if a.in_noise > 0:
                x = x + a.in_noise * torch.randn_like(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(x)
            p = post(p.float(), Wall_t[idx] if Wall_t is not None else None)
            y = Ys[idx]
            m = torch.isfinite(y)
            d = p - torch.nan_to_num(y)
            l = d * d
            if a.loss == "huber":
                ad_ = d.abs(); hd = a.huber_delta
                hub = torch.where(ad_ < hd, 0.5 * d * d, hd * (ad_ - 0.5 * hd)) * 2
                l = torch.where(hub_mask[None, :], hub, l)
            loss = (l * tw * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step(); tot += float(loss.detach())
            else:
                skipped += 1; opt.zero_grad(set_to_none=True)
            for s_ in scheds:
                s_.step()
            step += 1
        rec = {"epoch": ep + 1, "loss": tot / max(1, spe - skipped), "skipped": skipped, "time_s": time.time() - t0}
        if len(ho_idx) and ((ep + 1) % 2 == 0 or ep + 1 == a.epochs):
            P = predict(Xall, ho_idx, prev_t, nxt_t, Wall_t)
            for j, t in enumerate(targets):
                rec[f"r2_{t}"] = r2(Yraw[ho_idx, j], P[:, j])
        log["epochs"].append(rec)
        print(" ".join(f"{k} {v:.5f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    bad = [n for n, p_ in net.named_parameters() if not torch.isfinite(p_).all()]
    if bad:
        raise RuntimeError(f"non-finite weights {bad[:3]}; nothing written")
    torch.save({"state": net.state_dict(), "n_in": n_in, "targets": targets, "y_mu": y_mu.cpu(), "y_sd": y_sd.cpu(),
                "args": vars(a)}, a.out / "model.pt")

    def dump(name, idx_np, st, Xs, pv, nx, sh, Ws=None):
        P = predict(Xs, idx_np, pv, nx, Ws)
        assert np.isfinite(P).all(), f"non-finite {name} predictions"
        sub = sh[idx_np]
        uniq = {s: i for i, s in enumerate(st)}
        # rows are in stem order already (shots are contiguous and sorted in the export)
        out = a.out / name
        out.mkdir(exist_ok=True)
        o = 0
        for s in st:
            n = int((sub == sub[o]).sum()) if o < len(sub) else 0
            blk = P[o:o + n]
            np.savez(out / f"{s}.npz", **{t: blk[:, j].astype(np.float32) for j, t in enumerate(targets)})
            o += n
        assert o == len(P)
        return P

    if len(ho_idx):
        P = dump("preds", ho_idx, ho_st, Xall, prev_t, nxt_t, shot, Wall_t)
        log["final"] = {t: r2(Yraw[ho_idx, j], P[:, j]) for j, t in enumerate(targets)}
    if oof_mask.any():
        dump("preds_oof", np.where(oof_mask)[0], oof_st, Xall, prev_t, nxt_t, shot, Wall_t)
    if not a.no_test:
        ctx_of.clear(); del Xall
        torch.cuda.empty_cache()
        DT = np.load(OUTC / f"inp_{a.tag}_test.npz")
        Xt = torch.from_numpy(build(DT, Wte)).to(dev)
        if a.ctxw > 0 or a.shotmean:
            ctx_of[id(Xt)] = context(Xt, DT["shot"], DT["k"], DT["T"])
        pv, nx = nbr_index(DT["shot"], DT["k"], DT["T"])
        dump("preds_test", np.arange(len(DT["shot"])), list(DT["stems"]), Xt, torch.from_numpy(pv).to(dev),
             torch.from_numpy(nx).to(dev), DT["shot"], torch.from_numpy(Wte).to(dev) if Wte is not None else None)
    log["wall_s"] = time.time() - T0
    (a.out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"final {log.get('final')}; wall {log['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
