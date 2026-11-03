"""Predict-only run of a saved ml/ch1h (train_c1h.py) or ml/ch1c (train_c1c.py) network on any inputs-only split.

    python infer_c1h.py --run <run dir with model.pt + log.json> --cache $FEQ_CACHE/nn/<split> \
        --px $FEQ_CACHE/nn/px/<split> --hist $FEQ_CACHE/ml/ch1h/hist/<split> \
        --out <dir>/preds_test [--ref <run dir>/preds_test] [--mem-cap-gb 80]

Rebuilds the network inputs exactly as the training run built them for its --test-cache: Ridge z-scores and the two
missing-Thomson flags, the Group A / Group B / Group H blocks standardised with the training statistics stored in model.pt,
context columns, the same 8192-frame chunks, whole-shot batches of 48 for sequence heads. The code below is copied line for
line from the nested functions of train_c1h.main() (inputs, add_extra_inputs, first_pass_raw, add_context [per-shot sums in
float64 so that repeated runs are identical], predict,
predict_seq, extra_keys); the module-level classes (Physics, Model, SeqModel, load_shots, shot_index, gather_shots) are
imported from ml/ch1h/train_c1h.py itself. Output files follow the training run's format (psi float32 (T, 65, 65) except for
train_c1h scalar heads, q95, betaN, cons / li / geo when trained, t). With --ref every file is compared with the predictions
written at training time and the largest differences are printed. Needs no training data and no labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ML = HERE.parent
sys.path.insert(0, str(ML / "ch1h"))
import train_c1h as T  # noqa: E402  (read-only import; also puts c1/ and c1/nn/ on sys.path)
from firstpass import FirstPass  # noqa: E402  (c1/nn/firstpass.py)


def is_true(v) -> bool:
    return str(v) == "True"


FEQ_CACHE = __import__("os").environ.get("FEQ_CACHE", "/fsx/zara.ercan/cache")  # (package) cache root
TRAIN_CACHE_ROOT = "/fsx/zara.ercan/cache"   # cache root of the training runs whose paths model.pt / log.json store


def remap(p) -> Path:
    """Paths stored at training time (absolute, training machine) -> the same file under this machine's FEQ_CACHE."""
    s = str(p)
    if Path(s).exists():                     # stored path valid on this machine (same cache root): use it as it is
        return Path(s)
    if s == TRAIN_CACHE_ROOT or s.startswith(TRAIN_CACHE_ROOT + "/"):   # whole folder names only
        return Path(FEQ_CACHE + s[len(TRAIN_CACHE_ROOT):])
    return Path(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training run directory (model.pt, log.json)")
    ap.add_argument("--cache", type=Path, required=True, help="c1/nn per-shot cache of the split to predict")
    ap.add_argument("--px", type=Path, default=None, help="Group A cache of the split (models trained with --px)")
    ap.add_argument("--hist", type=Path, default=None, help="Group H cache of the split (models trained with --hist)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None, help="training-time predictions of the same split, to compare")
    ap.add_argument("--train-cache", type=Path, default=None,
                    help="cache holding meta.json / basis.npz of the training split (default: the run's --cache)")
    ap.add_argument("--stems", type=Path, default=None, help="file with the shots to predict (default: all in --cache)")
    ap.add_argument("--keep-psi", action="store_true", help="also write the Ridge map of train_c1h scalar heads")
    ap.add_argument("--mem-cap-gb", type=float, default=0.0, help="cap the CUDA memory of this process (GB)")
    a = ap.parse_args()
    t_start = time.time()

    ck = torch.load(a.run / "model.pt", map_location="cpu", weights_only=False)
    la = json.loads((a.run / "log.json").read_text())["args"]
    trainer = "train_c1h" if "hist" in la else "train_c1c"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and a.mem_cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb / total))
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))

    train_cache = a.train_cache or remap(la["cache"])
    meta = json.loads((train_cache / "meta.json").read_text())
    basis = dict(np.load(train_cache / "basis.npz"))
    ridge = dict(np.load(remap(ck["ridge"])))
    names = meta["names"]
    assert list(ridge["names"]) == names, "feature names differ between cache and ridge model"
    assert list(ck["names"]) == names, "feature names differ between cache and model.pt"
    tmeta = a.cache / "meta.json"
    if tmeta.exists():
        assert json.loads(tmeta.read_text())["names"] == names, "split cache has a different feature layout"

    head = ck["head"]
    seq = is_true(la.get("seq"))
    ctx = bool(ck["ctx"])
    helicity = bool(ck["helicity"])
    ctx_wins = [int(v) for v in (la.get("ctx_wins") or "").split(",") if v and v != "None"]
    use_px = ck.get("px") is not None
    firstpass = bool(ck.get("firstpass"))
    use_hist = ck.get("hist") is not None
    n_scal, n_in, r_scale = int(ck["n_scal"]), int(ck["n_in"]), float(ck["r_scale"])
    s_mu, s_sd = ck["s_mu"].to(dev), ck["s_sd"].to(dev)
    li_head = float(la.get("li_head") or 0.0)
    aux_cons = float(la.get("aux_cons") or 0.0)
    aux_geo = is_true(la.get("aux_geo"))
    extra_stats = {k: (v[0].to(dev), v[1].to(dev)) for k, v in (ck.get("extra_stats") or {}).items()}
    if use_px and a.px is None:
        ap.error("this network was trained with --px: pass --px <Group A cache of the split>")
    if use_hist and a.hist is None:
        ap.error("this network was trained with --hist: pass --hist <Group H cache of the split>")
    hist_cols = None
    if use_hist:
        hmeta = json.loads((remap(ck["hist"]) / "meta.json").read_text())
        hist_cols = np.array([hmeta["names"].index(n) for n in ck["hist_names"]], dtype=int)
        smeta = a.hist / "meta.json"
        if smeta.exists():
            assert json.loads(smeta.read_text())["names"] == hmeta["names"], "Group H column layout differs"
    phys = T.Physics(basis, ridge, meta["cols"], dev)

    stems = (a.stems.read_text().split() if a.stems else
             sorted(p.stem for p in a.cache.glob("*.npz") if p.stem not in ("basis", "meta")))
    te, te_off, te_T = T.load_shots(a.cache, stems, with_psi=False, px=a.px if use_px else None,
                                    hist=a.hist if use_hist else None, hist_cols=hist_cols)
    if helicity:
        bt = json.loads((a.cache.parent / f"bt_sign_{a.cache.name}.json").read_text())
        h = np.array([bt[s]["bt_sign"] * bt[s]["ip_sign"] for s in stems], np.float32)
        te["hel"] = h[te["shot"]]
    d = {k: torch.from_numpy(v).to(dev) for k, v in te.items()}
    del te

    core = torch.tensor([n.startswith(("coreTe", "coreNe")) for n in names], device=dev)
    tan = torch.tensor([n.startswith(("tanTe", "tanNe")) for n in names], device=dev)
    i_bcoil = names.index("bcoil")
    fpm = FirstPass(T.GRID_R, T.GRID_Z, T.ENVELOPE, dev) if firstpass else None
    ibz = names.index("bz_over_bip") if firstpass else None

    # ---------------------------------------------------------------- copied from train_c1h.main()
    def standardize(F0, mu, sd):
        return torch.nan_to_num((F0 - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5, 5)

    def first_pass_raw(d):
        """Group B features of one split from the Ridge map (chunked); NaN where undefined."""
        outs = []
        for i in range(0, len(d["ip"]), 8192):
            sl = slice(i, i + 8192)
            z = phys.z(d["X"][sl])
            P = phys.ridge_psi(z, d["I"][sl], d["ip"][sl])
            Pv = phys.vacuum(d["I"][sl]) + phys.ecoil(d["I"][sl])
            outs.append(fpm(P, d["ip"][sl], d["bt"][sl], d["X"][sl][:, ibz], d["tspos"][sl], d["tste"][sl], d["tsne"][sl], Pv))
        return torch.cat(outs)

    def add_context(d):
        """Per-shot mean of z and the mean of z over the previous/current/next frame (same shot)."""
        z = phys.z(d["X"])
        shot = d["shot"].long()
        n_shot = int(shot.max()) + 1
        cnt = torch.zeros(n_shot, device=dev).index_add_(0, shot, torch.ones(len(shot), device=dev))
        # (only change vs train_c1h) per-shot sums accumulated in float64: the float32 CUDA index_add_ of the training code
        # adds in a run-dependent order, so its last bits (and the context heads' outputs, by up to ~1e-3 relative) change
        # from run to run; float64 accumulation makes repeated runs identical and agrees with float32 sums to rounding.
        mean = (torch.zeros(n_shot, z.shape[1], dtype=torch.float64, device=dev).index_add_(0, shot, z.double())
                / cnt[:, None].double()).float()
        i = torch.arange(len(shot), device=dev)
        prev = torch.where((i > 0) & (shot == shot[(i - 1).clamp(min=0)]), (i - 1).clamp(min=0), i)
        nxt = torch.where((i < len(shot) - 1) & (shot == shot[(i + 1).clamp(max=len(shot) - 1)]),
                          (i + 1).clamp(max=len(shot) - 1), i)
        parts = [mean[shot], (z[prev] + z + z[nxt]) / 3]
        if ctx_wins:                    # (ch1c) centred moving means over longer windows, inside each shot (float64 cumsum)
            first = torch.zeros(n_shot, dtype=torch.long, device=dev).scatter_reduce_(
                0, shot, i, reduce="amin", include_self=False)
            start = first[shot]
            end = start + cnt.long()[shot]
            cs = torch.cat([torch.zeros(1, z.shape[1], dtype=torch.float64, device=dev), torch.cumsum(z.double(), 0)])
            for W in ctx_wins:
                h = W // 2
                lo = torch.maximum(start, i - h)
                hi = torch.minimum(end, i + h + 1)
                parts.append(((cs[hi] - cs[lo]) / (hi - lo).double()[:, None]).float())
            del cs
        d["ctx"] = torch.cat(parts, 1)

    def add_extra_inputs(d):
        """Group A / Group B / context columns of a non-training split, with the training statistics."""
        if use_px:
            d["zp"] = standardize(d.pop("Xp"), *extra_stats["px"])
        if use_hist:
            d["zh"] = standardize(d.pop("Xh"), *extra_stats["hist"])
        if firstpass:
            d["zf"] = standardize(first_pass_raw(d), *extra_stats["fp"])
        for k in ("tspos", "tste", "tsne"):
            d.pop(k, None)
        if ctx:
            add_context(d)

    def inputs(d, sl):
        X = d["X"][sl]
        z = phys.z(X)
        fin = torch.isfinite(X)
        parts = [z, torch.stack([(~fin[:, core]).all(1), (~fin[:, tan]).all(1)], 1).float()]
        if helicity:
            h = d["hel"][sl][:, None]
            parts += [h, z[:, i_bcoil:i_bcoil + 1] * h]
        if ctx:
            parts.append(d["ctx"][sl])
        if "zp" in d:
            parts.append(d["zp"][sl])
        if "zf" in d:
            parts.append(d["zf"][sl])
        if "zh" in d:
            parts.append(d["zh"][sl])
        return torch.cat(parts, 1), z

    def extra_keys(sc):
        if li_head > 0:
            return {"li": sc[:, 2].astype(np.float32)}
        if aux_cons > 0:
            out = {"cons": sc[:, 2:9].astype(np.float32), "li": sc[:, 8].astype(np.float32)}
            if aux_geo:
                out["geo"] = sc[:, 9:15].astype(np.float32)
            return out
        return {}

    @torch.no_grad()
    def predict_seq(model, d, chunk=8192, shots_per_batch=48):
        N = len(d["ip"])
        psi_out = torch.empty((N, T.NPIX), device=dev)
        sc_out = torch.empty((N, n_scal), device=dev)
        zin_all = []
        for i in range(0, N, chunk):
            sl = slice(i, i + chunk)
            zin, z = inputs(d, sl)
            psi_out[sl] = phys.ridge_psi(z, d["I"][sl], d["ip"][sl])
            zin_all.append(zin.to(torch.bfloat16))
        Z = torch.cat(zin_all)
        del zin_all
        st, Tn = T.shot_index(d["shot"].long())
        model.net.eval()
        for j in range(0, len(st), shots_per_batch):
            b = torch.arange(j, min(j + shots_per_batch, len(st)), device=dev)
            X, valid, idx = T.gather_shots(Z, st, Tn, b)
            out, ps = model.run(X, valid[..., None].to(X.dtype))
            fi = idx[valid]
            sc_out[fi] = out[valid] * s_sd + s_mu
            if ps is not None:
                psi_out[fi] += r_scale * ps[valid]
        return psi_out, sc_out

    @torch.no_grad()
    def predict(model, d, chunk=8192):
        if getattr(model, "seq", False):
            return predict_seq(model, d, chunk)
        psi_out = torch.empty((len(d["ip"]), T.NPIX), device=dev)
        sc_out = torch.empty((len(d["ip"]), n_scal), device=dev)
        for i in range(0, len(d["ip"]), chunk):
            sl = slice(i, i + chunk)
            zin, z = inputs(d, sl)
            p, s = model.forward(zin, z, d["I"][sl], d["ip"][sl])
            psi_out[sl] = p
            sc_out[sl] = s * s_sd + s_mu
        return psi_out, sc_out
    # ---------------------------------------------------------------- end of the copied code

    add_extra_inputs(d)
    if seq:
        sq = ck.get("seq") or {k: la[k] for k in ("seq_width", "seq_dil", "seq_kernel", "seq_drop")}
        sargs = SimpleNamespace(seq_width=int(sq["seq_width"]), seq_dil=str(sq["seq_dil"]), seq_kernel=int(sq["seq_kernel"]),
                                seq_drop=float(sq["seq_drop"]), width_scal=int(ck["width_scal"]), head=head)
        model = T.SeqModel(n_in, sargs, n_scal, dev)
    else:
        model = T.Model(head, phys, n_in, int(ck["width"]), int(ck["depth"]), float(ck["drop"]), dev, r_scale,
                        int(ck["width_scal"]), n_scal)
    model.net.load_state_dict(ck["state"])
    model.net.eval()
    zin0, _ = inputs(d, slice(0, 1))
    assert zin0.shape[1] == n_in, f"input width {zin0.shape[1]} != {n_in} stored in model.pt"
    psi_t, sc_t = predict(model, d)
    psi_t, sc_t, t_te = psi_t.cpu().numpy(), sc_t.cpu().numpy(), d["t"].cpu().numpy()
    T.check_finite_preds(psi_t, sc_t, "predictions")
    write_psi = head != "scalar" or trainer == "train_c1c" or a.keep_psi
    a.out.mkdir(parents=True, exist_ok=True)
    for s, o, n in zip(stems, te_off, te_T):
        kw = {"psi": psi_t[o:o + n].reshape(n, 65, 65).astype(np.float32)} if write_psi else {}
        np.savez(a.out / f"{s}.npz", **kw, q95=sc_t[o:o + n, 0].astype(np.float32),
                 betaN=sc_t[o:o + n, 1].astype(np.float32), **extra_keys(sc_t[o:o + n]), t=t_te[o:o + n].astype(np.float32))
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
    print(f"{a.run.name}: {trainer} {'sequence' if seq else head} head, {len(stems)} shots, {len(t_te):,} frames -> {a.out} "
          f"({time.time() - t_start:.0f}s on {gpu}, peak CUDA memory {peak:.1f} GB)", flush=True)

    if a.ref is not None:
        worst = {}
        same = total = 0
        for s in stems:
            p, q = np.load(a.out / f"{s}.npz"), np.load(a.ref / f"{s}.npz")
            for k in sorted(set(p.files) & set(q.files)):
                x, y = p[k], q[k]
                assert x.shape == y.shape, (s, k, x.shape, y.shape)
                total += 1
                same += int(x.tobytes() == y.tobytes())
                dmax = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
                scale = float(np.max(np.abs(y))) if y.size else 0.0
                w = worst.get(k, (0.0, 0.0, ""))
                if dmax > w[0]:
                    worst[k] = (dmax, scale, s)
                elif k not in worst:
                    worst[k] = (0.0, scale, s)
            missing = sorted(set(q.files) - set(p.files))
            if missing and s == stems[0]:
                print(f"  reference keys not written here: {missing}")
        print(f"  vs {a.ref}: {same} of {total} arrays byte-identical; largest |difference| per key: "
              + ", ".join(f"{k} {v[0]:.3g} (max |ref| {v[1]:.3g}, {v[2][9:]})" for k, v in worst.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
