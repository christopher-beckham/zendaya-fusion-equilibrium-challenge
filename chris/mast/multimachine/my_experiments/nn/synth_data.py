"""Labelled synthetic equilibria (synth_cache.py) as homogeneous training batches and a held-out check (train.py --synth).

One SynthSet per cache folder <cache>/synth/<tag>/ (one machine). Each batch comes from that set only, so the
machine-specific pieces (grid, coil picture, flux scale, the linear decoder's G) apply to the whole batch, exactly as
the label-free MAST batch does (physics_loss.MastInputs).

Inputs are standardised with the run's own norm (feat_mean / feat_std, clipped), with history frames stacked from the
case's input window when the run uses --history-frames. The coil picture: MAST cur @ green of the run's MAST picture
variant (solenoid ×0.5 by default); DIII-D the same `_external(stamp)` as real DIII-D training. Labels stay in
physical units (Wb/rad, our sign convention), and the loss is the mean squared flux error per grid point divided by
the frame's own flux scale μ0 R0 |Ip| / 2π, so frames with different Ip and machines weigh alike.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import CACHE, INPUT_CLIP, IP_COL, stack_history
from model import _plasma_coord_map, flux_scale, plasma_prior_table, prior_from_table

HISTORY_STEP = {"mast": 4, "d3d": 1}   # input-window frames per history offset (MAST 5 ms frames, DIII-D 20 ms)


class SynthSet:
    def __init__(self, folder: Path, norm: dict, device, *, mast_variant: str = "_sol0.5", d3d_picture=None,
                 weight: float = 1.0, frac: float = 1.0):
        import json
        import features_v2
        import mast_region
        folder = Path(folder)
        if not folder.is_absolute() and not folder.exists():
            folder = CACHE / "synth" / folder
        sfx0 = mast_region.suffix()
        if sfx0 and not (folder / "manifest.json").exists() and (folder.with_name(folder.name + sfx0) / "manifest.json").exists():
            folder = folder.with_name(folder.name + sfx0)        # only the region-specific variant was built
        meta = json.loads((folder / "manifest.json").read_text())
        # MAST sets carry MAST-region features (Z0): with a non-default region (mast_region.py), TAG means TAG<suffix>,
        # e.g. zara_g4_v2 -> zara_g4_v2_limiter. DIII-D sets do not depend on it.
        sfx = mast_region.suffix()
        if meta["machine"] == "mast":
            other = [f"_{c}" for c in mast_region.CHOICES if c != "envelope" and f"_{c}" != sfx]
            if any(folder.name.endswith(o) for o in other):
                raise SystemExit(f"{folder.name}: built for another MAST region than {mast_region.name()!r}")
            if sfx and not folder.name.endswith(sfx):
                folder = folder.with_name(folder.name + sfx)
                if not (folder / "manifest.json").exists():
                    raise SystemExit(f"{folder}: no {mast_region.name()} variant of this synthetic set; build it with "
                                     f"{mast_region.ENV_VAR}={mast_region.name()} synth_cache(_seq).py --tag {folder.name}")
                meta = json.loads((folder / "manifest.json").read_text())
        self.machine, self.tag, self.weight = meta["machine"], folder.name, float(weight)
        z = np.load(folder / "arrays.npz")
        mid = int(z["mid"])
        cols = list(norm["feature_columns"])
        k = int(norm.get("history_frames", 0))
        step = HISTORY_STEP[self.machine]
        if mid < k * step:
            raise SystemExit(f"{self.tag}: input window ±{mid} frames is too short for {k} history frames × {step}")
        raw = np.stack([stack_history(np.asarray(f, np.float32)[:, cols], k, step)[mid] for f in z["feats"]])
        x = (raw - norm["feat_mean"]) / norm["feat_std"]
        x = np.clip(np.nan_to_num(x, nan=0.0), -INPUT_CLIP, INPUT_CLIP)
        rax = raw[:, k * len(cols) + norm["feature_names"].index("R_ax_th")]
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=device)  # noqa: E731
        self.split = np.asarray(z["split"])
        self.coils, self.rax = t(x), t(rax)
        self.psi, self.scal = t(z["psi"]), t(z["scal"])
        if self.machine == "mast":
            pics = features_v2.load_pictures(CACHE / "features_v2" / f"pictures_mast{mast_variant}.npz")
            green = t(np.asarray(pics["green"], np.float32))
            self.picture = t(z["cur"][:, mid]) @ green
            self.ip = t(z["ip"][:, mid])
            g = features_v2.geometry("mast")
            self.r0 = float(g["R0"])
            self.coord = _plasma_coord_map(g["r"], g["z"], g["R0"], g["Z0"], g["a"]).to(device)
            self.prior_table = t(plasma_prior_table(g["r"], g["z"], g["R0"], g["Z0"], g["a"]))
            self.geom = g
            st = folder / "shape_targets.npz"                 # synth_shape_targets.py, for --synth-shape
            if st.exists():
                zt = np.load(st)
                self.psi_ax, self.psi_bnd = t(zt["psi_axis"]), t(zt["psi_bnd"])
        else:
            if d3d_picture is None:
                raise SystemExit(f"{self.tag}: a DIII-D synthetic set needs the run's DIII-D picture function")
            stamp = t(z["stamp"][:, mid])
            self.picture = d3d_picture(stamp).float()
            self.ip = stamp[:, IP_COL]
            self.r0, self.coord, self.prior_table = None, None, None
        self.shape, self.shape_w, self.shape_sums, self.shape_n = None, {}, {}, 0
        keep = self.split == "train"
        if frac < 1:
            # nested subsets of training origin shots: a shot is kept when its salted hash falls below frac
            import hashlib
            u = np.array([int(hashlib.sha1(f"subset:{o}".encode()).hexdigest()[:8], 16) / 2 ** 32 for o in z["origin"]])
            keep &= u < frac
        rows = {"train": np.nonzero(keep)[0], "heldout": np.nonzero(self.split == "heldout")[0]}
        self.rows = {s: torch.as_tensor(r, device=device) for s, r in rows.items()}
        n_orig = len(set(np.asarray(z["origin"])[rows["train"]].tolist()))
        print(f"synthetic set {self.tag} ({self.machine}): {len(rows['train'])} train from {n_orig} origin shots"
              f"{f' (share {frac:g})' if frac < 1 else ''}, {len(rows['heldout'])} held out, weight {self.weight:g}",
              flush=True)

    def _forward(self, net, idx: torch.Tensor, linear: bool, noise=None):
        kw = {}
        if self.machine == "mast":
            kw = dict(coord=self.coord, prior=prior_from_table(self.prior_table, self.ip[idx], self.rax[idx]), r0=self.r0)
            if linear:
                kw["machine"] = "mast"
        x = self.coils[idx] if noise is None else noise(self.coils[idx])
        psi_n, _ = net(x, self.picture[idx], self.ip[idx], self.rax[idx], **kw)
        return psi_n.float() * net.psi_std + net.pixel_mean

    def batch_idx(self, b: int) -> torch.Tensor | None:
        rows = self.rows["train"]
        if len(rows) == 0:
            return None
        return rows[torch.randint(0, len(rows), (b,), device=rows.device)]

    def enable_shape(self, weights: dict) -> bool:
        """Add the shape losses (shape_loss.py) with these weights; needs a MAST set with shape_targets.npz."""
        if self.machine != "mast" or not hasattr(self, "psi_ax"):
            return False
        from shape_loss import ShapeLoss
        g = self.geom
        import mast_region
        self.shape = ShapeLoss(g["r"], g["z"], mast_region.mask(), g["a"], self.psi.device)   # the run's MAST region
        self.shape_w = {k: float(v) for k, v in weights.items() if float(v)}
        return True

    def take_shape_sums(self) -> dict:
        """Mean of each shape term since the last call (for the epoch record)."""
        out = {k: v / max(self.shape_n, 1) for k, v in self.shape_sums.items()}
        self.shape_sums, self.shape_n = {}, 0
        return out

    def loss(self, net, idx: torch.Tensor, linear: bool, noise=None) -> torch.Tensor:
        """`noise`, if given, perturbs the standardised inputs (train_mm.py --thomson-noise)."""
        phys = self._forward(net, idx, linear, noise)
        with torch.autocast(device_type=phys.device.type, enabled=False):
            scale = flux_scale(self.ip[idx], self.r0 if self.r0 is not None else net.r0).view(-1, 1)
            out = ((phys.float() - self.psi[idx]) / scale).square().mean()
            if self.shape is not None and self.shape_w:
                extra, vals = self.shape(phys, self.psi[idx], self.psi_ax[idx], self.psi_bnd[idx],
                                         self.scal[idx, 3], self.scal[idx, 4], self.shape_w)
                out = out + extra
                for k, v in vals.items():
                    self.shape_sums[k] = self.shape_sums.get(k, 0.0) + v
                self.shape_n += 1
            return out

    @torch.no_grad()
    def evaluate(self, net, linear: bool, split: str = "heldout", batch: int = 256) -> dict:
        """Pooled R²ψ in physical units over the split, and the mean scaled loss."""
        rows = self.rows[split]
        if len(rows) == 0:
            return {"r2_psi": float("nan"), "loss": float("nan"), "n": 0}
        net.eval()
        ss_res = ss_y = s_y = 0.0
        n_pix = 0
        loss_sum = 0.0
        for i in range(0, len(rows), batch):
            idx = rows[i:i + batch]
            phys = self._forward(net, idx, linear).double()
            y = self.psi[idx].double()
            ss_res += float((phys - y).square().sum())
            s_y += float(y.sum()); ss_y += float(y.square().sum()); n_pix += y.numel()
            scale = flux_scale(self.ip[idx], self.r0 if self.r0 is not None else net.r0).view(-1, 1).double()
            loss_sum += float(((phys - y) / scale).square().mean(1).sum())
        var = ss_y - s_y ** 2 / n_pix
        return {"r2_psi": 1.0 - ss_res / var if var > 0 else float("nan"), "loss": loss_sum / len(rows), "n": len(rows)}
