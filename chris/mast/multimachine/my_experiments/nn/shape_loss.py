"""Shape losses on labelled synthetic MAST frames (train_mm.py --synth-shape), aimed at the scorer's Consistency term
(axis, elongation, triangularities, volume, li derived from the flux map) and the boundary distance.

The plain flux loss weighs every pixel alike; the derived quantities depend on small features of the map (where the
flux has its minimum, how steep it is inside, where one contour runs). Each term below is a relative error computed
with the same operator on the predicted and the true map, so label definitions cancel:

  grad   slope of the map, inside the limiter:              Σ|∇(ψ̂−ψ)|² / Σ|∇ψ|²
  jtor   toroidal current density (Δ*ψ ∝ R·J), in plasma:   Σ(Δ*ψ̂−Δ*ψ)² / Σ(Δ*ψ)²
  axis   slope of ψ̂ at the labelled magnetic axis (zero there), over the typical slope |ψ_b−ψ_axis| / a
  bnd    ψ̂ against ψ on pixels near the true boundary (ψ_N 0.85–1.15), over |ψ_b−ψ_axis|
  li     log ratio of ∫|∇ψ|²/R² dV over the true plasma region, ψ̂ against ψ (li ∝ that integral at fixed Ip)

Grid: maps are (B, 65·65) with rows Z and columns R, as model.flux_gradients and fusion_scoring/derive.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

TERMS = ("grad", "jtor", "axis", "bnd", "li")


class ShapeLoss:
    def __init__(self, r, z, inside, a: float, device):
        self.r = torch.as_tensor(r, dtype=torch.float32, device=device)
        self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
        self.inside = torch.as_tensor(inside, dtype=torch.float32, device=device)       # (Z, R)
        self.R = self.r.view(1, 1, -1)
        self.a = float(a)
        n = len(self.r)
        self.n = n

    def _grad(self, psi):
        dz, dr = torch.gradient(psi, spacing=(self.z, self.r), dim=(-2, -1), edge_order=1)
        return dr, dz

    def _dstar(self, psi):
        dr, dz = self._grad(psi)
        drr = torch.gradient(dr, spacing=(self.r,), dim=-1, edge_order=1)[0]
        dzz = torch.gradient(dz, spacing=(self.z,), dim=-2, edge_order=1)[0]
        return drr - dr / self.R + dzz

    def _at(self, field, r_ax, z_ax):
        """Bilinear value of field (B, Z, R) at one point per frame."""
        x = 2 * (r_ax - self.r[0]) / (self.r[-1] - self.r[0]) - 1
        y = 2 * (z_ax - self.z[0]) / (self.z[-1] - self.z[0]) - 1
        grid = torch.stack([x, y], -1).view(-1, 1, 1, 2)
        return F.grid_sample(field.unsqueeze(1), grid, align_corners=True).view(-1)

    def __call__(self, pred_flat, true_flat, psi_ax, psi_bnd, r_ax, z_ax, weights: dict) -> tuple[torch.Tensor, dict]:
        """Weighted sum of the active terms and each term's value (float), all in float32."""
        b = pred_flat.shape[0]
        p = pred_flat.float().view(b, self.n, self.n)
        t = true_flat.float().view(b, self.n, self.n)
        span = (psi_bnd - psi_ax).view(b, 1, 1)                        # < 0 in our sign; ψ_N = (ψ−ψ_ax)/span
        psin = (t - psi_ax.view(b, 1, 1)) / span
        plasma = ((psin < 1) & (psin >= 0)).float() * self.inside
        vals, total = {}, p.new_zeros(())
        need_grad = any(weights.get(k, 0) for k in ("grad", "li", "axis"))
        if need_grad:
            pr, pz = self._grad(p)
            tr, tz = self._grad(t)
        if weights.get("grad", 0):
            m = self.inside
            vals["grad"] = (((pr - tr) ** 2 + (pz - tz) ** 2) * m).sum() / ((tr ** 2 + tz ** 2) * m).sum().clamp_min(1e-12)
        if weights.get("jtor", 0):
            dp, dt = self._dstar(p), self._dstar(t)
            vals["jtor"] = (((dp - dt) ** 2) * plasma).sum() / ((dt ** 2) * plasma).sum().clamp_min(1e-12)
        if weights.get("axis", 0):
            gr, gz = self._at(pr, r_ax, z_ax), self._at(pz, r_ax, z_ax)
            typ = (span.view(b).abs() / self.a).clamp_min(1e-6)
            vals["axis"] = ((gr ** 2 + gz ** 2) / typ ** 2).mean()
        if weights.get("bnd", 0):
            band = ((psin > 0.85) & (psin < 1.15)).float() * self.inside
            d = (p - t) / span.abs()
            vals["bnd"] = ((d ** 2) * band).sum() / band.sum().clamp_min(1.0)
        if weights.get("li", 0):
            w = plasma / self.R                                         # |∇ψ|²/R² · R dR dZ
            ip_ = ((pr ** 2 + pz ** 2) * w).sum((-2, -1))
            it_ = ((tr ** 2 + tz ** 2) * w).sum((-2, -1)).clamp_min(1e-12)
            vals["li"] = torch.log(ip_.clamp_min(1e-12) / it_).square().mean()
        for k, v in vals.items():
            total = total + float(weights[k]) * v
        return total, {k: float(v.detach()) for k, v in vals.items()}
