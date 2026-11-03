"""Label-free loss: the plasma current should be a smooth function of the map's own flux (train.py --lambda-fluxfn).

In an equilibrium the current density is J(R, ψ) = R·p'(ψ) + FF'(ψ) / (μ0 R): on each flux surface it depends on R
only through those two terms, and it falls to zero at the plasma boundary. The linear decoder may place current in
any envelope cell, so nothing ties its edge to its core. Per frame this loss fits

    J_cell / |Σ J|  ≈  Σ_k  [ a_k · (R / R0) + b_k · (R0 / R) ] · h_k(u)

where u = (φ_axis − φ) / (φ_axis − φ_edge) is the normalised flux (φ = ψ signed so the axis is its maximum, φ_edge
the largest φ on the envelope's outermost cells: the first surface that touches the envelope), and h_k are
piecewise-linear hats on knots 0 … 1, zero for u ≥ 1. The loss is the share of Σ J² the fit cannot explain:
current off the flux-function form, and all current beyond the first surface touching the envelope.
u is computed from the map without gradient; the gradient reaches J (and through the projection, the fit).
Frames with |Ip| below `min_ip_ka` are skipped. The share is of Σ J², so a thin haze of small currents over many
cells costs little; `out_weight` > 0 adds that weight times the share of Σ |J| beyond the first touching surface.

With `cold` (Thomson Te, R, Z per frame, from build_thomson_raw.py, rows aligned with the `frames` passed in), the
boundary is also the first surface through a channel with Te ≤ `cold_te` eV: φ_edge becomes the larger of the
envelope level and the `cold_rank`-th highest φ over those channels (rank 2 ignores one stray reading). Without it
the loss fixes the current's shape but not the plasma's size: models trained with it grew the MAST plasma ~9 cm.

With `thomson` (Te, ne, R, Z per frame, rows aligned with `frames`) and `pressure_weight` > 0, the R-term of the fit
is R·p'(ψ), so the implied pressure is P(u) = ∫_u^1 a(u') du' (zero at the boundary). Thomson's p_e = ne·Te at
channels inside the boundary (0 ≤ u < 1) should equal s·P(u) with one s ≥ 0 per frame (ions, units); the term is
the unexplained share of p_e's variance, 1 − R², over frames with ≥ 6 such channels. On DIII-D val labels its R² is
0.86 (median); on MAST maps of earlier models it was −1.5 and −0.5 (mast_diag/pressure_match_check.py).

`edge_ext` > 0 lets the fitted profiles run past the first touching surface to u = 1 + edge_ext (current beyond
that counts as outside). With 0 the fit forces the current to zero at the boundary, which on DIII-D lowers the edge
field 2–3% and raises li by about +0.02 (li_study/REPORT.md); real plasmas carry current to the edge.

`spread_weight` > 0 adds that weight × relu(spread_floor − s)², s = current-weighted rms distance of |J| from its
centroid divided by `a_norm`, the machine's nominal minor radius. (The plasma's own boundary is no reference: in a
collapse it shrinks with the current; spread / boundary half-width stayed ≈ 0.76 on collapsed MAST frames, while
spread / nominal a is 0.26 there against ≥ 0.44 on normal frames.) On MAST the loss
without the pressure term collapses the current to a filament around the axis (li ≈ 2.8 on 92% of frames); a floor
below every DIII-D true map removes only that collapse. `last_low` is the share of frames below the floor.
"""
from __future__ import annotations

import numpy as np
import torch


class FluxFunctionLoss:
    def __init__(self, r, z, mask, idx, r0: float, device, n_knots: int = 8, ridge: float = 1e-3,
                 min_ip_ka: float = 150.0, out_weight: float = 0.0, cold: dict | None = None,
                 cold_te: float = 20.0, cold_rank: int = 2, thomson: dict | None = None,
                 pressure_weight: float = 0.0, edge_ext: float = 0.0, spread_weight: float = 0.0,
                 spread_floor: float = 0.0, a_norm: float = 1.0):
        mask = np.asarray(mask, bool)
        inner = mask.copy()
        inner[1:] &= mask[:-1]; inner[:-1] &= mask[1:]; inner[:, 1:] &= mask[:, :-1]; inner[:, :-1] &= mask[:, 1:]
        rr = np.meshgrid(np.asarray(r, np.float64), np.asarray(z, np.float64), indexing="xy")[0].ravel()
        idx = np.asarray(idx, np.int64)
        self.idx = torch.as_tensor(idx, device=device)
        self.edge = torch.as_tensor((mask & ~inner).ravel()[idx], device=device)
        zz = np.meshgrid(np.asarray(r, np.float64), np.asarray(z, np.float64), indexing="xy")[1].ravel()
        self.R = torch.as_tensor(rr[idx], dtype=torch.float32, device=device)
        self.Z = torch.as_tensor(zz[idx], dtype=torch.float32, device=device)
        self.ext, self.sw, self.sfloor, self.a_norm = float(edge_ext), float(spread_weight), float(spread_floor), float(a_norm)
        self.last_low, self.last_spread = float("nan"), None
        rn = torch.as_tensor(rr[idx] / r0, dtype=torch.float32, device=device)
        self.rr = torch.stack([rn, 1.0 / rn])                                   # (2, N)
        self.knots = torch.linspace(0.0, 1.0, n_knots, device=device)
        self.width = 1.0 / (n_knots - 1)
        self.ridge, self.min_ip, self.out_weight = ridge, min_ip_ka, out_weight
        self.cold = None
        if cold is not None:
            r = np.asarray(r, np.float64); z = np.asarray(z, np.float64)
            te, cr, cz = (np.asarray(cold[k], np.float32) for k in ("te", "r", "z"))
            ir = np.clip(np.round((cr - r[0]) / (r[1] - r[0])), 0, len(r) - 1).astype(int)
            iz = np.clip(np.round((cz - z[0]) / (z[1] - z[0])), 0, len(z) - 1).astype(int)
            ok = np.isfinite(te) & np.isfinite(cr) & np.isfinite(cz) & (te <= cold_te) & inner[iz, ir]
            gx = 2 * (np.nan_to_num(cr) - r[0]) / (r[-1] - r[0]) - 1
            gy = 2 * (np.nan_to_num(cz) - z[0]) / (z[-1] - z[0]) - 1
            self.cold = torch.as_tensor(ok, device=device)
            self.cold_grid = torch.as_tensor(np.stack([gx, gy], -1).astype(np.float32), device=device)
            self.cold_rank = int(cold_rank)
        self.pw, self.th = float(pressure_weight), None
        self.last_pressure = float("nan")
        if thomson is not None and self.pw > 0:
            r = np.asarray(r, np.float64); z = np.asarray(z, np.float64)
            te, ne, cr, cz = (np.asarray(thomson[k], np.float32) for k in ("te", "ne", "r", "z"))
            ir = np.clip(np.round((np.nan_to_num(cr) - r[0]) / (r[1] - r[0])), 0, len(r) - 1).astype(int)
            iz = np.clip(np.round((np.nan_to_num(cz) - z[0]) / (z[1] - z[0])), 0, len(z) - 1).astype(int)
            pe = te * ne
            ok = np.isfinite(pe) & np.isfinite(cr) & np.isfinite(cz) & (pe >= 0) & inner[iz, ir]
            with np.errstate(all="ignore"):
                p95 = np.nanpercentile(np.where(ok, pe, np.nan), 95, axis=1, keepdims=True)
            ok &= pe <= 1.5 * np.nan_to_num(p95, nan=np.inf)
            self.th = torch.as_tensor(ok, device=device)
            self.th_pe = torch.as_tensor(np.where(ok, pe, 0.0).astype(np.float32), device=device)
            gx = 2 * (np.nan_to_num(cr) - r[0]) / (r[-1] - r[0]) - 1
            gy = 2 * (np.nan_to_num(cz) - z[0]) / (z[-1] - z[0]) - 1
            self.th_grid = torch.as_tensor(np.stack([gx, gy], -1).astype(np.float32), device=device)
            # ∫_u^1 h_k on a table of u (201 points) for each hat, trapezoid rule.
            uu = np.linspace(0, 1, 201)
            kn = np.linspace(0, 1, n_knots)
            hat = np.clip(1 - np.abs(uu[:, None] - kn) * (n_knots - 1), 0, None)
            seg = (hat[1:] + hat[:-1]) / 2 * np.diff(uu)[:, None]
            table = np.concatenate([np.cumsum(seg[::-1], 0)[::-1], np.zeros((1, n_knots))], 0)
            self.th_table = torch.as_tensor(table.astype(np.float32), device=device)   # (201, K)

    def __call__(self, phys: torch.Tensor, J: torch.Tensor, ip_ka: torch.Tensor, frames: torch.Tensor | None = None):
        """phys (B, 4225) ψ, J (B, N) current per envelope cell, ip_ka (B,) -> (loss, unexplained share, share of
        |J| beyond the first surface touching the envelope); the shares are means over the frames used (floats)."""
        keep = ip_ka.abs() >= self.min_ip
        if not bool(keep.any()):
            z = J.new_zeros(())
            return z, float("nan"), float("nan")
        if self.cold is not None or self.th is not None:
            frames = frames[keep]
        J = J[keep].float()
        tot = J.sum(1, keepdim=True)
        y = J / tot.abs().clamp_min(1e-6)
        with torch.no_grad():
            phi = phys[keep].float()[:, self.idx] * torch.sign(tot)
            ax = phi.max(1, keepdim=True).values
            bd = phi.masked_fill(~self.edge, float("-inf")).max(1, keepdim=True).values
            if self.cold is not None:
                full = phys[keep].float() * torch.sign(tot)
                val = torch.nn.functional.grid_sample(full.view(-1, 1, 65, 65), self.cold_grid[frames].unsqueeze(1),
                                                      align_corners=True).view(full.shape[0], -1)
                val = val.masked_fill(~self.cold[frames], float("-inf"))
                k = min(self.cold_rank, val.shape[1])
                cb = val.topk(k, dim=1).values[:, k - 1:k]
                bd = torch.maximum(bd, torch.where(torch.isfinite(cb) & (cb < ax), cb, bd))
            u = (ax - phi) / (ax - bd).clamp_min(1e-9) / (1.0 + self.ext)
            h = (1.0 - (u.unsqueeze(-1) - self.knots).abs() / self.width).clamp_min(0.0) * (u < 1).unsqueeze(-1)
            basis = torch.cat([h * self.rr[0].view(1, -1, 1), h * self.rr[1].view(1, -1, 1)], -1)   # (B, N, 2K)
            gram = basis.transpose(1, 2) @ basis
            lam = self.ridge * gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)
            gram = gram + lam.view(-1, 1, 1) * torch.eye(gram.shape[-1], device=gram.device)
            proj = torch.linalg.solve(gram, basis.transpose(1, 2))                                  # (B, 2K, N)
            outside = u >= 1
        fit = (basis @ (proj @ y.unsqueeze(-1))).squeeze(-1)
        resid = (y - fit).square().sum(1) / y.square().sum(1).clamp_min(1e-12)
        out_share = (y.abs() * outside).sum(1) / y.abs().sum(1).clamp_min(1e-12)
        extra = resid.new_zeros(())
        if self.th is not None:
            with torch.no_grad():
                full = phys[keep].float() * torch.sign(tot)
                val = torch.nn.functional.grid_sample(full.view(-1, 1, 65, 65), self.th_grid[frames].unsqueeze(1),
                                                      align_corners=True).view(full.shape[0], -1)
                uc = (ax - val) / (ax - bd).clamp_min(1e-9) / (1.0 + self.ext)
                m = (self.th[frames] & (uc >= 0) & (uc < 1.0 / (1.0 + self.ext))).float()
                pos = uc.clamp(0, 1) * 200
                i0 = pos.floor().long().clamp(max=199); fr = (pos - i0).unsqueeze(-1)
                H = self.th_table[i0] * (1 - fr) + self.th_table[i0 + 1] * fr                       # (B, C, K)
                pe = self.th_pe[frames] * m
                n = m.sum(1)
                mean = pe.sum(1) / n.clamp_min(1)
                var = ((pe - mean.unsqueeze(1)) * m).square().sum(1)
                use = (n >= 6) & (var > 0)
            k = self.knots.shape[0]
            a = (proj @ (y * torch.sign(tot)).unsqueeze(-1)).squeeze(-1)[:, :k]                    # R·p' coefficients
            P = (H @ a.unsqueeze(-1)).squeeze(-1) * m
            sc = torch.relu((pe * P).sum(1)) / P.square().sum(1).clamp_min(1e-12)
            pres = ((pe - sc.unsqueeze(1) * P) * m).square().sum(1) / var.clamp_min(1e-12)
            if bool(use.any()):
                pl = pres[use].clamp(max=4.0).mean()
                extra = self.pw * pl
                self.last_pressure = float(pl.detach())
        if self.sw > 0 or self.sfloor > 0:
            w = y.abs()
            wn = w / w.sum(1, keepdim=True).clamp_min(1e-12)
            cR = (wn * self.R).sum(1, keepdim=True); cZ = (wn * self.Z).sum(1, keepdim=True)
            spread = ((wn * ((self.R - cR) ** 2 + (self.Z - cZ) ** 2)).sum(1)).clamp_min(1e-12).sqrt()
            s_a = spread / self.a_norm
            self.last_spread = s_a.detach()
            self.last_low = float((s_a.detach() < self.sfloor).float().mean())
            if self.sw > 0:
                extra = extra + self.sw * torch.relu(self.sfloor - s_a).square().mean()
        return resid.mean() + self.out_weight * out_share.mean() + extra, float(resid.detach().mean()), float(out_share.detach().mean())
