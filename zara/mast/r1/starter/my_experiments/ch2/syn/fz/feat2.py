"""ch2/syn: second group of physics-derived, machine-agnostic input embeddings (user idea, 2026-10-07).

Appended to embed.NAMES when a model is trained with --feat2 1 (computed on the GPU from the cached raw inputs and first-pass
geometry when a FrameSet is loaded; no cache rebuild). Every feature is dimensionless and carries a Z-parity label.

What embed.py already has (not repeated here): first-pass-based electron beta_p / beta_t, Te / ne / pe peaking on first-pass
psi_N, binned Te / ne shapes, Te-peak radius, q* and Uckan q95 from B_T (TF current) and Ip, vacuum-field harmonics, B_Z / B_R
at the first-pass centre, Fourier modes of the coil flux on the first-pass LCFS (odd modes = up-down asymmetry).

New here:
  Thomson chord moments WITHOUT the first pass (the first pass misplaces 10-20 % of MAST frames, so first-pass-referenced
  features inherit its error). Coordinates along the chord, normalised by the hot extent (ne > 0.2 ne_max): half-length
  h = L / 2, centre (R_mid, Z_mid).
    pe_cen_R / pe_cen_Z   electron-pressure-weighted centroid offset from the hot-extent centre, / h  (Shafranov-shift proxy;
                          R even, Z odd)
    pe_pk_R / pe_pk_Z     location of the smoothed pe maximum, same normalisation (R even, Z odd)
    pe_width              pe-weighted rms distance from the centroid, / h (even)
    pe_peaked             log(pe_max / chord-mean pe over the hot extent) (even)
    pe_ped                mean pe at 0.70-0.95 h from the centre / pe_max (pedestal height proxy, even)
    te_ped                same for Te (even)
    betap_chord           log(2 mu0 <pe>_chord / B_pc^2), B_pc = mu0 Ip / (pi L): stored-energy proxy (chord integral of
                          ne Te) in units of the poloidal field (even)
    betat_chord           log(2 mu0 <pe>_chord / B_T^2), B_T at the pe centroid radius from the TF current (even)
    log_qchord            log(h B_T / (R_c B_pc)): cylindrical q proxy, B_T / B_p from TF current, Ip and the chord (even)
  Pure input ratios (no geometry):
    log_ntf_ip            log(N_TF I_TF / Ip) = B_T R / (mu0 Ip / 2 pi)  (q proxy; even)
  External poloidal field over the first-pass plasma cells, units of Bpbar = mu0 Ip / l_p:
    log_evac              log mean (B_R^2 + B_Z^2) / Bpbar^2: external poloidal-field energy density (even)
    log_eshape            same for B - B(centroid): energy of the shaping (gradient) part of the coil field (even)
    ud_asym               (E_up - E_lo) / (E_up + E_lo) of the shaping energy above / below the first-pass axis (odd)
    br_grad               mean dB_R/dZ over the plasma cells x a / Bpbar (vertical-field curvature / decay index proxy; even)
"""
from __future__ import annotations

import math

import torch

from embed import E_KEV
from physics import MU0

NAMES2 = ["pe_cen_R", "pe_cen_Z", "pe_pk_R", "pe_pk_Z", "pe_width", "pe_peaked", "pe_ped", "te_ped", "betap_chord",
          "betat_chord", "log_qchord", "log_ntf_ip", "log_evac", "log_eshape", "ud_asym", "br_grad"]
PARITY2 = [1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1]
assert len(NAMES2) == len(PARITY2)


@torch.no_grad()
def features2(gs, b) -> torch.Tensor:
    """b: FrameSet batch (pts (B, N, 2), ts_te keV, ts_ne 1e19, ip, itf, I, Jfp, fp_*) -> (B, len(NAMES2)), NaN = missing."""
    B = b["ip"].shape[0]
    dev = b["ip"].device
    nan = torch.full((B,), float("nan"), device=dev)
    pts, te, ne = b["pts"], b["ts_te"].float(), b["ts_ne"].float()
    ok = torch.isfinite(pts).all(-1) & torch.isfinite(te) & torch.isfinite(ne) & (te > 0) & (ne > 0)
    R = torch.nan_to_num(pts[..., 0]); Z = torch.nan_to_num(pts[..., 1])
    pe = torch.where(ok, te * ne * 1e19 * E_KEV, torch.zeros_like(te))                 # Pa
    nev = torch.where(ok, ne, torch.full_like(ne, float("nan")))
    ne_ref = torch.nanquantile(nev, 0.95, dim=1)
    hot = ok & (ne > 0.2 * ne_ref[:, None])
    big, small = torch.full_like(R, 1e9), torch.full_like(R, -1e9)
    Rin, Rout = torch.where(hot, R, big).min(1).values, torch.where(hot, R, small).max(1).values
    Zin, Zout = torch.where(hot, Z, big).min(1).values, torch.where(hot, Z, small).max(1).values
    LR, LZ = (Rout - Rin).clamp(min=0), (Zout - Zin).clamp(min=0)
    L = torch.maximum(LR, LZ)
    good = (hot.sum(1) >= 8) & (L > 0.05)
    h = (0.5 * L).clamp(min=0.025)
    Rm, Zm = 0.5 * (Rin + Rout), 0.5 * (Zin + Zout)
    Rm = torch.where(good, Rm, torch.ones_like(Rm)); Zm = torch.where(good, Zm, torch.zeros_like(Zm))
    # chord coordinate along the longer extent; sorted order for the integral and the smoothed peak
    along_R = LR >= LZ
    s = torch.where(along_R[:, None], R, Z)
    s_sorted, order = torch.sort(torch.where(ok, s, big), 1)
    ok_s = ok.gather(1, order)
    pe_s = pe.gather(1, order)
    te_s = torch.where(ok, te, torch.zeros_like(te)).gather(1, order)
    ds = torch.diff(s_sorted, dim=1)
    pair = ok_s[:, 1:] & ok_s[:, :-1] & (ds < 0.2)          # gaps > 20 cm (dropouts, core / edge system gap) are skipped
    seg = torch.where(pair, 0.5 * (pe_s[:, 1:] + pe_s[:, :-1]) * ds, torch.zeros_like(ds))
    lseg = torch.where(pair, ds, torch.zeros_like(ds))
    in_hot_seg = pair & (s_sorted[:, 1:] <= torch.where(along_R, Rout, Zout)[:, None]) & (s_sorted[:, :-1] >= torch.where(along_R, Rin, Zin)[:, None])
    pe_mean = (seg * in_hot_seg).sum(1) / (lseg * in_hot_seg).sum(1).clamp(min=1e-6)    # chord-mean pe over the hot extent
    # smoothed pe (5-point running mean over valid sorted points) -> peak value and location
    w = ok_s.float()
    k = torch.ones(1, 1, 5, device=dev)
    num = torch.nn.functional.conv1d((pe_s * w)[:, None], k, padding=2)[:, 0]
    den = torch.nn.functional.conv1d(w[:, None], k, padding=2)[:, 0]
    pe_sm = torch.where(ok_s & (den >= 3), num / den.clamp(min=1), torch.full_like(num, -1.0))
    pk_val, pk_i = pe_sm.max(1)
    R_s, Z_s = R.gather(1, order), Z.gather(1, order)
    R_pk, Z_pk = R_s.gather(1, pk_i[:, None])[:, 0], Z_s.gather(1, pk_i[:, None])[:, 0]
    W = pe.sum(1).clamp(min=1e-9)
    Rc, Zc = (pe * R).sum(1) / W, (pe * Z).sum(1) / W
    width = torch.sqrt((pe * ((R - Rc[:, None]) ** 2 + (Z - Zc[:, None]) ** 2)).sum(1) / W)
    rho = torch.sqrt((R - Rm[:, None]) ** 2 + (Z - Zm[:, None]) ** 2) / h[:, None]
    ring = ok & (rho >= 0.70) & (rho <= 0.95)
    nring = ring.sum(1)
    te_max = torch.where(ok, te, torch.zeros_like(te)).amax(1).clamp(min=1e-6)
    pe_ped = (pe * ring).sum(1) / nring.clamp(min=1) / pk_val.clamp(min=1e-9)
    te_ped = (torch.where(ok, te, torch.zeros_like(te)) * ring).sum(1) / nring.clamp(min=1) / te_max
    ip = b["ip"].clamp(min=1e4)
    BT_c = MU0 * gs.tf_turns * b["itf"] / (2 * math.pi * Rc.clamp(min=0.1))
    Bpc = MU0 * ip / (math.pi * L.clamp(min=0.05))
    lg = lambda v: torch.log(v.clamp(min=1e-8))
    gate = lambda v, g=good: torch.where(g & torch.isfinite(v), v, nan)
    f = {}
    f["pe_cen_R"] = gate((Rc - Rm) / h); f["pe_cen_Z"] = gate((Zc - Zm) / h)
    pk_ok = good & (pk_val > 0)
    f["pe_pk_R"] = gate((R_pk - Rm) / h, pk_ok); f["pe_pk_Z"] = gate((Z_pk - Zm) / h, pk_ok)
    f["pe_width"] = gate(width / h)
    f["pe_peaked"] = gate(lg(pk_val / pe_mean.clamp(min=1e-6)), pk_ok & (pe_mean > 0))
    f["pe_ped"] = gate(pe_ped, pk_ok & (nring >= 2)); f["te_ped"] = gate(te_ped, good & (nring >= 2))
    f["betap_chord"] = gate(lg(2 * MU0 * pe_mean / Bpc ** 2), good & (pe_mean > 0))
    f["betat_chord"] = gate(lg(2 * MU0 * pe_mean / BT_c.clamp(min=1e-3) ** 2), good & (pe_mean > 0))
    f["log_qchord"] = gate(lg(h * BT_c / (Rc.clamp(min=0.1) * Bpc)))
    f["log_ntf_ip"] = lg(gs.tf_turns * b["itf"] / ip)
    # ---- external poloidal field over the first-pass plasma cells
    vac = gs.vacuum(b["I"]).view(B, gs.nZ, gs.nR)
    gR, gZ = gs.grad(vac)
    BZ, BR = gR / gs.RR, -gZ / gs.RR
    dBRdZ = torch.gradient(BR, spacing=gs.dZ, dim=1)[0]
    flat = gs.flat
    bz_c, br_c, dbr_c = BZ.reshape(B, -1)[:, flat], BR.reshape(B, -1)[:, flat], dBRdZ.reshape(B, -1)[:, flat]   # (B, K)
    wcell = (b["Jfp"].float().abs() > 1e-5).float() * gs.Rk[None]
    nw = wcell.sum(1).clamp(min=1e-9)
    Bpbar = MU0 * ip / b["fp_lp"].clamp(min=0.1)
    mean = lambda v, ww=wcell: (v * ww).sum(1) / ww.sum(1).clamp(min=1e-9)
    e_vac = mean(bz_c ** 2 + br_c ** 2) / Bpbar ** 2
    bz0, br0 = mean(bz_c), mean(br_c)
    esh = ((bz_c - bz0[:, None]) ** 2 + (br_c - br0[:, None]) ** 2) / Bpbar[:, None] ** 2
    up = (gs.Zk[None] > b["fp_Z_axis"][:, None]).float() * wcell
    lo = (gs.Zk[None] < b["fp_Z_axis"][:, None]).float() * wcell
    e_up, e_lo = mean(esh, up), mean(esh, lo)
    has_cells = nw > 0
    f["log_evac"] = gate(lg(e_vac), has_cells)
    f["log_eshape"] = gate(lg(mean(esh)), has_cells)
    f["ud_asym"] = gate((e_up - e_lo) / (e_up + e_lo).clamp(min=1e-9), has_cells & (up.sum(1) > 0) & (lo.sum(1) > 0))
    f["br_grad"] = gate(mean(dbr_c) * b["fp_a"].clamp(min=0.05) / Bpbar, has_cells)
    X2 = torch.stack([f[k].float() for k in NAMES2], 1)
    return torch.where(torch.isfinite(X2), X2, torch.full_like(X2, float("nan")))


@torch.no_grad()
def extend(fs, gs, batch=2048):
    """Append the NAMES2 features to a FrameSet's X in place (once)."""
    if getattr(fs, "feat2", False):
        return fs
    out = []
    for i in range(0, len(fs), batch):
        b = fs.batch(torch.arange(i, min(len(fs), i + batch), device=fs.dev))
        out.append(features2(gs, b))
    X2 = torch.cat(out) if out else fs.d["X"].new_zeros(0, len(NAMES2))
    fs.d["X"] = torch.cat([fs.d["X"], X2], 1)
    fs.feat2 = True
    return fs


def describe(X2: torch.Tensor) -> str:
    """Per-feature coverage and 5/50/95 % percentiles (for the training log)."""
    parts = []
    for j, n in enumerate(NAMES2):
        v = X2[:, j]
        v = v[torch.isfinite(v)]
        if len(v) == 0:
            parts.append(f"{n}: none"); continue
        q = torch.quantile(v[torch.randperm(len(v), device=v.device)[:100000]].float(), torch.tensor([0.05, 0.5, 0.95], device=v.device))
        parts.append(f"{n}: {len(v) / X2.shape[0]:.0%} [{q[0]:.3g} {q[1]:.3g} {q[2]:.3g}]")
    return "; ".join(parts)
