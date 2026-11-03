""""Group B" inputs: a first-pass equilibrium read off a flux map (the Ridge prediction), batched in torch.

    fp = FirstPass(grid_R, grid_Z, envelope, device)
    feats = fp(P, ip, bt, bzbip, pos, te, ne)       # (B, len(fp.names)), NaN where undefined

P: (B, 4225) flux maps in the shipped DIII-D convention (axis = minimum). ip [A], bt [A] (toroidal-field coil
current), bzbip: the `bz_over_bip` field feature, pos (B, C, 2) Thomson channel R, Z, te keV, ne 1e19 m^-3 (B, C).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

MU0 = 4e-7 * math.pi
N_TF = 144
PE_PA = 1.602176634e3                     # keV x 1e19 m^-3 -> Pa
BINS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.15]


def _erode(m, n):
    for _ in range(n):
        e = m.copy(); e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]; m = e
    return m


class FirstPass:
    def __init__(self, grid_R, grid_Z, envelope, device):
        self.dev = device
        self.gR = torch.tensor(np.asarray(grid_R, np.float32), device=device)
        self.gZ = torch.tensor(np.asarray(grid_Z, np.float32), device=device)
        self.dR, self.dZ = float(grid_R[1] - grid_R[0]), float(grid_Z[1] - grid_Z[0])
        ZZ, RR = torch.meshgrid(self.gZ, self.gR, indexing="ij")
        self.RR, self.ZZ = RR, ZZ
        env = np.asarray(envelope, bool)
        self.env = torch.tensor(env, device=device)
        er2 = _erode(env, 2); er2[0, :] = er2[-1, :] = er2[:, 0] = er2[:, -1] = False
        self.env2 = torch.tensor(er2, device=device)
        self.edges = torch.tensor(BINS, device=device)
        nb = len(BINS) - 1
        self.names = (["fp_Raxis", "fp_Zaxis", "fp_psiaxis",
                       "fp_RXlow", "fp_ZXlow", "fp_dpsiXlow", "fp_RXup", "fp_ZXup", "fp_dpsiXup", "fp_dpsi", "fp_updown",
                       "fp_Rmin", "fp_Rmax", "fp_Zmin", "fp_Zmax", "fp_a", "fp_Rgeo", "fp_kappa", "fp_delta_up", "fp_delta_low",
                       "fp_area", "fp_volume", "fp_Rcentroid", "fp_Zcentroid", "fp_RJ", "fp_ZJ", "fp_Jpeak", "fp_Jtot_over_ip",
                       "fp_li", "fp_qcyl", "fp_q95_iter"]
                      + [f"fp_te_bin{b}" for b in range(nb)] + [f"fp_ne_bin{b}" for b in range(nb)]
                      + ["fp_te_vol", "fp_ne_vol", "fp_pe_vol", "fp_beta_t", "fp_betaN", "fp_beta_p", "fp_W_MJ",
                         "fp_greenwald", "fp_pe_peak", "fp_shafranov_lambda", "fp_li_shafranov", "fp_ts_nvalid",
                         "fp_bzvac_axis_over_bip", "fp_brvac_axis_over_bip", "fp_decay_index_axis",
                         "fp_shafranov_lambda_axis", "fp_li_shafranov_axis"])

    @torch.no_grad()
    def __call__(self, P, ip, bt, bzbip, pos, te, ne, Pvac=None):
        B = P.shape[0]
        nan = torch.tensor(float("nan"), device=self.dev)
        P3 = P.view(B, 65, 65).float()
        RR, ZZ = self.RR, self.ZZ
        # ---- O-point (axis = minimum of psi), as the scorer finds it
        phi = -P3[:, None]
        mp = F.max_pool2d(phi, 3, stride=1, padding=1)
        lm = (phi >= mp)[:, 0] & self.env2
        cand = torch.where(lm, -P3, torch.full_like(P3, -float("inf"))).view(B, -1)
        none = torch.isinf(cand.max(1).values)
        idx = cand.argmax(1)
        if none.any():
            g = torch.where(self.env, -P3[none], torch.full_like(P3[none], -float("inf"))).view(int(none.sum()), -1)
            idx[none] = g.argmax(1)
        iz, ir = idx // 65, idx % 65
        ar = torch.arange(B, device=self.dev)
        izc, irc = iz.clamp(1, 63), ir.clamp(1, 63)
        f0 = -P3[ar, izc, irc]
        def off(a_, c_):
            den = a_ - 2 * f0 + c_
            return torch.where(den == 0, torch.zeros_like(f0), (0.5 * (a_ - c_) / den).clamp(-1, 1))
        dz = off(-P3[ar, izc - 1, irc], -P3[ar, izc + 1, irc]); dr = off(-P3[ar, izc, irc - 1], -P3[ar, izc, irc + 1])
        Rax, Zax = self.gR[ir] + dr * self.dR, self.gZ[iz] + dz * self.dZ
        psi_ax = P3[ar, iz, ir]
        # ---- gradients, Hessian, Grad-Shafranov operator
        gZ, gRd = torch.gradient(P3, spacing=(self.dZ, self.dR), dim=(1, 2))
        gZZ, gZR = torch.gradient(gZ, spacing=(self.dZ, self.dR), dim=(1, 2))
        gRZ, gRR = torch.gradient(gRd, spacing=(self.dZ, self.dR), dim=(1, 2))
        det = gRR * gZZ - 0.25 * (gRZ + gZR) ** 2
        g2 = gRd ** 2 + gZ ** 2
        dstar = gRR - gRd / RR + gZZ                                  # Delta* psi
        # ---- X-points: minimum |grad psi| among saddle cells, lower and upper half
        def xpoint(sel):
            sc = torch.where(self.env2 & (det < 0) & sel, g2, torch.full_like(g2, float("inf"))).view(B, -1)
            k = sc.argmin(1); okx = torch.isfinite(sc.min(1).values)
            kz, kr = k // 65, k % 65
            return (torch.where(okx, self.gR[kr], nan), torch.where(okx, self.gZ[kz], nan),
                    torch.where(okx, P3[ar, kz, kr], nan))
        RXl, ZXl, psXl = xpoint(ZZ < -0.3)
        RXu, ZXu, psXu = xpoint(ZZ > 0.3)
        psi_sep = torch.fmin(psXl, psXu)
        dpsi = psi_sep - psi_ax
        # ---- plasma region: inside envelope, psi below the separatrix, between the X-points
        zlo = torch.where(torch.isfinite(ZXl), ZXl, torch.full_like(ZXl, -9.0))
        zhi = torch.where(torch.isfinite(ZXu), ZXu, torch.full_like(ZXu, 9.0))
        inside = (self.env[None] & (P3 < psi_sep[:, None, None]) & (ZZ[None] > zlo[:, None, None])
                  & (ZZ[None] < zhi[:, None, None]) & torch.isfinite(psi_sep)[:, None, None])
        n_in = inside.sum((1, 2)).float()
        has = n_in > 4
        inf = torch.full_like(P3, float("inf"))
        Rmin = torch.where(inside, RR, inf).amin((1, 2)); Rmax = torch.where(inside, RR, -inf).amax((1, 2))
        Zmin = torch.where(inside, ZZ, inf).amin((1, 2)); Zmax = torch.where(inside, ZZ, -inf).amax((1, 2))
        a = 0.5 * (Rmax - Rmin + self.dR); Rgeo = 0.5 * (Rmax + Rmin)
        kappa = (Zmax - Zmin + self.dZ) / (2 * a)
        top = inside & (ZZ[None] >= (Zmax - 0.5 * self.dZ)[:, None, None])
        bot = inside & (ZZ[None] <= (Zmin + 0.5 * self.dZ)[:, None, None])
        Rtop = (RR * top).sum((1, 2)) / top.sum((1, 2)).clamp(min=1)
        Rbot = (RR * bot).sum((1, 2)) / bot.sum((1, 2)).clamp(min=1)
        d_up, d_lo = (Rgeo - Rtop) / a, (Rgeo - Rbot) / a
        dA = self.dR * self.dZ
        w = inside.float()
        area = n_in * dA
        vol = (2 * math.pi * RR * w).sum((1, 2)) * dA
        Rc = (RR * w).sum((1, 2)) / n_in.clamp(min=1); Zc = (ZZ * w).sum((1, 2)) / n_in.clamp(min=1)
        J = dstar / RR                                               # toroidal current density, up to sign and mu0
        Jw = J * w
        Jsum = Jw.sum((1, 2))
        RJ = (RR * Jw).sum((1, 2)) / Jsum; ZJ = (ZZ * Jw).sum((1, 2)) / Jsum
        Jpeak = J[ar, iz, ir] / (Jsum / n_in.clamp(min=1))
        Jtot = (Jsum * dA).abs() / (MU0 * ip.abs().clamp(min=1.0))
        lp = 2 * math.pi * a * torch.sqrt((1 + kappa ** 2) / 2)
        Bp2 = g2 / RR ** 2
        Bp2V = (Bp2 * RR * w).sum((1, 2)) / (RR * w).sum((1, 2)).clamp(min=1e-9)
        Bpa = MU0 * ip.abs() / lp
        li = Bp2V / Bpa ** 2
        btT = MU0 * N_TF * bt.abs() / (2 * math.pi * Rgeo)
        IpMA = ip.abs() / 1e6
        eps = a / Rgeo
        dl = 0.5 * (d_up + d_lo)
        qcyl = 5 * a ** 2 * btT * (1 + kappa ** 2) / (2 * Rgeo * IpMA)
        q95 = (5 * a ** 2 * btT / (Rgeo * IpMA) * (1 + kappa ** 2 * (1 + 2 * dl ** 2 - 1.2 * dl ** 3)) / 2
               * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2)
        # ---- Thomson channels in normalised flux
        g = torch.stack([(pos[..., 0] - self.gR[0]) / (self.gR[-1] - self.gR[0]) * 2 - 1,
                         (pos[..., 1] - self.gZ[0]) / (self.gZ[-1] - self.gZ[0]) * 2 - 1], -1)
        psi_ch = F.grid_sample(P3[:, None], torch.nan_to_num(g, nan=-2.0)[:, None], mode="bilinear",
                               align_corners=True, padding_mode="border")[:, 0, 0]
        psiN = (psi_ch - psi_ax[:, None]) / dpsi[:, None]
        valid = torch.isfinite(te) & torch.isfinite(ne) & torch.isfinite(psiN) & torch.isfinite(pos).all(-1) \
            & (psiN >= self.edges[0]) & (psiN < self.edges[-1])
        nb = len(BINS) - 1
        b = (torch.bucketize(torch.nan_to_num(psiN, nan=-1.0), self.edges, right=True) - 1).clamp(0, nb - 1)
        oh = F.one_hot(b, nb).float() * valid[..., None].float()               # (B, C, nb)
        cnt = oh.sum(1)
        te_b = (torch.nan_to_num(te)[..., None] * oh).sum(1) / cnt.clamp(min=1)
        ne_b = (torch.nan_to_num(ne)[..., None] * oh).sum(1) / cnt.clamp(min=1)
        te_b = torch.where(cnt > 0, te_b, nan); ne_b = torch.where(cnt > 0, ne_b, nan)
        pe_b = te_b * ne_b
        psiN_grid = (P3 - psi_ax[:, None, None]) / dpsi[:, None, None]
        bg = (torch.bucketize(torch.nan_to_num(psiN_grid, nan=-1.0).contiguous(), self.edges, right=True) - 1).clamp(0, nb - 1)
        wg = F.one_hot(bg, nb).float() * (w * RR)[..., None]
        vw = wg.sum((1, 2))                                                       # (B, nb) volume weights
        okb = (cnt > 0).float() * vw
        def vavg(x):
            return (torch.nan_to_num(x) * okb).sum(1) / okb.sum(1)
        te_v, ne_v, pe_v = vavg(te_b), vavg(ne_b), vavg(pe_b)
        pPa = 2 * pe_v * PE_PA                                                    # electrons + ions (Ti ~ Te)
        beta_t = 2 * MU0 * pPa / btT ** 2
        betaN = 100 * beta_t * a * btT / IpMA
        beta_p = 2 * MU0 * pPa / Bpa ** 2
        W = 1.5 * pPa * vol / 1e6
        fG = (ne_v / 10) / (IpMA / (math.pi * a ** 2))
        pe_pk = pe_b[:, 0] / pe_v
        lam = bzbip - torch.log(8 * Rgeo / (a * torch.sqrt(kappa))) + 1.5
        li_sh = 2 * (lam - beta_p)
        # ---- coil (vacuum) field at the first-pass magnetic axis: radial force balance there
        if Pvac is not None:
            V3 = Pvac.view(B, 65, 65).float()
            vZ, vR = torch.gradient(V3, spacing=(self.dZ, self.dR), dim=(1, 2))
            BZv, BRv = vR / RR, -vZ / RR
            dBZdR = torch.gradient(BZv, spacing=(self.dZ, self.dR), dim=(1, 2))[1]
            gax = torch.stack([(Rax - self.gR[0]) / (self.gR[-1] - self.gR[0]) * 2 - 1,
                               (Zax - self.gZ[0]) / (self.gZ[-1] - self.gZ[0]) * 2 - 1], -1)[:, None, None]
            smp = F.grid_sample(torch.stack([BZv, BRv, dBZdR], 1), gax, mode="bilinear", align_corners=True)[:, :, 0, 0]
            bip = MU0 * ip / (4 * math.pi * Rax)
            bz_ax, br_ax = smp[:, 0] / bip, smp[:, 1] / bip
            n_ax = -Rax * smp[:, 2] / smp[:, 0]
            lam_ax = bz_ax - torch.log(8 * Rax / (a * torch.sqrt(kappa))) + 1.5
            li_ax = 2 * (lam_ax - beta_p)
        else:
            bz_ax = br_ax = n_ax = lam_ax = li_ax = torch.full_like(Rax, float("nan"))
        cols = [Rax, Zax, psi_ax, RXl, ZXl, psXl - psi_ax, RXu, ZXu, psXu - psi_ax, dpsi, (psXu - psXl) / dpsi,
                Rmin, Rmax, Zmin, Zmax, a, Rgeo, kappa, d_up, d_lo, area, vol, Rc, Zc, RJ, ZJ, Jpeak, Jtot, li, qcyl, q95]
        out = torch.cat([torch.stack(cols, 1), te_b, ne_b,
                         torch.stack([te_v, ne_v, pe_v, beta_t, betaN, beta_p, W, fG, pe_pk, lam, li_sh, valid.sum(1).float(),
                                      bz_ax, br_ax, n_ax, lam_ax, li_ax], 1)], 1)
        geo = list(range(11, 31))
        out[:, geo] = torch.where(has[:, None], out[:, geo], nan)
        out[~torch.isfinite(out)] = float("nan")
        return out
