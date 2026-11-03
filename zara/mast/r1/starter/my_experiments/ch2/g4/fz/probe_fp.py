"""ch2/intr probe: which equilibrium physics lets real-MAST frames be diverted with the outer midplane boundary on the Thomson
30 eV edge and the inner edge off the centre column? Label-free (MAST inputs + Thomson only).

For each selected frame (ramp-up = first fifth of the shot, flat-top = 40-60 %), the GS layer's converged solve (embed.first_pass
machinery, no network) over a grid of current shapes (incl. hollow, as in a fast current ramp), pressure shares and uniform
vertical-field corrections bv. Records diverted / residual / inner and outer midplane boundary / axis per candidate.
    python probe_fp.py <stems file> <out.npz> [--every 2]"""
import argparse
import itertools
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shots as S                                   # noqa: E402
from build_cache import machine_for                 # noqa: E402
from physics import GSLayer                         # noqa: E402

SHAPES = {"lin": lambda x: 1 - x, "broad": lambda x: 1 - x ** 2, "flat": lambda x: 1 - x ** 4,
          "hollow": lambda x: 4 * x * (1 - x) + 0.05, "hollow2": lambda x: 6.75 * x ** 2 * (1 - x) + 0.05}
SHARES = [0.0, 0.35, 0.7, 1.2, 2.0, 3.0]
BVS = [0.0, 3.0, 6.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device("cuda")
    m = machine_for("MAST", 0.5); gs = GSLayer(m, dev)
    I_l, ip_l, tin_l, tout_l, sat_l, ph_l, sid_l, k_l = [], [], [], [], [], [], [], []
    for si, s in enumerate(a.stems.read_text().split()):
        r = S.mast_shot(s, "test"); T = len(r["ip"])
        R = r["ts_pts"][:, 0]; rmin = np.nanmin(R)
        for k in range(0, T, a.every):
            f = k / max(T - 1, 1)
            ph = 0 if f < 0.2 else (1 if 0.4 <= f <= 0.6 else -1)
            if ph < 0:
                continue
            te = r["ts_te"][k]; ok = np.isfinite(te) & (te > 0.03)
            if ok.sum() < 3:
                continue
            I_l.append(r["I"][k]); ip_l.append(r["ip"][k]); tin_l.append(R[ok].min()); tout_l.append(R[ok].max())
            sat_l.append(R[ok].min() <= rmin + 1e-3); ph_l.append(ph); sid_l.append(si); k_l.append(k)
    I = torch.tensor(np.array(I_l), dtype=torch.float32, device=dev); ip = torch.tensor(np.array(ip_l), dtype=torch.float32, device=dev).clamp(min=1e4)
    N = len(ip_l)
    x = torch.linspace(0, 1, 33, device=dev)
    RR = gs.RR.reshape(-1)
    bvflux = 1e-3 * RR ** 2 / 2
    cands = list(itertools.product(BVS, SHARES, SHAPES))
    out = np.full((N, len(cands), 7), np.nan, np.float32)        # is_x, res, R_in, R_out, R_axis, Z_axis, a
    nr = gs.cos.shape[0]
    for i0 in range(0, N, a.batch):
        sl = slice(i0, min(N, i0 + a.batch)); B = sl.stop - sl.start
        psi0 = gs.vacuum(I[sl])
        for c, (bv, sh, shp) in enumerate(cands):
            q = SHAPES[shp](x)[None].expand(B, -1).contiguous()
            P = (sh * (1 - x))[None].expand(B, -1).contiguous()
            pe = psi0 + bv * bvflux[None]
            J, res = gs.solve(pe, ip[sl], P, q, n_iter=40, omega=0.5)
            st = gs.state(pe + gs.plasma(J), J, ip[sl])
            poly, okp = gs.contour(st["phi"], st["R_axis"], st["Z_axis"], st["psi_b"])
            a_ = gs.shape(poly)["a"]
            v = torch.stack([st["is_x"].float(), res, poly[:, nr // 2, 0], poly[:, 0, 0], st["R_axis"], st["Z_axis"], a_], 1)
            out[sl, c] = v.cpu().numpy()
        print(f"  {sl.stop}/{N} frames", flush=True)
    np.savez(a.out, out=out, cands=np.array([f"{b}|{s}|{p}" for b, s, p in cands]), tin=np.array(tin_l), tout=np.array(tout_l),
             sat=np.array(sat_l), phase=np.array(ph_l), sid=np.array(sid_l), k=np.array(k_l))
    tin, tout, ph = np.array(tin_l), np.array(tout_l), np.array(ph_l)
    good = (out[..., 0] > 0.5) & (out[..., 1] < 3e-2) & (out[..., 2] >= 0.20) & (out[..., 6] > 0.25)
    match = good & (np.abs(out[..., 3] - tout[:, None]) < 0.02)
    gen = cands.index((0.0, 0.7, "lin"))
    for p, name in ((0, "ramp-up"), (1, "flat-top")):
        mm = ph == p
        print(f"== {name}: {mm.sum()} frames | generic profile: diverted+converged {good[mm, gen].mean():.1%}, + outer edge within 2 cm {match[mm, gen].mean():.1%} "
              f"| any candidate: {good[mm].any(1).mean():.1%} / {match[mm].any(1).mean():.1%}")
        for b in BVS:
            cb = [i for i, c in enumerate(cands) if c[0] == b]
            print(f"   bv {b:+.0f} mT: any profile matching {match[mm][:, cb].any(1).mean():.1%}")
        for shp in SHAPES:
            cs = [i for i, c in enumerate(cands) if c[2] == shp]
            print(f"   shape {shp:8s}: any share/bv matching {match[mm][:, cs].any(1).mean():.1%}; diverted+converged {good[mm][:, cs].any(1).mean():.1%}")
        for sh in SHARES:
            cs = [i for i, c in enumerate(cands) if c[1] == sh]
            print(f"   share {sh:4.2f}: any shape/bv matching {match[mm][:, cs].any(1).mean():.1%}")
        # outer edge of the diverted candidates vs Thomson
        d = np.where(good[mm], out[mm][..., 3] - tout[mm][:, None], np.nan)
        print(f"   diverted candidates: R_out - TS_out median {np.nanmedian(d) * 100:+.1f} cm; per frame best |dR_out| median {np.nanmedian(np.nanmin(np.abs(d), 1)) * 100:.1f} cm")


if __name__ == "__main__":
    main()
