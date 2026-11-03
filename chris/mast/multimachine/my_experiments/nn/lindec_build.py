"""Green's matrices for the linear decoder (train.py --decoder linear), and the DIII-D ceiling check.

usage: python lindec_build.py                 # write <cache>/lindec_G_{d3d,mast}.npz
       FEC_MAST_MASK=limiter python lindec_build.py --machine mast   # write <cache>/lindec_G_mast_limiter.npz
       python lindec_build.py --verify [N]    # also rebuild ~N DIII-D val maps from their true current (labels)

G[p, c] is the flux at grid point p from 1 A in envelope cell c (gs_solver.gs.Grid: filament Green's function, self
term at the cell's geometric mean distance), in Wb/rad per A. With J in the labels' sign convention,
J = −Δ*ψ / (μ0 R) · ΔR ΔZ per cell (it sums to −Ip on DIII-D), the plasma's flux in the labels' convention is
G · J with this G unchanged: G is the filament ("physical") flux per amp, labels are S_PIC = −1 times physical flux,
and J_label = −J_physical, so the two signs cancel. --verify checks this on real maps.

Each file holds G (4225, N_env) float32, idx (N_env,) the flat grid index of each envelope cell (rows Z, columns R),
the envelope mask, the grid and the cell size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "gs_solver"))
import numpy as np  # noqa: E402

from common import CACHE  # noqa: E402

MU0 = 4e-7 * np.pi


def g_path(machine: str, cache: Path = CACHE, mast_mask: str | None = None) -> Path:
    """MAST's file carries the region's suffix (mast_region.py): lindec_G_mast.npz is the envelope's."""
    import mast_region
    sfx = mast_region.suffix(mast_mask) if machine == "mast" else ""
    return cache / f"lindec_G_{machine}{sfx}.npz"


def build(machine: str, cache: Path = CACHE, mast_mask: str | None = None) -> Path:
    from gs import Grid
    g = Grid(machine, mast_mask=mast_mask)
    out = g_path(machine, cache, mast_mask)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, G=g.G.astype(np.float32), idx=g.idx.astype(np.int64), mask=g.mask, r=g.r, z=g.z,
             dr=np.float64(g.dr), dz=np.float64(g.dz))
    tmp.replace(out)
    print(f"{machine}: G {g.G.shape} -> {out}", flush=True)
    return out


def load_g(machine: str, cache: Path = CACHE, mast_mask: str | None = None) -> dict:
    """The saved matrix for `machine`, built first if the file is missing. MAST: the region `mast_mask`
    (default the FEC_MAST_MASK selection)."""
    path = g_path(machine, cache, mast_mask)
    if not path.exists():
        build(machine, cache, mast_mask)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def true_current(psi: np.ndarray, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(…, 65, 65) label ψ -> current per cell in amps, label convention (sums to −Ip). 0 on the grid border."""
    psi = np.asarray(psi, np.float64)
    dr, dz = r[1] - r[0], z[1] - z[0]
    c = psi[..., 1:-1, 1:-1]
    p_r = (psi[..., 1:-1, 2:] - psi[..., 1:-1, :-2]) / (2 * dr)
    p_rr = (psi[..., 1:-1, 2:] - 2 * c + psi[..., 1:-1, :-2]) / dr ** 2
    p_zz = (psi[..., 2:, 1:-1] - 2 * c + psi[..., :-2, 1:-1]) / dz ** 2
    R = r[1:-1][None, :]
    out = np.zeros_like(psi)
    out[..., 1:-1, 1:-1] = -(p_rr - p_r / R + p_zz) / (MU0 * R) * dr * dz
    return out


def verify(n_frames: int, cache: Path = CACHE, ceiling_sample: bool = False) -> None:
    """Rebuild DIII-D val maps as coil picture + G · J_true, with J_true over the whole envelope, and score them."""
    import pandas as pd
    from common import IP_COL, external_flux, load_coil_pictures
    sys.modules.pop("common")
    sys.path.insert(0, str(REPO / "fusion_scoring"))
    from common import AXIS_SIGN, CONS_SCALARS  # noqa: E402
    from derive import derive_frame  # noqa: E402
    from lcfs import extract_lcfs_with_sign  # noqa: E402
    S = AXIS_SIGN["DIII-D"]
    gd = load_g("d3d", cache)
    G, idx, r, z, mask = gd["G"].astype(np.float64), gd["idx"], gd["r"], gd["z"], gd["mask"]

    def scalars(psi, want_contour=False):
        p = psi.reshape(65, 65)
        ex, _ = extract_lcfs_with_sign(p, r, z, "DIII-D", mask, mask.astype(float), axis_sign=S)
        if ex is None:
            return (None, None) if want_contour else None
        d = derive_frame(p, r, z, "DIII-D", mask, mask.astype(float), contour=ex, with_li=True, axis_sign=S)
        out = [d[k] for k in CONS_SCALARS]
        return (out, ex) if want_contour else out

    def r2(y, p):
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    sh = pd.read_parquet(cache / "shots.parquet")
    ps = np.load(cache / "psi.npy", mmap_mode="r"); co = np.load(cache / "coils.npy", mmap_mode="r")
    rows = []
    if ceiling_sample:
        # The frames of mast_diag/linear_decoder_ceiling.py (defaults: 8 val shots, every 8th frame).
        for row in sh[sh.split == "val"].sample(8, random_state=11).itertuples():
            rows += [k for k in range(int(row.start), int(row.start + row.n), 8) if abs(float(co[k, IP_COL])) >= 300]
        n_frames = len(rows)
    val = sh[sh.split == "val"].sample(25, random_state=11) if not ceiling_sample else sh.iloc[:0]
    for row in val.itertuples():
        step = max(1, int(row.n) // max(1, n_frames // 25))
        rows += [k for k in range(int(row.start), int(row.start + row.n), step)
                 if abs(float(co[k, IP_COL])) >= 300][: n_frames // 25 + 1]
    rows = np.array(rows[:n_frames])
    coils = np.asarray(co[rows], np.float32); ip = coils[:, IP_COL].astype(np.float64) * 1000.0
    true = np.asarray(ps[rows], np.float64)
    J = true_current(true.reshape(-1, 65, 65), r, z).reshape(len(rows), -1)[:, idx]
    print(f"{len(rows)} DIII-D val frames; Σ J / Ip median {np.median(J.sum(1) / ip):+.4f} "
          f"(5–95% {np.percentile(J.sum(1) / ip, 5):+.4f} … {np.percentile(J.sum(1) / ip, 95):+.4f})", flush=True)
    from matplotlib.path import Path as MPath
    from scipy import ndimage
    pts = np.column_stack([np.meshgrid(r, z, indexing="xy")[0].ravel(), np.meshgrid(r, z, indexing="xy")[1].ravel()])
    ct, near = [], np.zeros_like(J, bool)
    for k in range(len(rows)):
        s, ex = scalars(true[k], want_contour=True)
        ct.append(s if s is not None else [np.nan] * len(CONS_SCALARS))
        if ex is not None:
            inside = MPath(ex).contains_points(pts).reshape(65, 65)
            near[k] = ndimage.binary_dilation(inside, np.ones((3, 3), bool)).ravel()[idx]
        else:
            near[k] = True
    ct = np.array(ct, float)
    # "plasma": J kept only inside the true boundary dilated by one cell, as in mast_diag/linear_decoder_ceiling.py.
    variants = (("envelope", "baseline_pictures_fitgain2.npz", J, 1.0),
                ("envelope", "baseline_pictures_fitgain2.npz", J, -1.0),
                ("plasma", "baseline_pictures_fitgain2.npz", np.where(near, J, 0.0), 1.0),
                ("envelope", "baseline_pictures_ecoil135.npz", J, 1.0),
                ("plasma", "baseline_pictures_ecoil135.npz", np.where(near, J, 0.0), 1.0))
    for region, name, Jv, flip in variants:
        pics = load_coil_pictures(HERE / name)
        pic = external_flux(coils, pics["green"], pics["ecoil"]).astype(np.float64)
        label = f"{region:8s} {'+G' if flip > 0 else '−G'}"
        if True:
            P = pic + flip * Jv @ G.T
            raw = 1 - np.sum((true - P) ** 2) / np.sum((true - true.mean()) ** 2)
            Po = P - P.mean(1, keepdims=True) + true.mean(1, keepdims=True)
            off = 1 - np.sum((true - Po) ** 2) / np.sum((true - true.mean()) ** 2)
            if flip < 0:
                print(f"  {name:32s} {label}  R²ψ {raw:.4f}  offset removed {off:.4f}  (wrong sign)", flush=True)
                continue
            cp = np.array([scalars(P[k]) or [np.nan] * len(CONS_SCALARS) for k in range(len(rows))], float)
            each = [r2(ct[:, j], cp[:, j]) for j in range(len(CONS_SCALARS))]
            cons = np.mean([max(0.0, v) for v in each if np.isfinite(v)])
            print(f"  {name:32s} {label}  R²ψ {raw:.4f}  offset removed {off:.4f}  "
                  + " ".join(f"{c} {v:.3f}" for c, v in zip(CONS_SCALARS, each)) + f"  Consistency {cons:.3f}",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=int, nargs="?", const=200, default=0)
    ap.add_argument("--ceiling-sample", action="store_true",
                    help="--verify on the frames mast_diag/linear_decoder_ceiling.py uses instead of 25 shots")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--machine", choices=["d3d", "mast"], action="append",
                    help="build only this machine's matrix (repeatable); default both")
    a = ap.parse_args()
    for m in (a.machine or ("d3d", "mast")):
        build(m, a.cache)
    if a.verify:
        verify(a.verify, a.cache, a.ceiling_sample)
