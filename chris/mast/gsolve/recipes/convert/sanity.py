#!/usr/bin/env python3
"""Checks of a converted gsolve set against Zara's gen4_set5000v4 training labels.

    python sanity.py /fsx/chris.beckham/gsolve_data/pilot/gen4_layout [--max-files N]
Writes label_compare.json, psi_maps.png and label_hists.png into the set folder.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

ZARA = Path("/fsx/zara.ercan/more_data/m2/gen4_set5000v4/mast_g4s5000v4_train")
LAB = [("q95", "efit_q95"), ("betaN", "efit_beta_n"), ("li", "efit_li"), ("kappa", "efit_elongation"),
       ("R_axis", "efit_r_axis"), ("Z_axis", "efit_z_axis")]
ZMETA = {"q95": "q95", "betaN": "beta_n", "li": "li", "kappa": "elongation", "R_axis": "r_axis", "Z_axis": "z_axis"}


def axis_from_map(psi, R, Z):
    """Independent axis: maximum of a cubic spline of the stored map near the grid maximum (inner region)."""
    m = (R[None, :] > 0.3) & (R[None, :] < 1.4) & (np.abs(Z[:, None]) < 1.0)
    iz, ir = np.unravel_index(np.argmax(np.where(m, psi, -np.inf)), psi.shape)
    spl = RectBivariateSpline(Z, R, psi)
    zz = np.linspace(Z[iz] - 0.08, Z[iz] + 0.08, 161); rr = np.linspace(R[ir] - 0.08, R[ir] + 0.08, 161)
    v = spl(zz, rr)
    a, b = np.unravel_index(np.argmax(v), v.shape)
    return rr[b], zz[a], v[a, b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    files = sorted(a.folder.glob("*/mast_synth*.parquet"))
    lab = {k: [] for k, _ in LAB}
    ax_err, psi_ax_err, peaks, maps = [], [], [], []
    for f in files:
        r = pq.read_table(f).to_pylist()[0]
        R, Z = np.asarray(r["efit_grid_R"]), np.asarray(r["efit_grid_Z"])
        P = np.asarray(r["efit_psirz"], float)
        for k, c in LAB:
            lab[k] += list(r[c])
        for q in range(len(P)):
            ra, za, pa_ = axis_from_map(P[q], R, Z)
            ax_err.append(np.hypot(ra - r["efit_r_axis"][q], za - r["efit_z_axis"][q]))
            psi_ax_err.append(abs(pa_ - r["efit_psi_axis"][q]) / abs(r["efit_psi_axis"][q] - r["efit_psi_boundary"][q]))
        te = np.asarray(r["thomson_core_Te"], float)
        if te.ndim == 2:
            te = te[:, 1:]
            ok = np.isfinite(te).any(1)
            peaks += list(np.nanmax(te[ok], 1))
        if len(maps) < 3 and "_r0_" in f.name:
            maps.append((f.name, P[len(P) // 2], R, Z, r["efit_r_axis"][len(P) // 2], r["efit_z_axis"][len(P) // 2],
                         r["efit_lcfs_r"][len(P) // 2], r["efit_lcfs_z"][len(P) // 2]))
    zm = pq.read_table(ZARA / "frames_metadata.parquet").to_pandas()
    rows = {}
    for k, _ in LAB:
        g = np.asarray(lab[k], float); zz = zm[ZMETA[k]].to_numpy(float)
        rows[k] = {"gsolve_5_50_95": np.nanpercentile(g, [5, 50, 95]).round(3).tolist(),
                   "zara_v4_5_50_95": np.nanpercentile(zz, [5, 50, 95]).round(3).tolist()}
    ax_err, psi_ax_err = np.array(ax_err), np.array(psi_ax_err)
    out = {"files": len(files), "frames": int(len(lab["q95"])), "labels": rows,
           "axis_from_map_vs_label_m": {"median": float(np.median(ax_err)), "p99": float(np.percentile(ax_err, 99)),
                                        "max": float(ax_err.max())},
           "psi_axis_rel_err": {"median": float(np.median(psi_ax_err)), "max": float(psi_ax_err.max())},
           "thomson_core_Te_peak_eV_5_50_95": np.percentile(peaks, [5, 50, 95]).round(0).tolist(),
           "zara_note": "Zara v4 train README: Te peak 318/676/1280 eV (real MAST 367/717/1290)"}
    (a.folder / "label_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    # psi maps: three gsolve frames next to three Zara frames
    zf = sorted(ZARA.glob("mast_synth_g4_*.parquet"))[:3]
    fig, axs = plt.subplots(2, 3, figsize=(10, 9))
    for j, (name, P, R, Z, ra, za, lr, lz) in enumerate(maps):
        ax = axs[0, j]; ax.contour(R, Z, P, 30); ax.plot(ra, za, "r+", ms=12); ax.plot(lr, lz, "k-", lw=1)
        ax.set_title(f"gsolve {name[14:30]}\nmax {P.max():.3f} at axis", fontsize=8); ax.set_aspect("equal")
    for j, f in enumerate(zf):
        r = pq.read_table(f).to_pylist()[0]
        P = np.asarray(r["efit_psirz"][0]); R, Z = np.asarray(r["efit_grid_R"]), np.asarray(r["efit_grid_Z"])
        n = int(r["efit_lcfs_n"][0])
        ax = axs[1, j]; ax.contour(R, Z, P, 30); ax.plot(r["efit_r_axis"][0], r["efit_z_axis"][0], "r+", ms=12)
        ax.plot(r["efit_lcfs_r"][0][:n], r["efit_lcfs_z"][0][:n], "k-", lw=1)
        ax.set_title(f"Zara v4 {f.stem[-5:]}\nmax {P.max():.3f}", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(a.folder / "psi_maps.png", dpi=90)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, _) in zip(axs.ravel(), LAB):
        g = np.asarray(lab[k], float); zz = zm[ZMETA[k]].to_numpy(float)
        lo, hi = np.nanpercentile(np.r_[g, zz], [0.5, 99.5])
        b = np.linspace(lo, hi, 50)
        ax.hist(zz, b, density=True, alpha=0.5, label="Zara v4 train"); ax.hist(g, b, density=True, alpha=0.5, label="gsolve")
        ax.set_title(k)
    axs[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(a.folder / "label_hists.png", dpi=90)


if __name__ == "__main__":
    main()
