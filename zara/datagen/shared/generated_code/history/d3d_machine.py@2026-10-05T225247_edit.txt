"""FreeGSNKE machine description for DIII-D from the challenge-supplied coil geometry.

Sources (DIII-D data carries no rule-3 restriction; DIII-D labels are the challenge's training
data):
  * Active coils: the 19 lumped rectangles in every diii_d_train row (coil_R/Z/width/height and
    the parallelogram skew coil_angle1/coil_angle2, joined to the current columns by
    coil_input_column). Each is filled with a grid of filaments following its parallelogram.
  * E-coil turns: magnetics_ECOILA is a per-turn current (kA) with no turn count in the data.
    E_TURNS = 152.8 is the effective turn count the C1 physics stage fitted against DIII-D EFIT
    flux (theta[0] = a_E in cache/nn/ridge_full.npz, also A_E_NOMINAL in physics_features.py).
  * Limiter/wall: the outline of the challenge scorer's DIII-D plasma-envelope mask
    (fusion_scoring/masks/d3d_envelope.npz), R 0.97-2.38 m, Z -1.45-1.35 m.

Current convention: F-coil columns are kA*turn, spread evenly over the coil's filaments
(multiplier 1/n_filaments, so the FreeGSNKE coil current equals the column in amperes). The
E-coil carries E_TURNS * column. TF: the B-coil has 144 turns, R*B_tor = mu0*144*I_b/(2 pi).
Parallelogram convention (as in EFIT/OMFIT): angle1 tilts the horizontal sides,
Z += u*w*tan(angle1); angle2 tilts the vertical sides, R += v*h/tan(angle2) (90 deg = upright).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pyarrow.parquet as pq

MU0 = 4e-7 * np.pi
E_TURNS = 152.8
N_BCOIL = 144
E_COLUMN = "magnetics_ECOILA"
F_COLUMNS = [f"magnetics_F{i}{s}" for s in "AB" for i in range(1, 10)]
COIL_COLUMNS = F_COLUMNS + [E_COLUMN]
GEOM_COLUMNS = ["coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width",
                "coil_height", "coil_angle1", "coil_angle2"]
ENVELOPE = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/fusion_scoring/masks/d3d_envelope.npz"
FIL_SPACING = 0.02   # target filament spacing (m)
MAX_FIL_SIDE = 60


def read_geometry(parquet_path: str) -> dict:
    row = pq.read_table(parquet_path, columns=GEOM_COLUMNS).to_pylist()[0]
    return {k: np.asarray(v) for k, v in row.items()}


def filaments(R, Z, w, h, a1, a2):
    """Filament centres filling the coil's parallelogram, and the filament cell size."""
    nu = int(np.clip(round(w / FIL_SPACING), 2, MAX_FIL_SIDE))
    nv = int(np.clip(round(h / FIL_SPACING), 2, MAX_FIL_SIDE))
    u = (np.arange(nu) + 0.5) / nu - 0.5
    v = (np.arange(nv) + 0.5) / nv - 0.5
    uu, vv = np.meshgrid(u, v, indexing="ij")
    r = R + uu * w
    z = Z + vv * h
    if a1 != 0:
        z = z + uu * w * np.tan(np.deg2rad(a1))
    if a2 != 0:
        r = r + vv * h / np.tan(np.deg2rad(a2))
    return r.ravel(), z.ravel(), w / nu, h / nv


def active_coils_from_geometry(geom: dict) -> dict:
    coils = {}
    for i, col in enumerate(geom["coil_input_column"]):
        r, z, dr, dz = filaments(*(float(geom[k][i]) for k in
                                   ["coil_R", "coil_Z", "coil_width", "coil_height",
                                    "coil_angle1", "coil_angle2"]))
        turns = E_TURNS if col == E_COLUMN else 1.0
        coils[col.replace("magnetics_", "")] = {
            "R": r.tolist(), "Z": z.tolist(), "dR": float(dr), "dZ": float(dz),
            "resistivity": 1.7e-8, "polarity": 1.0, "multiplier": turns / len(r),
        }
    return coils


def limiter_polygon() -> np.ndarray:
    """Outer outline of the scorer's DIII-D envelope mask (closed, counter-clockwise)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.load(ENVELOPE)
    fig = plt.figure()
    cs = plt.contour(m["grid_R"], m["grid_Z"], m["mask_coarse"].astype(float), [0.5])
    paths = cs.allsegs[0]
    plt.close(fig)
    poly = max(paths, key=len)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[:1]])
    area = 0.5 * np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1])
    return poly if area > 0 else poly[::-1]


def coil_currents_A(frame_kA: dict) -> dict:
    """Challenge input values (kA*turn for F-coils, kA per turn for the E-coil) -> FreeGSNKE (A)."""
    return {c.replace("magnetics_", ""): float(frame_kA[c]) * 1e3 for c in COIL_COLUMNS}


def fvac_from_bcoil(b_kA: float) -> float:
    return MU0 * N_BCOIL * abs(b_kA) * 1e3 / (2 * np.pi)


def write_machine(parquet_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    coils = active_coils_from_geometry(read_geometry(parquet_path))
    lim = [{"R": float(r), "Z": float(z)} for r, z in limiter_polygon()]
    paths = {"active_coils_path": os.path.join(out_dir, "DIIID_challenge_active_coils.pickle"),
             "limiter_path": os.path.join(out_dir, "DIIID_envelope_limiter.pickle"),
             "wall_path": os.path.join(out_dir, "DIIID_envelope_wall.pickle")}
    pickle.dump(coils, open(paths["active_coils_path"], "wb"))
    pickle.dump(lim, open(paths["limiter_path"], "wb"))
    pickle.dump(lim, open(paths["wall_path"], "wb"))
    return paths


def build_tokamak(machine_dir: str):
    from freegsnke import build_machine
    return build_machine.tokamak(
        active_coils_path=os.path.join(machine_dir, "DIIID_challenge_active_coils.pickle"),
        limiter_path=os.path.join(machine_dir, "DIIID_envelope_limiter.pickle"),
        wall_path=os.path.join(machine_dir, "DIIID_envelope_wall.pickle"),
    )


if __name__ == "__main__":
    import glob
    data = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/challenge_data/data/diii_d_train"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "machines", "DIIID_challenge")
    print(write_machine(sorted(glob.glob(os.path.join(data, "*.parquet")))[0], out))
