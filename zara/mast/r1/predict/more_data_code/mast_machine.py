"""Build a FreeGSNKE machine description for MAST from the challenge-supplied coil geometry.

Sources (all permitted under challenge rule 3 as "MAST machine geometry ... coil descriptions"):
  * Active coils: the 812 conductor elements (coil_R/Z/width/height, coil_input_column) that ship
    with every row of the challenge's mast_public_test split. Nothing else is read from the rows.
  * Limiter/wall: a hand-drawn MAST-like outline built from the same coil positions (centre column
    plus clearance around the in-vessel P2-P6 armour). It is an approximation, NOT a measured
    MAST first-wall contour, and no EFIT/equilibrium data was used to draw it.

Current convention (one FreeGSNKE coil per challenge input column):
  * P2-P6 columns are kA*turn summed over the circuit. Each conductor element carries
    column / n_elements, implemented with multiplier = 1/n_elements so that the FreeGSNKE coil
    current equals the column value in amperes.
  * The solenoid column is the per-turn current in kA on 656 single-turn elements. The challenge
    notes it reads 2x the FAIR-MAST level-2 value; SOL_FACTOR (default 1, i.e. take the column at
    face value) scales the physical current: I_element = SOL_FACTOR * column.
  * TF: MAST has 24 single-turn TF limbs, so fvac = R*B_tor = mu0 * 24 * I_tf / (2 pi).
"""
from __future__ import annotations

import collections
import os
import pickle

import numpy as np
import pyarrow.parquet as pq
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

MU0 = 4e-7 * np.pi
N_TF = 24
SOL_COLUMN = "magnetics_sol_current"
COIL_COLUMNS = [
    "magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current",
    "magnetics_p3u_current", "magnetics_p4l_current", "magnetics_p4u_current",
    "magnetics_p5l_current", "magnetics_p5u_current", "magnetics_p6l_current",
    "magnetics_p6u_current", SOL_COLUMN,
]
GEOM_COLUMNS = ["coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width",
                "coil_height", "coil_angle1", "coil_angle2"]

# MAST-like limiter (upper half, R, Z in m), mirrored to the lower half. Drawn to stay
# 3-5 cm clear of the coil armour listed in the challenge geometry:
#   centre column  R = 0.20 (innermost Thomson chord is at R = 0.235)
#   P2 armour face Z = 1.60 (P2 elements start at Z = 1.654)
#   P3 at R 1.07-1.14, Z >= 1.07; P4 at R >= 1.43, Z >= 1.04; P6 at R >= 1.43, Z 0.86-0.93;
#   P5 at R >= 1.585, |Z| 0.44-0.56.
_LIMITER_UPPER = [
    (0.200, 0.000), (0.200, 1.600), (0.600, 1.600), (1.020, 1.110), (1.020, 1.030),
    (1.180, 1.030), (1.380, 0.990), (1.380, 0.810), (1.540, 0.600), (1.540, 0.000),
]


def limiter_polygon() -> np.ndarray:
    up = np.array(_LIMITER_UPPER)
    lo = up[::-1].copy()
    lo[:, 1] *= -1
    pts = np.vstack([up, lo[1:]])  # closed loop from inner midplane, counter-clockwise
    return pts


def read_geometry(parquet_path: str) -> dict:
    row = pq.read_table(parquet_path, columns=GEOM_COLUMNS).to_pylist()[0]
    return {k: np.asarray(v) for k, v in row.items()}


def active_coils_from_geometry(geom: dict) -> dict:
    """One FreeGSNKE coil per input column; sub-coils per conductor group (e.g. p2 inner/outer)."""
    names = geom["coil_name"]
    cols = geom["coil_input_column"]
    coils: dict = {}
    for col in COIL_COLUMNS:
        m = cols == col
        n_el = int(m.sum())
        groups = collections.OrderedDict()
        for i in np.where(m)[0]:
            groups.setdefault(names[i].rsplit("_", 1)[0], []).append(i)
        mult = 1.0 if col == SOL_COLUMN else 1.0 / n_el
        sub = {}
        for g, idx in groups.items():
            idx = np.array(idx)
            sub[g] = {
                "R": geom["coil_R"][idx].astype(float).tolist(),
                "Z": geom["coil_Z"][idx].astype(float).tolist(),
                "dR": float(np.mean(geom["coil_width"][idx])),
                "dZ": float(np.mean(geom["coil_height"][idx])),
                "resistivity": 1.7e-8,
                "polarity": 1.0,
                "multiplier": mult,
            }
        key = col.replace("magnetics_", "").replace("_current", "")
        coils[key] = sub[next(iter(sub))] if len(sub) == 1 else sub
    return coils


def coil_currents_A(frame_kA: dict, sol_factor: float = 1.0) -> dict:
    """Map challenge input values (kA or kA*turn) to FreeGSNKE coil currents (A)."""
    out = {}
    for col in COIL_COLUMNS:
        key = col.replace("magnetics_", "").replace("_current", "")
        scale = sol_factor if col == SOL_COLUMN else 1.0
        out[key] = float(frame_kA[col]) * 1e3 * scale
    return out


def fvac_from_tf(tf_kA: float) -> float:
    return MU0 * N_TF * abs(tf_kA) * 1e3 / (2 * np.pi)


def write_machine(parquet_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    geom = read_geometry(parquet_path)
    coils = active_coils_from_geometry(geom)
    lim = [{"R": float(r), "Z": float(z)} for r, z in limiter_polygon()]
    paths = {
        "active_coils_path": os.path.join(out_dir, "MAST_challenge_active_coils.pickle"),
        "limiter_path": os.path.join(out_dir, "MAST_like_limiter.pickle"),
        "wall_path": os.path.join(out_dir, "MAST_like_wall.pickle"),
    }
    pickle.dump(coils, open(paths["active_coils_path"], "wb"))
    pickle.dump(lim, open(paths["limiter_path"], "wb"))
    pickle.dump(lim, open(paths["wall_path"], "wb"))
    return paths


def build_tokamak(machine_dir: str):
    from freegsnke import build_machine
    return build_machine.tokamak(
        active_coils_path=os.path.join(machine_dir, "MAST_challenge_active_coils.pickle"),
        limiter_path=os.path.join(machine_dir, "MAST_like_limiter.pickle"),
        wall_path=os.path.join(machine_dir, "MAST_like_wall.pickle"),
    )


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{R1_ROOT}/fusion-equilibrium-challenge-starter/"
                    "challenge_data/data/mast_public_test")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "machines",
                                                  "MAST_challenge"))
    a = ap.parse_args()
    src = sorted(glob.glob(os.path.join(a.data, "*.parquet")))[0]
    print(write_machine(src, a.out))
