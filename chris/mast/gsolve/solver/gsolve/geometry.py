"""Machine description: grid, coil elements, wall box, toroidal-field turns.

Each machine is two small files in gsolve/machines/:
  <name>.json        settings (grid, R0, TF turns, wall box, coil margin, input column order, column scales)
  <name>_coils.csv   coil elements (input column, centre R and Z, width, height, two shear angles, turns)
The coil elements are the ones shipped with the Fusion Equilibrium Challenge data
(scripts/build_machine_files.py). Machines available: "mast", "diii-d".
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

MU0 = 4e-7 * math.pi
DATA = Path(__file__).resolve().parent / "machines"


@dataclass(frozen=True, eq=False)
class Machine:
    name: str
    R: np.ndarray                 # (nR,) grid radii (m)
    Z: np.ndarray                 # (nZ,) grid heights (m); maps are (nZ, nR), rows = Z
    R0: float                     # nominal major radius (m), sets profile and field units
    tf_turns: int                 # toroidal-field turns, F_vac = mu0 tf_turns I_tf / (2 pi)
    wall_box: tuple               # (R_min, R_max, Z_min, Z_max): plasma current only flows inside
    coil_margin: float            # cells closer than this to a coil element are not plasma (None = plain box)
    filament_step: float          # filament spacing (m) inside each coil element
    columns: tuple                # input column names, in the order of the coil-current vector
    column_scale: dict            # extra factor per column (MAST: solenoid 0.5 by default, see load_machine)
    elements: dict = field(repr=False)   # arrays: col (index into columns), R, Z, width, height, angle1, angle2, turns
    input_format: str = "{}"
    tf_input: str = ""

    @property
    def shape(self):
        return (len(self.Z), len(self.R))

    def wall_region(self) -> np.ndarray:
        """(nZ, nR) bool: grid cells inside the wall box and away from in-vessel coils."""
        r0, r1, z0, z1 = self.wall_box
        RR, ZZ = np.meshgrid(self.R, self.Z)
        W = (RR >= r0) & (RR <= r1) & (ZZ >= z0) & (ZZ <= z1)
        if self.coil_margin is not None:
            e, mg = self.elements, self.coil_margin
            for rc, zc, w, h in zip(e["R"], e["Z"], e["width"], e["height"]):
                W &= ~((np.abs(RR - rc) < 0.5 * w + mg) & (np.abs(ZZ - zc) < 0.5 * h + mg))
        return W

    def fvac(self, i_tf_kA):
        """Vacuum toroidal-field function F = R B_T (T m) from the TF coil current (kA)."""
        return MU0 * self.tf_turns * np.abs(np.asarray(i_tf_kA, np.float64)) * 1e3 / (2 * math.pi)

    def key(self) -> str:
        """Short text that changes whenever anything affecting the cached matrices changes."""
        e = self.elements
        h = hashlib.sha1(repr((self.wall_box, self.coil_margin, self.filament_step,
                               sorted(self.column_scale.items()))).encode())
        for a in (self.R, self.Z, *(e[k] for k in sorted(e))):
            h.update(np.ascontiguousarray(a, np.float64).tobytes())
        return f"{self.name.lower()}_{h.hexdigest()[:10]}"


def load_machine(name: str = "mast", solenoid_scale: float | None = None, **overrides) -> Machine:
    """Machine from gsolve/machines/<name>.json + <name>_coils.csv. Any field can be overridden, e.g.
    load_machine("mast", coil_margin=None).

    solenoid_scale (MAST only): factor applied to the solenoid column "sol". None keeps the machine file's
    value, 0.5, which rests on documentation alone: the organisers' data notes state that the column reads
    exactly 2x the level-2 pf_active solenoid current and leave the convention open, and Berkery et al. 2021
    (UKAEA-CCFE-PR(21)79) give a MAST solenoid current of about +22 kA where the column reads about +44 kA.
    Use solenoid_scale=1.0 to take the column as stored."""
    name = name.lower()
    cfg = json.loads((DATA / f"{name}.json").read_text())
    if solenoid_scale is not None:
        if "sol" not in cfg["columns"]:
            raise ValueError(f"{name} has no solenoid column 'sol'")
        cfg["column_scale"] = {**cfg["column_scale"], "sol": float(solenoid_scale)}
    cols = tuple(cfg["columns"])
    rows = list(csv.DictReader(open(DATA / f"{name}_coils.csv")))
    el = {k: np.array([float(r[k]) for r in rows]) for k in ("R", "Z", "width", "height", "angle1", "angle2", "turns")}
    el["col"] = np.array([cols.index(r["column"]) for r in rows])
    m = Machine(name=cfg["name"], R=np.linspace(*cfg["grid_R"][:2], int(cfg["grid_R"][2])),
                Z=np.linspace(*cfg["grid_Z"][:2], int(cfg["grid_Z"][2])), R0=cfg["R0"], tf_turns=cfg["tf_turns"],
                wall_box=tuple(cfg["wall_box"]), coil_margin=cfg["coil_margin"], filament_step=cfg["filament_step"],
                columns=cols, column_scale=dict(cfg["column_scale"]), elements=el,
                input_format=cfg["input_format"], tf_input=cfg["tf_input"])
    return replace(m, **overrides) if overrides else m
