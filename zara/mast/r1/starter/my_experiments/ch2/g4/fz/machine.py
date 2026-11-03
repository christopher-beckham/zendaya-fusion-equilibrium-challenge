"""upegs copy of pegs/machine.py (2026-10-05 23:10 UTC) + synthetic test-set machines (SPARC, ITER, synthetic DIII-D).

Machine descriptions for the cross-geometry (xg) model: any tokamak, described by data only.

A tokamak enters the model as a MachineSpec: the computational grid, its PF coil elements (rectangles or
parallelograms) with the current column that feeds each element and the amperes per element per unit of that
column, the toroidal-field turns, an optional wall polygon, and the flux convention of its stored maps.
Nothing else about a machine is known to the model. From a spec, exact physics gives

    vacuum flux     psi_vac = I_columns @ M          (filament Green's functions summed over coil elements)
    plasma flux     psi_pl  = I_cells  @ G_cells     (Green's function of a uniformly filled grid cell)
    domain          grid cells inside the wall polygon, or, without one, inside the convex hull of the coil
                    elements; always keeping a clearance `gap` from every coil element and the central solenoid.
                    Plasma current may flow only here, and its rim is the first wall for the LCFS search.

Adding a machine = writing a spec (see spec_from_coil_table: the challenge parquet coil table is one such source).
Per-dataset unit conventions and documented data corrections are part of the spec, never of the model.
Flux convention inside the model: physical, Ip > 0, magnetic axis = maximum. `to_stored` maps to a dataset's.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import label
from scipy.spatial import Delaunay
R1_ROOT = os.environ.get("R1_ROOT", "/fsx/zara.ercan")   # Zara's data folder (zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.append(str(HERE.parents[2] / "c1"))
from greens import coil_filaments, filament_psi  # noqa: E402

MU0 = 4e-7 * np.pi
XG_CACHE = Path(f"{R1_ROOT}/cache/pegs")   # machine matrices shared with pegs (same specs -> same keys)
GEN = Path(f"{R1_ROOT}/more_data/generated")
# synthetic FreeGSNKE TEST sets (never trained on): model name -> generator key
SYNTH_SETS = {"SPARCS": "sparc", "ITERS": "iter", "D3DS": "d3d"}
SYNTHSET_CACHE = Path(f"{R1_ROOT}/cache/synthsets")        # xg/synthset_to_cache.py records (SPARCS, ITERS); D3DS: ours
# synthetic TRAINING machines (FreeGSNKE on public OMAS geometry, converted by the pegs session; MAST/SPARC/ITER never here)
MULTI_ROOT = Path(f"{R1_ROOT}/cache/pegs/multi")
MULTI_MACHINES = ("JET", "EAST", "CMOD", "KSTAR", "NSTXU_FWD", "NSTX_FWD", "MASTSYN", "MASTVAL", "MASTSYN10", "MASTVAL10", "MASTSEQ",
                  "MASTG4", "MASTG4V3", "MASTG4H", "MASTG4RT", "MASTG4V3RT")
# m2: synthetic MAST training (MASTSYN) / validation (MASTVAL) sets on the exact MAST machine (solenoid x0.5), records in cache/m2
M2_MACHINES = ("MASTSYN", "MASTVAL", "MASTSYN10", "MASTVAL10", "MASTSEQ", "MASTG4", "MASTG4V3", "MASTG4H", "MASTG4RT", "MASTG4V3RT")
# ch2/g4: gen4 case sets (recipe v2: 20k + 5k sets -> MASTG4; recipe v3 -> MASTG4V3; held-out cases of all three -> MASTG4H)
G4_MACHINES = ("MASTG4", "MASTG4V3", "MASTG4H", "MASTG4RT", "MASTG4V3RT")   # RT = real Thomson input (ch2/g4/datarec)
G4_ROOT = Path(f"{R1_ROOT}/cache/ch2/g4/multi")
M2_ROOT = Path(f"{R1_ROOT}/cache/m2/multi")
M2_SOL_FACTOR = 0.5
M2_SOL = {"MASTSYN": 0.5, "MASTVAL": 0.5, "MASTSYN10": 1.0, "MASTVAL10": 1.0, "MASTSEQ": 0.5, "MASTG4": 0.5, "MASTG4V3": 0.5, "MASTG4H": 0.5, "MASTG4RT": 0.5, "MASTG4V3RT": 0.5}   # solenoid convention of each synthetic set


def multi_root(name: str) -> Path:
    if name in G4_MACHINES:
        return G4_ROOT
    return M2_ROOT if name in M2_MACHINES else MULTI_ROOT
D3D_CACHE = {"train": Path(f"{R1_ROOT}/cache/nn/diii_d_train"),
             "test": Path(f"{R1_ROOT}/cache/nn/diii_d_public_test")}
D3D_PX = {"train": Path(f"{R1_ROOT}/cache/nn/px/diii_d_train"),
          "test": Path(f"{R1_ROOT}/cache/nn/px/diii_d_public_test")}
MAST_CACHE = {"test": Path(f"{R1_ROOT}/cache/mast/mast_public_test"),
              # 500 synthetic FreeGSNKE MAST equilibria (xg/synth_to_cache.py --fix-thomson, frozen copy 2026-10-05 21:50)
              "synth": Path(f"{R1_ROOT}/cache/pegs/src_mast_synth")}


@dataclass
class MachineSpec:
    name: str
    grid_R: np.ndarray                      # (nR,) m, uniform
    grid_Z: np.ndarray                      # (nZ,) m, uniform
    columns: list                           # current column names, in the order of the shot arrays
    elements: np.ndarray                    # (E, 6) R, Z, W, H, angle1, angle2 (deg; EFIT parallelogram convention)
    el_col: np.ndarray                      # (E,) column index feeding each element
    el_scale: np.ndarray                    # (E,) amperes in the element per unit of its column's value
    tf_turns: float                         # B_phi R = mu0 tf_turns I_tf / (2 pi)
    wall: np.ndarray | None = None          # (W, 2) first-wall polygon (optional)
    solenoid_cols: tuple = ()               # columns whose elements form the central solenoid
    col_uniform: dict = field(default_factory=dict)   # column -> uniform flux per unit value (far field of a long
                                                      # solenoid whose elements are not all in the coil table)
    stored_sign: float = 1.0                # stored psi = stored_sign * psi_phys + stored_offset
    stored_offset: float = 0.0
    n_sub: int = 4                          # sub-filaments per element side
    notes: str = ""

    def key(self) -> str:
        h = hashlib.sha1()
        for a in (self.grid_R, self.grid_Z, self.elements, self.el_col, self.el_scale):
            h.update(np.ascontiguousarray(a, np.float64).tobytes())
        h.update(json.dumps([self.name, list(self.columns), self.n_sub, None if self.wall is None else np.round(self.wall, 6).tolist(),
                             sorted(self.col_uniform.items())]).encode())
        return h.hexdigest()[:12]


@dataclass
class Machine:
    spec: MachineSpec
    M: np.ndarray                           # (C, P) flux per unit of each column value
    domain: np.ndarray                      # (nZ, nR) bool
    cell_iz: np.ndarray
    cell_ir: np.ndarray
    G_cells: np.ndarray                     # (K, P) float32

    name = property(lambda self: self.spec.name)
    grid_R = property(lambda self: self.spec.grid_R)
    grid_Z = property(lambda self: self.spec.grid_Z)
    tf_turns = property(lambda self: self.spec.tf_turns)
    dR = property(lambda self: float(self.spec.grid_R[1] - self.spec.grid_R[0]))
    dZ = property(lambda self: float(self.spec.grid_Z[1] - self.spec.grid_Z[0]))

    def to_stored(self, psi_phys):
        return self.spec.stored_sign * psi_phys + self.spec.stored_offset

    def from_stored(self, psi_stored):
        return (psi_stored - self.spec.stored_offset) / self.spec.stored_sign


# ----------------------------------------------------------------------------- physics from a spec
def vacuum_matrix(spec: MachineSpec) -> np.ndarray:
    RR, ZZ = np.meshgrid(spec.grid_R, spec.grid_Z)
    M = np.zeros((len(spec.columns), RR.size))
    for e, (R, Z, W, H, a1, a2) in enumerate(spec.elements):
        if spec.el_scale[e] == 0:
            continue
        fr, fz = coil_filaments(R, Z, W, H, a1, a2, spec.n_sub)
        g = filament_psi(fr[:, None, None], fz[:, None, None], RR[None], ZZ[None]).mean(0).ravel()
        M[spec.el_col[e]] += spec.el_scale[e] * g
    for c, u in spec.col_uniform.items():
        M[spec.columns.index(c)] += u
    return M


def coil_domain(spec: MachineSpec, gap=0.03) -> np.ndarray:
    gR, gZ = spec.grid_R, spec.grid_Z
    RR, ZZ = np.meshgrid(gR, gZ)
    pts = np.stack([RR.ravel(), ZZ.ravel()], 1)
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    el = spec.elements[spec.el_scale != 0] if (spec.el_scale != 0).any() else spec.elements
    if spec.wall is not None:
        inside = MPath(spec.wall).contains_points(pts)
    else:
        corners = np.concatenate([el[:, :2] + np.array([sx * 0.5, sz * 0.5]) * el[:, 2:4] for sx in (-1, 1) for sz in (-1, 1)])
        inside = Delaunay(corners).find_simplex(pts) >= 0
    far = np.ones(len(pts), bool)
    for R, Z, W, H, *_ in el:
        sep_r = np.abs(pts[:, 0] - R) - 0.5 * (W + dR)
        sep_z = np.abs(pts[:, 1] - Z) - 0.5 * (H + dZ)
        far &= np.maximum(sep_r, sep_z) > gap
    sol = np.isin(spec.el_col, [spec.columns.index(c) for c in spec.solenoid_cols])
    if sol.any():
        far &= pts[:, 0] - 0.5 * dR > float((spec.elements[sol, 0] + 0.5 * spec.elements[sol, 2]).max()) + gap
    dom = (inside & far).reshape(RR.shape)
    dom[0, :] = dom[-1, :] = dom[:, 0] = dom[:, -1] = False
    lab, n = label(dom)
    if n > 1:
        dom = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    return dom


def cell_greens_full(grid_R, grid_Z, n_sub=4):
    """Flux at every node from unit current spread uniformly over every grid cell: (P_src, P_obs) float32.
    Translation invariance in Z: the value depends on (R_src, R_obs, Z_obs - Z_src) only. Sub-filaments sit at
    (k + 1/2)/n_sub offsets, so none coincides with the cell's own node (finite self-flux)."""
    nR, nZ = len(grid_R), len(grid_Z)
    dR, dZ = grid_R[1] - grid_R[0], grid_Z[1] - grid_Z[0]
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    lags = np.arange(-(nZ - 1), nZ) * dZ
    T = np.zeros((nR, nR, len(lags)))
    for su in u:
        for sv in u:
            T += filament_psi((grid_R + su * dR)[:, None, None], sv * dZ, grid_R[None, :, None], lags[None, None, :])
    T /= n_sub ** 2
    iz, ir = np.divmod(np.arange(nR * nZ), nR)
    return T[ir[:, None], ir[None, :], (iz[None, :] - iz[:, None]) + (nZ - 1)].astype(np.float32)


def build(spec: MachineSpec, verbose=True) -> Machine:
    key = spec.key()
    path = XG_CACHE / "machines" / f"{spec.name}_{key}.npz"
    if path.exists():
        d = np.load(path)
        M, dom = d["M"], d["domain"]
    else:
        t0 = time.time()
        M = vacuum_matrix(spec)
        dom = coil_domain(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")          # ch2/g4: per-process temp file (parallel cache shards)
        np.savez(tmp, M=M.astype(np.float32), domain=dom); os.replace(tmp, path)
        if verbose:
            print(f"  {spec.name}: vacuum matrix + domain built in {time.time() - t0:.0f}s -> {path.name}", flush=True)
    gkey = hashlib.sha1(np.concatenate([spec.grid_R, spec.grid_Z]).tobytes()).hexdigest()[:12]
    gpath = XG_CACHE / "machines" / f"cells_{gkey}.npy"
    if gpath.exists():
        Gf = np.load(gpath, mmap_mode="r")
    else:
        Gf = cell_greens_full(spec.grid_R, spec.grid_Z)
        tmp = gpath.with_suffix(".tmp.npy"); np.save(tmp, Gf); os.replace(tmp, gpath)
    iz, ir = np.nonzero(dom)
    G = np.ascontiguousarray(Gf[iz * len(spec.grid_R) + ir])
    return Machine(spec, M.astype(np.float32), dom, iz, ir, G)


# ----------------------------------------------------------------------------- spec sources
def spec_from_coil_table(name, row, grid_R, grid_Z, columns, column_units, tf_turns, **kw) -> MachineSpec:
    """Spec from a coil table (coil_name, coil_input_column, coil_R/Z/width/height/angle1/angle2), one row per element.

    column_units[col] = (mode, factor): 'total' -> the column value is the coil's total ampere-turns, shared
    equally by its elements; 'element' -> the column value is the current in every element; 'none' -> no
    poloidal flux (e.g. TF or feed currents). factor multiplies the value (unit or documented convention fixes)."""
    cic = np.asarray(row["coil_input_column"]).astype(str)
    el = np.stack([np.asarray(row[k], float) for k in ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")], 1)
    el_col = np.array([columns.index(c) if c in columns else -1 for c in cic])
    keep = el_col >= 0
    el, el_col, cic = el[keep], el_col[keep], cic[keep]
    scale = np.zeros(len(el))
    for c in set(cic):
        mode, factor = column_units.get(c, ("none", 0.0))
        m = cic == c
        scale[m] = {"total": factor / m.sum(), "element": factor, "none": 0.0}[mode]
    return MachineSpec(name, np.asarray(grid_R, float), np.asarray(grid_Z, float), list(columns), el, el_col, scale,
                       tf_turns, **kw)


def _parquet_row(subdir):
    import glob
    import pandas as pd
    f = sorted(glob.glob(str(ROOT / "challenge_data" / "data" / subdir / "*.parquet")))[0]
    return pd.read_parquet(f, columns=["coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                                       "coil_angle1", "coil_angle2"]).iloc[0]


# DIII-D E-coil correction, fitted on 1,046 frames of 120 training shots (xg/debug_ecoil2.py): the shipped ECOILA
# rectangle stands for 48 single-turn elements plus a co-located group (ECOILB) that is not shipped, so one scale cannot
# give both the fringe pattern and the near-uniform flux of the solenoid. psi_E = I_E (-22.05 G_rect + 1.046e-5 Wb/rad/A);
# vacuum residual against EFIT (with the exact EFIT current) drops from 0.19 to 0.0051 Wb/rad (frame-centred 0.0027).
D3D_E_PATTERN, D3D_E_UNIFORM, D3D_OFFSET = -22.05, 1.046e-5, -0.00322


def d3d_spec() -> MachineSpec:
    """DIII-D as shipped in the challenge. Cached currents: F-coils in A-turn (one element each), E-coil in A.
    Documented data corrections (DIII-D labels allowed, see D3D_E_*): E-coil pattern scale and uniform solenoid flux,
    stored maps = -psi_phys + D3D_OFFSET."""
    b = np.load(D3D_CACHE["train"] / "basis.npz")
    cols = [str(c) for c in b["cols"]]
    units = {c: ("total", 1.0) for c in cols}
    units["magnetics_ECOILA"] = ("total", D3D_E_PATTERN)
    return spec_from_coil_table("DIII-D", _parquet_row("diii_d_train"), b["grid_R"], b["grid_Z"], cols, units,
                                tf_turns=144, stored_sign=-1.0, stored_offset=D3D_OFFSET, n_sub=8,
                                col_uniform={"magnetics_ECOILA": D3D_E_UNIFORM},
                                notes="E-coil pattern/uniform terms and offset are DIII-D data-convention fixes")


def mast_spec(sol_factor: float = 1.0) -> MachineSpec:
    """MAST as shipped. Cached currents: P-coil columns in A-turn per coil (shared by its elements), the solenoid
    in A per turn on 656 single-turn elements (x sol_factor: the shipped value may be twice the true one),
    tf and efps carry no poloidal flux. Stored maps are psi_phys (axis = maximum)."""
    b = np.load(MAST_CACHE["test"] / "basis.npz")
    cols = [str(c) for c in b["cols"]]
    units = {c: ("total", 1.0) for c in cols if c.startswith("magnetics_p")}
    units["magnetics_sol_current"] = ("element", sol_factor)
    return spec_from_coil_table("MAST", _parquet_row("mast_public_test"), b["grid_R"], b["grid_Z"], cols, units,
                                tf_turns=24, solenoid_cols=("magnetics_sol_current",), n_sub=3,
                                notes=f"solenoid factor {sol_factor}")


# ----------------------------------------------------------------------------- synthetic test machines
def synth_spec(machine: str) -> MachineSpec:
    """Synthetic-set machines (SPARC, ITER) from the generator's coil table: one current column per circuit (records
    in A-turn), each element carrying polarity / n_elements of it (single filaments, as FreeGSNKE describes these
    coils); TF total ampere-turns (tf_turns = 1); wall = the set's limiter outline. Stored maps: axis = maximum."""
    key = SYNTH_SETS[machine]
    d = np.load(GEN / f"{key}_synthetic" / f"{key}_synthetic_500.npz", allow_pickle=True)
    import glob
    import pandas as pd
    row = pd.read_parquet(sorted(glob.glob(str(GEN / f"{key}_synthetic" / "parquet" / "*.parquet")))[0],
                          columns=["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1",
                                   "coil_angle2", "coil_polarity"]).iloc[0]
    cols = [str(c) for c in d["input_names"] if str(c) not in ("magnetics_plasma_current", "magnetics_tf_current")]
    cic = np.asarray(row["coil_input_column"]).astype(str)
    el = np.stack([np.asarray(row[k], float) for k in ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")], 1)
    pol = np.asarray(row["coil_polarity"], float)
    el_col = np.array([cols.index(c) for c in cic])
    n_el = np.array([np.sum(cic == c) for c in cic])
    wall = np.stack([d["limiter_R"].astype(float), d["limiter_Z"].astype(float)], 1)
    return MachineSpec(machine, d["grid_R"].astype(float), d["grid_Z"].astype(float), cols, el, el_col, pol / n_el,
                       tf_turns=1.0, wall=wall, n_sub=1, stored_sign=1.0, stored_offset=0.0,
                       notes=f"{key} synthetic set: FreeGSNKE coil table, limiter wall")


D3DS_E_TURNS = 152.8      # the synthetic DIII-D generator's E-coil model (more_data/generated/code/d3d_machine.py)


def d3d_synth_spec() -> MachineSpec:
    """The synthetic DIII-D set's machine: the challenge coil table as FreeGSNKE used it (E-coil = +152.8 turns on the
    shipped rectangle, no uniform term, no offset), wall = the scorer envelope outline the generator used as limiter.
    Stored maps follow the DIII-D EFIT convention (axis = minimum) like the real set."""
    b = np.load(D3D_CACHE["train"] / "basis.npz")
    cols = [str(c) for c in b["cols"]]
    units = {c: ("total", 1.0) for c in cols}
    units["magnetics_ECOILA"] = ("total", D3DS_E_TURNS)
    lim = np.load(GEN / "d3d_synthetic" / "d3d_limiter_65x65.npz") if (GEN / "d3d_synthetic" / "d3d_limiter_65x65.npz").exists() else None
    wall = np.stack([lim["limiter_R"], lim["limiter_Z"]], 1).astype(float) if lim is not None else None
    return spec_from_coil_table("D3DS", _parquet_row("diii_d_train"), b["grid_R"], b["grid_Z"], cols, units,
                                tf_turns=144, stored_sign=-1.0, stored_offset=0.0, n_sub=8, wall=wall,
                                notes="synthetic DIII-D set: FreeGSNKE E-coil convention (+152.8 turns)")


def multi_spec(name: str) -> MachineSpec:
    if name in M2_MACHINES:          # the real MAST machine as shipped, renamed so the caches / gains are separate
        import dataclasses
        return dataclasses.replace(mast_spec(M2_SOL[name]), name=name, notes=f"synthetic MAST set on the shipped MAST geometry, solenoid x{M2_SOL[name]}")
    """Training machine from cache/pegs/multi/<name>/spec.npz: columns = circuits (value = A per turn), elements = one
    row per FreeGSNKE filament with el_scale = ampere-turns per ampere of its circuit, wall = limiter polygon, tf_turns 1
    (itf = total TF ampere-turns). Stored maps are physical (axis = maximum)."""
    d = np.load(MULTI_ROOT / name / "spec.npz", allow_pickle=True)
    return MachineSpec(name, d["grid_R"].astype(float), d["grid_Z"].astype(float), [str(c) for c in d["columns"]],
                       d["elements"].astype(float), d["el_col"].astype(int), d["el_scale"].astype(float),
                       tf_turns=float(d["tf_turns"]), wall=d["wall"].astype(float), n_sub=2, stored_sign=1.0, stored_offset=0.0,
                       notes="synthetic training set (FreeGSNKE, public OMAS geometry)")


def multi_machine(name: str) -> Machine:
    return build(multi_spec(name))


def synth_machine(machine: str) -> Machine:
    return build(d3d_synth_spec() if machine == "D3DS" else synth_spec(machine))


def d3d() -> Machine:
    return build(d3d_spec())


def mast(sol_factor: float = 1.0) -> Machine:
    return build(mast_spec(sol_factor))


if __name__ == "__main__":
    for mk, cache in ((d3d, D3D_CACHE["train"]), (mast, MAST_CACHE["test"])):
        m = mk()
        d = m.domain
        RR, ZZ = np.meshgrid(m.grid_R, m.grid_Z)
        print(f"{m.name}: domain {int(d.sum())} cells, R {RR[d].min():.3f}-{RR[d].max():.3f}, Z {ZZ[d].min():.3f}-{ZZ[d].max():.3f}; "
              f"G_cells {m.G_cells.shape}; elements {len(m.spec.elements)}")
        # cross-check the generic vacuum matrix against the per-machine caches built earlier
        b = np.load(cache / "basis.npz")
        G = b["G"].reshape(len(b["cols"]), -1).astype(np.float64)
        if m.name == "DIII-D":
            ref = G.copy(); ie = list(b["cols"]).index("magnetics_ECOILA"); ref[ie] = m.M[ie]   # E-coil changed on purpose
        else:
            ref = np.zeros_like(G)
            for i, c in enumerate(b["cols"]):
                if str(c).startswith("magnetics_p"):
                    ref[i] = G[i] / b["counts"][i]
                elif str(c) == "magnetics_sol_current":
                    ref[i] = G[i]
        rel = np.abs(m.M - ref).max(1) / np.abs(ref).max(1).clip(1e-30)
        print("   max rel. difference to the earlier per-machine basis, per column:", np.round(rel, 4).tolist())
        if m.name == "DIII-D":
            e = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")["mask_coarse"].astype(bool)
            print(f"   scorer envelope {int(e.sum())} cells; envelope cells outside our domain: {int((e & ~d).sum())}")
