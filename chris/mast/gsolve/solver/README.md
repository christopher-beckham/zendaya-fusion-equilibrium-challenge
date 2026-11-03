# gsolve

A small free-boundary Grad-Shafranov solver for MAST and DIII-D, written in PyTorch and run on CPU. It solves
hundreds of equilibria at once, and it can make synthetic MAST equilibria seeded by real challenge input data
(coil currents, plasma current, Thomson scattering).

**Terms.** An *equilibrium* is the magnetic field configuration in which the plasma pressure is balanced by
magnetic forces. It is described by the *poloidal flux* ψ(R, Z) on a grid (R is the major radius, Z the height).
The *Grad-Shafranov (GS) equation* links ψ to the plasma current density
J = R p′(ψ) + FF′(ψ)/(μ0 R), where p is the pressure and F = R·B_toroidal. *Free boundary* means the plasma
edge is not given: it comes out of the solution, set either by an X-point (a point where the poloidal field
vanishes; the plasma is then *diverted*) or by contact with the wall (*limited*). ψ_N is the normalised flux:
0 on the magnetic axis, 1 on the plasma edge (the *last closed flux surface*, LCFS).

## What it does

- **Forward mode** (`forward_solve`): given coil currents, the plasma current Ip and the pressure and current
  profiles, it finds the equilibrium by Picard iteration (flux → axis, edge and ψ_N → current → flux, repeated).
  Two profile modes:
  - `"paxis"`: the FreeGS `ConstrainPaxisIp` model. Three numbers: pressure on axis, and the shape exponents
    α_m, α_n of (1 − ψ_N^α_m)^α_n.
  - `"tab"`: a tabulated pressure p(ψ_N) plus one FF′ shape (parametric or tabulated).
- **Vertical hold.** Elongated plasmas are vertically unstable, and with fixed coil currents the iteration drifts
  up or down. `z_hold` moves the current to a chosen height each pass and adds the small radial field that makes
  the net vertical force zero; this stands for the vertical feedback and vessel currents that hold a real plasma.
  `r_hold` adds a uniform vertical "vessel" field that places the outer midplane edge at a chosen radius.
- **Batching.** All frames of a batch are solved together as tensors; the plasma edge is found without contours
  (a "bottleneck" flood fill over the grid), so the whole loop is batched.
- **Wall-edge fix** (on by default, `FwdConfig.wall_fix`): for a plasma touching the wall, the edge flux is read on
  the wall lines themselves. Without it the MAST centre column acted as if it were at R = 0.18 m instead of 0.20 m.
- **Correction mode** (`correct`, optional): the equilibrium closest to a given flux map (for example a neural
  network's prediction). The coil flux is kept exactly; a few profile weights and vacuum fields are fitted.
  `CorrConfig` gives the source of every default next to it.
- **Coil force-balance height** (`balance_height`): the height at which the measured coil currents exert no net
  vertical force on a nominal plasma shape.
- **Diagnostics** (`analyse`, `analyse_all`): magnetic axis, X-point, LCFS, q95 (safety factor at ψ_N = 0.95),
  βN (normalised plasma pressure), li (internal inductance), elongation κ and upper / lower triangularity δ.

Coils are modelled exactly as circular current filaments filling each coil element (elliptic-integral Green's
functions). The grid is the challenge's 65 × 65 grid. Plasma current may flow only inside a wall box, minus
the cells within 3 cm of an in-vessel coil.

## Install

```bash
pip install -e .            # numpy, scipy, torch (a CPU build of torch is enough)
pip install pyarrow         # only for examples/generate_synthetic.py
```

The first use of a machine builds its two matrices (coil flux, plasma-cell flux) in a few seconds and caches
them in `~/.cache/gsolve` (or `$GSOLVE_CACHE`), about 40 MB for MAST.

## Usage

```python
import numpy as np, gsolve
m = gsolve.load_machine("mast")                         # solenoid x0.5 (solenoid_scale=1.0: column as stored)
I = np.array([[99, 101, 0.7, 0.8, -187, -188, -147, -146, 1.8, -2.1, -25.]])  # coil currents (kA), order m.columns
ext = gsolve.coil_flux(m, I)                            # vacuum flux of the coils, (1, 65, 65)
prof = dict(mode="paxis", paxis=[2e4], alpha_m=[2.0], alpha_n=[1.5])           # pressure on axis 20 kPa
J0 = gsolve.start_current(m, [0.9], [0.0], [0.55], [1.8], [0.3])               # rough D-shaped start
sol = gsolve.forward_solve(m, ext, [9e5], prof, gsolve.FwdConfig(n_iter=300), z_hold=[0.0], J0=J0)
print(sol.status, sol.R_ax, sol.Z_ax)                   # status 0 = converged
d = gsolve.analyse(sol, 0, i_tf_kA=85.0)                 # also X-point, LCFS, triangularity, ...
print({k: round(float(d[k]), 3) for k in ("q95", "betaN", "li", "kappa")})
```

`examples/generate_synthetic.py` makes synthetic MAST equilibria from challenge-format shot files (seed frame →
profiles from Thomson scattering → solve → Thomson data remapped onto the solved flux surfaces → output file).
It reads input columns only. Every recipe setting is a command-line default or a constant with its source
(physics, published documentation, DIII-D, or the frame's own inputs) written next to it.

```bash
python examples/generate_synthetic.py /data/mast_public_test/mast_shot_*.parquet --out synth.npz
```

## Data use

The package reads the machine description, coil currents, plasma current, toroidal-field current and Thomson
profiles. It contains no MAST equilibrium reconstruction and no setting derived from one, and it does not
download data. The only reconstructions it was compared with are DIII-D EFIT equilibria, in the validation
table below. The MAST solenoid factor is a documented, user-settable convention (see Limitations).

## Validation

**Against FreeGS 0.8.2** (an established open-source free-boundary solver). Same coils (the challenge's 812 MAST
coil elements), same Ip, same `ConstrainPaxisIp` profile, same box limiter. 40 MAST frames × 2 current shapes;
medians over the 49 cases where both converged. R²ψ is the fit of the flux map inside the wall box; "boundary"
is the mean distance between the two LCFS curves.

| Comparison | R²ψ | Axis shift | Boundary | X-point shift | q95 | βN |
|---|---|---|---|---|---|---|
| FreeGS 65² vs FreeGS 129² (FreeGS's own grid error) | 1.0000 | 0.16 cm | 0.23 cm | 0.11 cm | −0.3% | +0.3% |
| gsolve vs FreeGS 129², symmetric coils | 0.99996 | 0.02 cm | 0.35 cm | 0.31 cm | 0.0% | +0.4% |
| gsolve vs FreeGS, real coils, same vertical hold | 0.99999 | −0.15 cm | 0.41 cm | 0.06 cm | −0.4% | 0.0% |
| gsolve without the wall-edge fix vs FreeGS 129² | 0.9989 | −1.0 cm | 1.3 cm | 0.45 cm | +2.8% | −0.5% |
| DIII-D, 10 frames: gsolve vs FreeGS 129² | 0.9998 | −0.5 cm | 0.66 cm | 0.34 cm | +2.2% | +0.3% |
| DIII-D: gsolve vs EFIT reconstruction | 0.9843 | −3.4 cm | 2.5 cm | 4.9 cm | +2.7% | +3.1% |
| DIII-D: FreeGS 129² vs EFIT reconstruction | 0.9855 | −3.0 cm | 2.2 cm | 4.8 cm | −2.5% | +3.7% |

Both solvers fail on the same cases (very peaked current at high pressure runs into the outer wall). On DIII-D
both sit about 3 cm inboard of EFIT: the 3-number profile model is the limit there, not the solver. (DIII-D
cases used profiles fitted to the EFIT current and the same fitted vacuum fields in both solvers.)

**Tests in this repository** (`pytest tests/`, 10 tests, about 10 s on 16 CPU threads, 25 s with the live FreeGS solve):

- Green's function against direct Biot-Savart quadrature (agreement to 1e-9) and the far-field dipole limit.
- Regression against stored reference solutions (`tests/data/regression.npz`, written once by the code this
  package was extracted from, on MAST challenge input frames; no measured equilibrium): three forward cases
  (paxis + symmetrize, paxis + hold + wall fix, tabulated pressure + hold + vessel field) and two correction
  settings: largest flux difference 7e-7 of the flux range, identical iteration counts and status. Coil
  and cell matrices match the reference to 1e-6.
- The default correction setting, the force-balance height (symmetric coils give the midplane) and the
  solenoid option.
- Diagnostics against the challenge starter kit's shape code and an independent q95 loop integral (q95 within 2%).
- FreeGS: two stored FreeGS 129² maps (R²ψ 0.99995–0.99996, axis within 0.2 cm, q95 within 1.8%, βN within
  0.5%), and a live FreeGS 65² solve when FreeGS is available (`GSOLVE_FREEGS_PYTHON=/path/to/python`; it
  also checks that FreeGS's Green's function equals ours to 1e-10).

## Speed

16 CPU threads, MAST: the example recipe solves 1,000 two-pass equilibria (pass 1, then pass 2 with the pressure
remapped) in about 40 s, and makes 1,000 finished synthetic equilibria (4 candidates per frame, selection,
equilibrium numbers) in about 170 s. FreeGS needs about 15 s per solve per core at 65² (50 s at 129²); for 50,000
MAST equilibria that is about 1 node-hour for gsolve against about 210 core-hours for FreeGS.

## Limitations

- **Profiles.** Three numbers (paxis mode), or a pressure table plus one current shape. Real profiles have more
  freedom (pedestals, hollow current); on DIII-D this model puts the plasma about 3 cm inboard of EFIT whatever
  the solver.
- **Vertical position.** With real, slightly up-down asymmetric coil currents and no hold, the plasma drifts to
  the top or bottom wall (in FreeGS too). The radial field needed to hold it is only about ±0.4 mT against
  about 130 mT of vertical field, so the height is set by sub-mT fields that no input measures. Use `z_hold`;
  `balance_height` gives the height at which the coils alone hold a nominal plasma.
  Picard iteration cannot settle on unstable equilibria by itself (FreeGSNKE's Newton-Krylov solver can).
- **Wall.** A rectangular box (MAST: R 0.20–1.55 m, |Z| ≤ 1.6 m) minus cells near in-vessel coils, not the real
  first-wall shape. No vessel or passive-structure currents; `r_hold` and the correction mode's vacuum fields
  stand in for them.
- **Coils.** MAST: the solenoid column is used at ×0.5 by default. This rests on documentation alone: the
  organisers' data notes state that `magnetics_sol_current` reads exactly 2× the level-2 `pf_active` solenoid
  current and leave the convention open, and Berkery et al. 2021 (UKAEA-CCFE-PR(21)79) give a MAST solenoid
  current of about +22 kA before the swing, half the +44 to +46 kA the column reads before breakdown in the
  challenge inputs. Set it with `load_machine("mast", solenoid_scale=1.0)` or `--solenoid-scale`. DIII-D:
  all coil columns at face value; the E coil's true scale is unknown, and coils alone do not reproduce DIII-D
  equilibria without extra vacuum fields.
- **Grid.** Fixed 65 × 65 challenge grids. Diagnostics trace flux surfaces along rays from the axis, which assumes
  each surface is visible from the axis (true for normal tokamak shapes). The challenge scorer traces the LCFS on
  the coarse grid and lets it reach half a cell past the wall, so its κ comes out about 0.02–0.05 higher and its
  li 3–5% higher than ours.

## Layout

```
gsolve/geometry.py      machine description, loaded from gsolve/machines/<name>.json + <name>_coils.csv
gsolve/greens.py        Green's function, coil and plasma-cell matrices (cached)
gsolve/forward.py       forward solve, profiles, vertical hold, axis / edge finding
gsolve/correct.py       correction mode
gsolve/diagnostics.py   axis, X-point, LCFS, q95, betaN, li, kappa, delta
examples/               synthetic MAST generator
scripts/                rebuilds the coil files from the challenge data
tests/                  tests and stored reference data
```
