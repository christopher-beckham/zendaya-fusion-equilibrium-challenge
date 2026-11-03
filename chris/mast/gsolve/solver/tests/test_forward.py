"""Regression against stored reference solutions.

tests/data/regression.npz was written once by the solver code this package was extracted from, on MAST
challenge input frames (coil currents, Ip, TF current; pressure on axis 2.5 x the Thomson peak electron
pressure) and FreeGS 0.8.2 solves of the same inputs; it holds no measured equilibrium. Forward cases:
A paxis profile, averaged coils, symmetrized, no wall fix; B paxis profile, real coils, vertical hold at Z = 0,
wall fix; C tabulated pressure, hold at Z = -0.03 m, vessel field steering the outer edge to 1.30 m, wall fix.
Correction mode: case B's solution moved up by one grid row, corrected with the two settings below.
"""
from pathlib import Path

import numpy as np
import torch

from gsolve import (CorrConfig, FwdConfig, analyse, balance_height, coil_flux, correct, forward_solve, load_machine,
                    start_current)

REF = np.load(Path(__file__).parent / "data" / "regression.npz")
M = load_machine("mast")
# the two correction settings the stored reference was made with (explicit, so the test does not follow defaults)
REF_SIMPLE = CorrConfig(n_iter=30, tol=5e-3, max_shift=0.30, profile="pf3", vacuum=("H0", "H1", "H2", "H4"), w_out=0.25)
REF_RICH = CorrConfig(n_iter=30, tol=5e-3, max_shift=0.30, profile="pf4e", signed=True,
                      vacuum=tuple(f"H{j}" for j in range(9)), w_out=0.25)


def solve_case(c):
    ext = torch.as_tensor(coil_flux(M, REF[f"{c}_I_kA"]))
    am, an = REF[f"{c}_alpha"]
    prof = dict(mode="paxis" if c != "C" else "tab", alpha_m=[am], alpha_n=[an])
    if c == "C":
        prof["p_tab"] = REF["C_p_tab"][None]
    else:
        prof["paxis"] = [float(REF[f"{c}_paxis"])]
    cfg = FwdConfig(n_iter=300, tol=1e-5, symmetrize=c == "A", wall_fix=c != "A")
    J0 = start_current(M, [0.9], [0.0], [0.55], [1.8], [0.3])
    return forward_solve(M, ext, [float(REF[f"{c}_ip_A"])], prof, cfg, J0=J0,
                         z_hold={"A": None, "B": [0.0], "C": [-0.03]}[c], r_hold=[1.30] if c == "C" else None)


def test_forward_regression():
    for c in "ABC":
        sol = solve_case(c)
        ref = REF[f"{c}_psi"].astype(np.float64)
        fr = REF[f"{c}_psi_ax"] - REF[f"{c}_psi_b"]
        err = np.abs(sol.psi[0].numpy() - ref).max() / fr
        print(c, "max |dpsi| / flux range", err, "iterations", sol.n_iter, REF[f"{c}_n_iter"])
        assert int(sol.status[0]) == int(REF[f"{c}_status"]) == 0
        assert err < 1e-4
        for q, tol in (("R_ax", 1e-4), ("Z_ax", 1e-4), ("h_br", 1e-3), ("bv", 1e-3)):
            assert abs(float(sol.info[q][0]) - REF[f"{c}_{q}"]) < tol, (c, q)
        for q in ("L", "beta0", "lam"):
            if f"{c}_{q}" in REF.files:
                assert np.isclose(float(sol.info[q][0]), REF[f"{c}_{q}"], rtol=1e-4), (c, q)


def test_correction_regression():
    psi_model = torch.as_tensor(REF["corr_model"].astype(np.float64))[None]
    ext = torch.as_tensor(coil_flux(M, REF["B_I_kA"]))
    for tag, cfg in (("simple", REF_SIMPLE), ("rich", REF_RICH)):
        sol = correct(M, psi_model, ext, torch.tensor([float(REF["B_ip_A"])], dtype=torch.float64), cfg)
        ref = REF[f"corr_{tag}_psi"].astype(np.float64)
        err = np.abs(sol.psi[0].numpy() - ref).max() / float(sol.psi_ax[0] - sol.psi_b[0])
        coef = np.concatenate([sol.coef[0].numpy(), sol.vac[0].numpy()])
        print(tag, "max |dpsi| / flux range", err, "coef diff", np.abs(coef - REF[f"corr_{tag}_coef"]).max())
        assert int(sol.status[0]) == REF[f"corr_{tag}_status"] == 0
        assert err < 1e-4
        assert np.allclose(coef, REF[f"corr_{tag}_coef"], atol=1e-3)


def test_default_correction_runs():
    """The default correction settings settle the same displaced equilibrium (no stored reference)."""
    psi_model = torch.as_tensor(REF["corr_model"].astype(np.float64))[None]
    ext = torch.as_tensor(coil_flux(M, REF["B_I_kA"]))
    sol = correct(M, psi_model, ext, torch.tensor([float(REF["B_ip_A"])], dtype=torch.float64))
    assert int(sol.status[0]) == 0
    assert abs(float(sol.R_ax[0]) - REF["B_R_ax"]) < 0.03 and abs(float(sol.Z_ax[0]) - REF["B_Z_ax"]) < 0.07


def test_balance_height_and_solenoid_scale():
    """Up-down symmetric coils hold a centred plasma near the midplane (within one 6 cm grid row; the wall
    region is not exactly symmetric); the solenoid factor is a user setting."""
    ext = coil_flux(M, REF["A_I_kA"])
    assert abs(balance_height(M, ext, [0.9], [0.55])[0]) < 0.0625
    m1 = load_machine("mast", solenoid_scale=1.0)
    assert m1.column_scale["sol"] == 1.0 and M.column_scale["sol"] == 0.5
    e0, e1 = coil_flux(M, np.eye(11)[10]), coil_flux(m1, np.eye(11)[10])
    assert np.allclose(e1, 2 * e0)


def test_diagnostics_against_scorer():
    """Shape numbers against the challenge scorer's shape code (starter kit), q95 against an independent
    loop-integral q95 with the vacuum F.

    The scorer traces the LCFS with marching squares on the 65 x 65 grid and lets it reach half a cell past the
    wall box, so its surface is slightly larger (kappa about 0.02-0.05 higher, li 3-5% higher); hence the loose
    tolerances. q95 uses the same definition in both and must agree closely."""
    for c in "ABC":
        sol = solve_case(c)
        d = analyse(sol, 0, float(REF[f"{c}_tf_kA"]))
        print(c, {k: round(float(v), 4) for k, v in d.items() if k != "lcfs"})
        assert abs(d["kappa"] - REF[f"{c}_scorer_kappa"]) < 0.06
        assert abs(d["delta_upper"] - REF[f"{c}_scorer_tri_top"]) < 0.08
        assert abs(d["delta_lower"] - REF[f"{c}_scorer_tri_bot"]) < 0.08
        assert abs(d["volume"] / REF[f"{c}_scorer_volume"] - 1) < 0.02
        assert abs(d["li"] / REF[f"{c}_scorer_li"] - 1) < 0.07
        # the reference q95 uses the vacuum F; ours uses F(0.95) from the solved FF'
        xs = np.linspace(0, 1, 201)
        _, ffp = sol.profiles(xs, 0)
        fr = float(sol.psi_ax[0] - sol.psi_b[0])
        fvac = float(M.fvac(REF[f"{c}_tf_kA"]))
        F95 = np.sqrt(fvac ** 2 + 2 * fr * np.trapezoid(ffp[xs >= 0.95], xs[xs >= 0.95]))
        assert abs(d["q95"] * fvac / F95 / REF[f"{c}_fp_q95_vac"] - 1) < 0.02
        assert 0.05 < d["betaN"] < 6
