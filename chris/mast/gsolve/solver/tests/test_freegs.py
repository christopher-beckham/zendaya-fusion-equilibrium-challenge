"""Comparison with FreeGS 0.8.2 on MAST cross-check cases (same coils, profile, Ip and box limiter; coils up-down
averaged so the problem is symmetric).

test_stored_freegs_maps: against FreeGS maps on a 129 x 129 grid stored in tests/data (always runs).
test_live_freegs: runs FreeGS itself (65 x 65 grid, about 20 s) through tests/freegs_run.py with the interpreter in
$GSOLVE_FREEGS_PYTHON, or this one if it can import freegs; skipped otherwise.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gsolve import FwdConfig, Solution, analyse, coil_flux, forward_solve, load_machine, start_current
from gsolve.greens import filaments, greens

HERE = Path(__file__).parent
REF = np.load(HERE / "data" / "regression.npz")
M = load_machine("mast")


def ours(k):
    prof = dict(mode="paxis", paxis=[float(REF[f"fgs{k}_paxis"])], alpha_m=[float(REF[f"fgs{k}_alpha_m"])],
                alpha_n=[float(REF[f"fgs{k}_alpha_n"])])
    J0 = start_current(M, [0.9], [0.0], [0.55], [1.8], [0.3])
    return forward_solve(M, torch.as_tensor(coil_flux(M, REF[f"fgs{k}_I_sym"])), [float(REF[f"fgs{k}_ip_A"])], prof,
                         FwdConfig(n_iter=300, tol=1e-5, symmetrize=True), J0=J0)


def as_solution(psi, ref, k):
    """A FreeGS map wrapped as a Solution, so both maps go through the same diagnostics."""
    info = {q: torch.tensor([float(ref[q])], dtype=torch.float64) for q in ("R_ax", "Z_ax", "psi_ax", "psi_b", "L", "beta0")}
    info["status"] = torch.zeros(1, dtype=torch.int64)
    prof = {q: torch.tensor([float(REF[f"fgs{k}_{v}"])], dtype=torch.float64)
            for q, v in (("paxis", "paxis"), ("alpha_m", "alpha_m"), ("alpha_n", "alpha_n"))}
    return Solution(M, torch.as_tensor(np.asarray(psi, np.float64))[None], torch.tensor([float(REF[f"fgs{k}_ip_A"])]),
                    dict(mode="paxis", **prof), info)


def compare(sol, fgs_sol, k):
    RR, ZZ = np.meshgrid(M.R, M.Z)
    r0, r1, z0, z1 = M.wall_box
    box = (RR >= r0) & (RR <= r1) & (ZZ >= z0) & (ZZ <= z1)
    a, b = sol.psi[0].numpy()[box], fgs_sol.psi[0].numpy()[box]
    r2 = 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    d1, d2 = analyse(sol, 0, float(REF[f"fgs{k}_tf_kA"])), analyse(fgs_sol, 0, float(REF[f"fgs{k}_tf_kA"]))
    # (kappa is left out: in these near-double-null cases a tiny up-down asymmetry decides which X-point bounds the
    # plasma, which moves the LCFS tip by a few cm)
    out = dict(r2=r2, dR_ax=d1["R_ax"] - d2["R_ax"], dq95=d1["q95"] / d2["q95"] - 1, dbetaN=d1["betaN"] / d2["betaN"] - 1)
    print(k, {q: round(float(v), 6) for q, v in out.items()})
    return out


def test_stored_freegs_maps():
    for k in (4, 12):
        sol = ours(k)
        assert int(sol.status[0]) == 0
        ref = {q: REF[f"fgs{k}_{q}"] for q in ("R_ax", "Z_ax", "psi_ax", "psi_b", "L", "beta0")}
        c = compare(sol, as_solution(REF[f"fgs{k}_psi"], ref, k), k)
        assert c["r2"] > 0.9998 and abs(c["dR_ax"]) < 0.003 and abs(c["dq95"]) < 0.02 and abs(c["dbetaN"]) < 0.02


def _freegs_python():
    exe = os.environ.get("GSOLVE_FREEGS_PYTHON")
    if exe:
        return exe
    return sys.executable if importlib.util.find_spec("freegs") else None


@pytest.mark.skipif(_freegs_python() is None, reason="FreeGS not installed (set GSOLVE_FREEGS_PYTHON)")
def test_live_freegs(tmp_path):
    k = 4
    e = M.elements
    Rf, Zf, wf, col = [], [], [], []
    for i in range(len(e["R"])):
        r, z = filaments(e["R"][i], e["Z"][i], e["width"][i], e["height"][i], e["angle1"][i], e["angle2"][i],
                         step=M.filament_step)
        amps = 1e3 * e["turns"][i] * M.column_scale.get(M.columns[e["col"][i]], 1.0)
        Rf.append(r); Zf.append(z); wf.append(np.full(r.size, amps / r.size)); col.append(np.full(r.size, e["col"][i]))
    pts = np.array([[0.5, 0.0, 0.8, 0.3], [1.2, -0.4, 0.3, 0.9], [0.15, 1.4, 1.5, 0.0]])
    np.savez(tmp_path / "in.npz", Rf=np.concatenate(Rf), Zf=np.concatenate(Zf), wf=np.concatenate(wf), col=np.concatenate(col),
             I_kA=REF[f"fgs{k}_I_sym"], ip=REF[f"fgs{k}_ip_A"], fvac=M.fvac(REF[f"fgs{k}_tf_kA"]), paxis=REF[f"fgs{k}_paxis"],
             am=REF[f"fgs{k}_alpha_m"], an=REF[f"fgs{k}_alpha_n"], R0=M.R0, box=M.wall_box, green_pts=pts,
             grid_R=[M.R[0], M.R[-1], len(M.R)], grid_Z=[M.Z[0], M.Z[-1], len(M.Z)])
    subprocess.run([_freegs_python(), "-W", "ignore", str(HERE / "freegs_run.py"), str(tmp_path / "in.npz"),
                    str(tmp_path / "out.npz")], check=True, timeout=1200)
    f = np.load(tmp_path / "out.npz")
    assert np.allclose(f["greens"], greens(*pts.T), rtol=1e-10)              # same Green's function
    assert bool(f["converged"])
    c = compare(ours(k), as_solution(f["psi"], {q: f[q] for q in ("R_ax", "Z_ax", "psi_ax", "psi_b", "L", "beta0")}, k), k)
    assert c["r2"] > 0.999 and abs(c["dR_ax"]) < 0.005 and abs(c["dq95"]) < 0.03
