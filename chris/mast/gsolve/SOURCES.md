# Sources of the gsolve data-generation code

Generates the two gsolve datasets that Chris's models train on: `cond1000` (Z17, Z18) and `cond3_1000` (FA, FC). CPU only.
Every file is a verbatim copy, checked byte for byte (0 differences), except one word added for clarity in
`recipes/tune2/HANDOFF.md`: "Synthetic MAST label medians" (the original reads "MAST label medians"; `labels_runs.py`
computes them from generated equilibria only).

| Folder | Source | Version |
|---|---|---|
| `solver/` | Artur Toshev's gsolve | commit `dd3db1e` of 9 October 2026, its only commit, no local changes (21 files); `solver/LICENSE` added |
| `recipes/` | `my_experiments/gsolve_recipes/` in the starter (commit `46527a2`), itself a copy of `/fsx/chris.beckham/gsolve_data/`; every file also matches the `/fsx/chris.beckham/gsolve_data/` original that ran | `tune2/`, `tune3/`, `convert/`, `cond1000/`, `cond3_1000/`, `s1000/shots.txt`, notes and the environment list (231 files) |
| `recipes/tune2/cond_model.pkl`, `recipes/tune3/cond_model.pkl`, `recipes/tune3/sel_critic.pkl`, `recipes/tune3/sel_critic_v1.pkl` | fitted helper models from `/fsx/chris.beckham/gsolve_data/` | the two `cond_model.pkl` are identical; `sel_critic.pkl` = `sel_critic_v1.pkl` |

## gsolve

gsolve is a free-boundary Grad-Shafranov solver written from scratch in PyTorch by Artur Toshev, with an AI coding
assistant (Claude Code). It contains no code from FreeGS or FreeGSNKE. It takes FreeGS's `ConstrainPaxisIp` profile form
and was checked against FreeGS 0.8.2 (`solver/README.md`, section "Validation"); FreeGS appears only in that test, run by
`solver/tests/freegs_run.py` in a separate Python environment. Its coil files come from the challenge data
(`solver/scripts/build_machine_files.py`). Artur is not a member of team zenithon-zendaya;
he released gsolve under the MIT licence for this package (`solver/LICENSE`).

## How each set was made

| Set | Command (run from the recipe folder) | Knobs as run |
|---|---|---|
| `cond1000` | `tune2/final2.sh knobs_cond_base.json` | `recipes/cond1000/knobs.json` |
| `cond3_1000` | `NAME=cond3_1000 TAG=gsolve_cond3_1000 ... tune3/final3.sh knobs_final_sel.json` (`tune3/HANDOFF.md`) | `recipes/cond3_1000/knobs.json` |

Each launcher writes `knobs.json`, splits the 1,000 shots of `s1000/shots.txt` into shards, runs `gen_param2.py` per shard
(the solver), converts the output with `convert_perslice.py` into `gen4_layout/`, then builds the training cache with
`convert/build_set.sh` (which calls Chris's `synth_cache_seq.py` and `synth_shape_targets.py`).

## Fixed settings

- `cond_model.pkl`: per-frame li and pressure-ratio model fitted on DIII-D training data by `tune2/fit_prior.py`.
- `sel_critic_v1.pkl`: variant-selection critic fitted by `tune3/sel_critic.py`.

## Environment

The generation ran in its own virtual environment (`/fsx/chris.beckham/gsolve_data/tune/venv`, Python 3.11.16), listed in
`recipes/requirements_gsolve_venv.txt` (pip freeze), with `PYTHONPATH` pointing at the solver.
