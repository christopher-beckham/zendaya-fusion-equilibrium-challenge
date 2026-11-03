# gsolve synthetic-MAST recipes (code and notes)

Copy of the code, knob files, reports and shot lists from `/fsx/chris.beckham/gsolve_data/`, where the gsolve
datasets were generated (that folder is not a git repository). Data, solver outputs, evaluation runs, claims and
logs stay on /fsx.

| Folder | What it is |
|---|---|
| `tune/` | real-vs-synthetic input critic and the critic-tuned recipe (`REPORT.md`, `HANDOFF.md`) |
| `tune2/` | DIII-D-taught conditional recipe: li and pressure ratio predicted per frame (`HANDOFF.md`) |
| `tune3/` | DIII-D flux-map-supervised recipe: height rule and critic-guided selection (`REPORT.md`, `HANDOFF.md`) |
| `TUNE3_BRIEF.md` | task brief for tune3 |
| `s1000/`, `cond1000/`, `cond2z1000/`, `cond3_1000/`, ... | knobs, shard and shot lists of each generated set |
| `convert/` | conversion of solver output into the training-set layout (`build_set.sh`) |
| `pilots/` | 60–150-shot pilot sets and the pilot queue |

Environment: `requirements_gsolve_venv.txt` (pip freeze of `tune/venv`). Solver: Artur's gsolve, local clone
`/fsx/chris.beckham/gsolve_copy` at the commit in `gsolve_copy_commit.txt`, no local changes.

Model files the recipes load (`cond_model.pkl`, `sel_critic_v1.pkl`) are artifacts and live next to the scripts on
/fsx; they are rebuilt by `fit_prior.py` (tune2) and `sel_critic.py` (tune3).
