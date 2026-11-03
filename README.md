# zendaya-fusion-equilibrium-challenge

Reproduction package of team zenithon-zendaya (Christopher Beckham, Zara Ercan) for the Fusion Equilibrium Challenge Final
(19 to 26 October 2026). It covers the two Final submissions under review:

| Submission | Entry | Files | Models |
|---|---|---|---|
| A | FW-EDF58354, Challenge 1 (`fec-final-sub1` @ `807b5104`) | `diii_d_private_test.npz` | DIII-D recipe v13 |
| B | FW-41B8A13C, Challenge 2 (`fec-final-sub5` @ `3a2a2a26`) | `diii_d_private_test.npz`, `mast_private_test.npz` | DIII-D recipe Ch1-min; MAST blend of 8 of Chris's models and Zara's r1 |

**Start with `REPRODUCE.md`**: from an empty GPU machine to the submitted files regenerated from the shipped weights, then
every model retrained from scratch.

## What the package contains (rule 6)

| Item | Where |
|---|---|
| Model source | `zara/diii-d/` (DIII-D), `chris/mast/` and `zara/mast/r1/` (MAST), `blend/` (MAST blend) |
| Pinned environment specification | `zara/environment/`, `chris/environment/` (exact versions and builds, with the full package lists) |
| Trained weights | Hugging Face, pinned by commit with SHA256 per file: `zara/diii-d/weights_manifest.json`, `data/weights_manifest.json`, `zara/mast/r1/predict/r1_files.json`; downloaded by `steps/4_shipped_weights.sh` |
| Training entrypoint | `bash prepare.sh A` (CPU preprocessing, no GPU), then `bash train.sh A` (GPU); the same with `B` |
| Inference entrypoint | `bash predict.sh A OUT_DIR` or `bash predict.sh B OUT_DIR` (add `--rebuilt` for the retrained models) |
| Data provenance, per machine | `SUBMISSION_A.md` and `SUBMISSION_B.md`, section "Data provenance" |
| Code of the generated data the package downloads | `zara/datagen/` (Zara's FreeGSNKE sets), `chris/mast/gsolve/` (gsolve sets), `zara/mast/r1/cpu_caches/` (r1's training caches) |
| Training compute declaration | A 20.9 and B 17.84 GPU-hours on one A100-SXM4-80GB, each measured twice on two GCP machines a week apart (A 20.9 and 20.77, B 17.84 and 17.68): `SUBMISSION_A.md` and `SUBMISSION_B.md`, section "Compute" |

## How close a rebuild should be

The organisers check two things against the Final labels: files from the shipped weights within 1e-4 of the submitted S, and
files from retrained models within 0.01, on each machine. Teams do not have the Final labels, so neither check can be run
before hand-over. What we can do is compare rebuilt files with the submitted ones, and score retrained models on data that has
labels. The table sums up that evidence; it is the strongest available to us without the Final labels.

| | Shipped weights (1e-4) | Retrained (0.01) |
|---|---|---|
| A, DIII-D | Byte-identical on RTX PRO 6000. On an A100, differences in the last digits; estimated change in S about 1e-5 | Label-free bound on the change in S on the Final shots: 1.5e-3 (retrained on an A100), 1.6e-3 (on an RTX PRO 6000) |
| B, DIII-D | Byte-identical on RTX PRO 6000. On an A100, estimated change in S about 1e-5 | Label-free bound on the change in S on the Final shots: 1.9e-3 (retrained on an A100) |
| B, MAST | Byte-identical on RTX PRO 6000. On an A100, ψ R² 0.9999999995 against the submitted file; largest differences 0.003 in ψ, 0.12 in q95, 0.03 in βN; estimated change in S about 2e-5 on average (6e-5 one standard deviation above) | The whole blend retrained on an A100, against the shipped one, on 15 labelled synthetic MAST sets (11,360 cases): change in S −0.0033 to +0.0007 per set |

Every model of A and B was retrained from scratch on an A100 with this package, 30 October to 1 November
(`checks/a100-final/`). For DIII-D, a label-free bound on the Final shots settles the 0.01 check: it assumes every difference
moves the prediction away from the labels, and still comes out at 0.002 or less. For B's MAST file that bound is too loose
(0.05 to 0.09 even with only r1 retrained), so we measured the change in S directly on synthetic MAST data with labels: the
shipped and the retrained networks through the same `predict.sh B` and blend, on the held-out parts of all 15 synthetic
MAST sets we have, from two different solvers. Synthetic scores do not predict the Final score, but the change between the
shipped and the retrained blend stayed within a third of the tolerance on every set. Details: `SUBMISSION_A.md` and
`SUBMISSION_B.md`, sections "Checks" and "What to expect when B is retrained".

## Paths starting with `/fsx/`

Paths starting with `/fsx/` are our cluster's shared disk. They appear in records of what ran (copies of the generation
code checked by SHA256 against the originals, check results and run settings), in paths stored inside the shipped DIII-D
checkpoints, which `zara/diii-d/my_experiments/ml/final/infer_c1h.py` maps to `$FEQ_CACHE`, and as defaults that
`settings.sh` replaces. Steps 0 to 6 of `REPRODUCE.md` never use an `/fsx/` path: our A100 run of them, on a machine with no
`/fsx/`, logged none. `prepare.sh`, `train.sh` and `predict.sh` stop if a folder from `settings.sh` is not set. The
optional rebuilds of the gsolve sets and of Chris's synthetic caches run scripts that are records of our cluster and use
its folder names (`REPRODUCE.md`, "Optional: rebuild the generated data"). `zara/datagen/`, `chris/mast/gsolve/` and the
`checks/` folders are provenance and verification records, apart from `zara/diii-d/checks/compare_sub.py`, which
`steps/5_compare.sh` runs.

## Licence

MIT (`LICENSE`). Files copied from third parties keep their own licence: `THIRD_PARTY_NOTICES.md`.
