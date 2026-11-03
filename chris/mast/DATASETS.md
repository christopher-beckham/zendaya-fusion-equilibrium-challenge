# Datasets Chris's MAST models train on

Every model also trains on the official DIII-D training data (`diii_d_train`), read through Chris's feature cache. No MAST
labels or demo shots are used. All generated data is pinned in the Hugging Face dataset
`zenithon-zendaya/mast-synthetic-data` (`data/manifest.json`) and checked by SHA256 on download.

## Generated datasets

| Training cache | Made by | Why it was made | CPU only | Used by |
|---|---|---|---|---|
| `zara_v1_d3d` | Zara | Synthetic DIII-D set (500 cases) mixed into every run alongside the real DIII-D data | Yes | all 13 |
| `zara_g4_5000_limiter` | Zara (FreeGSNKE gen4), cached by Chris | Synthetic MAST training data built from the Development MAST inputs | Yes | Z, Z2, Z3 |
| `zara_g4_20000_limiter` | Zara (FreeGSNKE gen4), cached by Chris | Larger synthetic MAST set; its held-out part was also the checkpoint-selection set during training (`--synthS-metric g4_20000`) | Yes | inside all merged caches |
| `zara_g4_5000v4_limiter` | Zara (FreeGSNKE gen4, v4), cached by Chris | Revised synthetic MAST set | Yes | Z8 |
| `zara_g4_v6_limiter` | Zara (FreeGSNKE gen4, v6), cached by Chris | Revised synthetic MAST set | Yes | inside `mixv6` caches |
| `zara_g4_v7_limiter` | Zara (FreeGSNKE gen4, v7), cached by Chris | Revised synthetic MAST set | Yes | V7C |
| `gsolve_cond1000_limiter` | Chris (gsolve, `tune2` recipe) | Plasma shape and pressure settings per frame predicted from the inputs by a model taught on DIII-D (`cond_model.pkl`), so the synthetic MAST equilibria vary like real ones | Yes | Z17, Z18 |
| `gsolve_cond3_1000_limiter` | Chris (gsolve, `tune3` recipe) | As cond1000, plus a plasma-height rule and a selection critic taught on DIII-D (`sel_critic_v1.pkl`) that prefers the solver variants whose shape looks most like a real equilibrium with the same inputs | Yes | FC, inside FA's cache |

Merged caches (rebuilt by `data/fetch.py` with `synth_merge.py`, CPU only, about 2 minutes for all five):

| Merged cache | Base caches | Used by |
|---|---|---|
| `zara_g4_20k5k_limiter` | 20000 + 5000 | Z6b |
| `zara_g4_20k5kv4_limiter` | 20000 + 5000 + 5000v4 | Z7, Z9qb, Z18 (1 run) |
| `zara_g4_mixv6_limiter` | 20000 + 5000 + 5000v4 + v6 | Z15 |
| `zara_g4_20k5kv4cond_limiter` | 20000 + 5000 + 5000v4 + gsolve cond1000 | Z18 (2 runs) |
| `zara_g4_mixv6_gsolve_cond3_1000_limiter` | 20000 + 5000 + 5000v4 + v6 + gsolve cond3_1000 | FA |

## Zara's FreeGSNKE sets: hardware and time

Generation code, commands, seeds and the version of each file that ran: `zara/datagen/README.md`.

Reported by Zara Ercan. Every set ran on the CPU only: the FreeGSNKE environment (`zara/environment/freegsnke.yml`) has no GPU
libraries and the jobs requested no GPU. They ran on the AWS cluster's g7e nodes (g7e.48xlarge, Intel Xeon Platinum 8559C,
96 CPUs), with up to 93 single-threaded solver processes per node.

| Set | Training cache | Used by | Dates (October 2026) | Wall-clock | CPU-hours |
|---|---|---|---|---|---|
| 500 synthetic DIII-D cases | `zara_v1_d3d` | every model | 5-6 | 0.3 h | about 4.5 |
| MASTSYN, generations 1 and 3 | r1's `MASTSYN` records (`zara/mast/r1/`) | r1 | 6 | 3.8 h | about 210 |
| gen4 5000 | `zara_g4_5000_limiter`; r1's `MASTG4`, `MASTG4RT` | Z, Z2, Z3, Z15, FA, r1 | 8-9 | 20.6 h, including spot-node interruptions | about 465 |
| gen4 20000 | `zara_g4_20000_limiter` | Z15, FA, r1 | 12 | 9.0 h | 2,820 |
| gen4 5000v3 | r1's `MASTG4V3`, `MASTG4V3RT` | r1 | 12 | 2.6 h | about 484 |
| gen4 5000v4 | `zara_g4_5000v4_limiter` | Z8, Z15, FA | 13 | 1.5 h | about 235 |
| gen4 1500v6 | `zara_g4_v6_limiter` | Z15, FA | 15 | 0.5 h | 136 |
| gen4 1500v7 | `zara_g4_v7_limiter` | V7C | 16 | 1.1 h | 402 |
| **Total** | | | | **39.4 h** | **about 4,760** |

## Chris's gsolve sets: hardware and time

Generation code and commands: `gsolve/SOURCES.md`. Every step ran with the GPUs hidden (`CUDA_VISIBLE_DEVICES=`); the
solver runs on CPU threads. It ran on a shared cluster node alongside GPU training jobs, with a budget of at most 40 cores
for generation (`recipes/tune2/HANDOFF.md`, `recipes/TUNE3_BRIEF.md`). The node's name and CPU model were not logged.

| Step | cond1000 (16 October) | cond3_1000 (17 October) |
|---|---|---|
| Generate (`gen_param2.py`, 1,000 MAST input shots) | 50.6 min wall: 3 processes × 14 threads; 2,881–3,019 s each, 8,871 process-seconds | 76.9 min wall: 12 processes × 3 threads; 4,207–4,601 s each, 51,704 process-seconds |
| Convert (`convert_perslice.py`, 40 workers) | 1.3 min | 1.7 min |
| Build training cache (`synth_cache_seq.py` + `synth_shape_targets.py`, 28 workers) | 2.3 min | 2.2 min |
| **Total** | **54 min** | **81 min** |

Times come from `total …s` in each `gen_*.log` and the timestamps of `knobs.json`, the part files, `convert.log`,
`build.log` and the cache files. Rebuilding the training cache from the downloaded `gen4_layout` took 1.5 min for both
sets on 16 cores (`checks/rebuild_gsolve_caches.result.txt`).

## Fixed settings fitted before generation (CPU only, scikit-learn)

| File | Fitted by | Data |
|---|---|---|
| `gsolve/recipes/tune2/cond_model.pkl` (= `tune3/cond_model.pkl`) | `tune2/fit_prior.py` | DIII-D training data |
| `gsolve/recipes/tune3/sel_critic_v1.pkl` | `tune3/sel_critic.py` | DIII-D training data: real EFIT shapes against gsolve's shapes for the same DIII-D frames |

Fitting times were not logged.
