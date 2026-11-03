# r1: prediction

r1 is Zara Ercan's MAST model: the average of four networks (`rt_ft`, `rt_mix`, `rt_ftw3`, `rt_nonoise`), each with a per-frame
inboard field correction chosen by the selector `rt_ft`, then fixed post-processing. Training: `../README.md` and
`../scripts/train_r1.sh`. This folder regenerates r1's member file of the MAST blend, `OUT_DIR/r1/mast_private_test.npz`.

| File | Role |
|---|---|
| `predict_r1.sh OUT_DIR RUNS_DIR` | entrypoint: input cache and first-pass cache (CPU), then `r1_chain.sh` |
| `r1_chain.sh` | selector at 11 corrections, per-frame selection, 4 members, average, post-processing (pp2, centre-column guard), submission-format file |
| `phase_report.py`, `select_b.py`, `pp_guard.py`, `write_mast.py` | the chain's CPU steps |
| `fetch_r1.py`, `r1_files.json`, `r1_final_firstpass.json` | shipped weights and inference data: `zenithon-zendaya/zara-r1-weights` @ `5ca03a54` (the four members trained on the CPU-built caches), SHA256 per file |
| `more_data_code/mast_machine.py` | MAST machine description read by the post-processing |
| `../starter/my_experiments/` | the model code (`ch2/g4/fz`, shared with training) and the helpers the chain imports (`ch2/scal`, `ch2/syn/post`, `ch2/g4/lbfit`, `ch2/g4/diag`, `c1`, `c2`) |
| `../starter/fusion_scoring/`, `../starter/data_fixes.py` | the organisers' starter-kit functions the post-processing uses |

Data is read under `R1_ROOT` (`../README.md`):

| Path under `R1_ROOT` | Content | From |
|---|---|---|
| `cache/pegs/machines/MAST_3daee94ae0a6.npz`, `cells_9402e6db0bbf.npy` | MAST machine matrices (coil Green's functions on the grid) | `fetch_r1.py` |
| `cache/mast/mast_public_test/basis.npz` | MAST grid basis | `fetch_r1.py` |
| `cache/ch2/g4/caches/mast_private_iv_full_wdiv/` | first-pass cache of the Final inputs (the one the submitted file used) | `fetch_r1.py` (`r1_final_firstpass.json`); `predict_r1.sh` builds it on the CPU when it is absent (about 2 h on 96 cores) |
| `fusion-equilibrium-challenge-starter/challenge_data/data/mast_public_test/` | official MAST public-test inputs (one file is read) | `../link_data.sh` |

Environments: `zara/environment/fusion-eq-gpu.yml` (`R1_PY`) and `zara/environment/fusion-equilibrium-starter.yml`
(`R1_PY_STARTER`).
