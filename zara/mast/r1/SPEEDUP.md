# Faster training for Zara's r1 and u4 MAST models

Workspace: `/fsx/chris.beckham/zara_speedup` (git). Branch `master` holds the exact changes, branch `graphs` (worktree
`graphs_wt/`) adds the CUDA graph. Zara's own tree is untouched: every run here mounts `/fsx/zara.ercan` read-only.
Measurements on one NVIDIA RTX PRO 6000 Blackwell (Xeon Platinum 8559C host), 25 October 2026.

## Why the original code is slow

Most of each training step goes to finding the plasma boundary. For each of the 5 equilibrium solves per step, the code
tries 18 flux levels by bisection and, for each level, grows the plasma region from the magnetic axis one cell at a
time (`GSLayer.flood`, about 40 small GPU operations per level, plus a check that makes the CPU wait for the GPU).
That is about 24,500 tiny GPU launches and 626 CPU waits per step. The GPU sits idle while the CPU issues them, so the
step time depends on the CPU's speed: 2.17 times slower on the GCP A100 machine (slow 2.2 GHz CPU) than on the RTX.

## The changes

| Change | Where | Effect on results |
|---|---|---|
| **Reach level.** One pass computes, for every cell, the highest flux level at which it is still connected to the axis. Each bisection step then needs one comparison per frame instead of a full region growth, and the plasma mask is one comparison. The bisection itself is unchanged. | `physics.py` `reach_level`, `boundary` | identical (bit for bit) |
| Axis fallback without a CPU wait | `physics.py` `axis` | identical |
| Warm-start bounds tensor built once | `model.py` `forward` | identical |
| Loss logging summed on the GPU in float64, read back only when printed | `train.py` | identical printed values |
| **CUDA graph** (branch `graphs`). The equilibrium solve of each training batch (forward and backward) is recorded once per machine and replayed with one launch. Inside the graph the reach-level pass always runs its full 128 steps, which gives the same answer as stopping when it converges. `FZ_GRAPH_PICARD=0` turns it off. | `physics.py` `picard_graphed`, `model.py` `forward` | identical forward values; gradients within the original code's own run-to-run spread |

The same edits apply to the three code folders behind r1 and u4 (`scripts/apply_speedups.py`):
`ch2/g4/fz` (r1's five fine-tunes), `ch2/syn/fz` (r1's base model `h_muon_mo`) and `upegs` (u4).
Prediction and evaluation also use the reach-level pass; they do not use the graph.

## Checks

**Bit-for-bit checks during real training** (`scripts/check_boundary_wrap.py`): every boundary search and axis search
runs both Zara's original code and the new code on the same inputs and compares every output.

| Code folder (recipe) | Equilibria compared | Differences |
|---|---|---|
| `ch2/g4/fz` (`rt_ft`), three seeds | 815,000 (including 2,225 solves with the inner wall active) | 0 |
| `ch2/syn/fz` (`h_muon_mo`) | 299,000 | 0 |
| `upegs` (`u4_all`) | 277,000 | 0 |

**Graph against normal solve** (`scripts/check_graph_wrap.py`), 100 training solves per folder: forward outputs
identical in every solve. Gradients differ from the normal solve by at most 5.6e-7 (relative); the normal solve run twice
on the same inputs differs from itself by up to 4.8e-7, because GPU additions in the backward pass of indexing operations
happen in a varying order. This is also why two runs of Zara's unchanged code never give identical weights.

**Against Zara's numbers.** Full 10-epoch `rt_ft` retrains with her recorded arguments, scored with her internal
validation (`ch2/g4/slurm/eval_rt.sh`: 3,816 held-out FreeGSNKE cases whose inputs are the challenge's own
`mast_public_test` inputs, with FreeGSNKE labels and no MAST label (`SUBMISSION_B.md`, section "r1 (Zara Ercan's MAST
model)"); the competition's scoring code with a MAST limiter mask drawn from machine geometry instead of the EFIT-derived
envelope; raw member; `scripts/eval_rt_member.sh`). Rescoring her stored predictions with this pipeline gives her recorded 0.834306
exactly.

| `rt_ft` run | Time per epoch | 10 epochs | Internal validation S | DIII-D held-out R²ψ (epoch 10) |
|---|---|---|---|---|
| Zara's recorded run | 334 s | 59 min | 0.8343 | 0.99713 |
| Her unchanged code, rerun | 331-349 s | 59 min | 0.8303 | 0.99793 |
| Exact changes | 109 s | 21 min | 0.8349 | 0.99733 |
| Exact changes + CUDA graph | 69 s | 14.5 min | 0.8337 | 0.99735 |

Her unchanged code lands 0.004 from her own recorded score, which is the normal spread of this recipe. Both faster
versions land within 0.0006 of it, far inside the organisers' 0.01.

## Speed

**RTX PRO 6000, per epoch**

| Recipe | Zara's code | Exact changes | + CUDA graph |
|---|---|---|---|
| `rt_ft` (r1 fine-tunes) | 334 s | 109 s | 69 s |
| `h_muon_mo` (r1 base) | about 320 s | 110 s | 73 s |
| `u4_all` (u4) | about 325 s | 105 s | 67 s |

**Per training step** (`scripts/profile_rt_ft.py`, `rt_ft`): kernel launches 24,465 → 7,411 → 2,934; CPU waits
626 → 30 → 2; GPU busy 27% → 24% → 56%. Peak GPU memory 42.5 GB in every case.

**Whole models on the RTX** (epochs as recorded; data loading about 6 minutes for `h_muon_mo`, 1-2 minutes otherwise)

| Model | Zara's code | Exact changes | + CUDA graph |
|---|---|---|---|
| r1 (6 runs, 74 epochs) | 7.3 h | about 2.5 h | about 1.7 h |
| u4 (1 run, 28 epochs) | 2.6 h | about 0.85 h | about 0.55 h |

**A100 estimate (not measured).** On the GCP A100 machine Zara's code took 0.47 s per step for 24,400 launches, about
19 µs of CPU per launch. With launch counts carried over (they do not depend on the machine):

- exact changes: about 0.14-0.15 s per step, still limited by the CPU: about 3 times faster than Zara's code on the A100;
- with the CUDA graph: CPU about 0.06 s per step; the GPU work (0.034 s on the RTX) is probably 1.4-3 times slower on the
  A100, so about 0.06-0.10 s per step: about 5-8 times faster than Zara's code on the A100.

| Model, A100 hours | Zara's code (2.17 × RTX) | Exact changes | + CUDA graph |
|---|---|---|---|
| r1 | about 15.9 | about 5.2 | about 2.4-3.6 |
| u4 | about 5.6 | about 1.9 | about 0.8-1.3 |

One short run on an A100 (2 epochs of `rt_ft`, graph version) would replace these estimates with a measurement.

These are the whole GPU cost of r1 and u4: generating their synthetic training data uses no GPU (Zara, 25 October).

## r1 retrained end to end

The whole r1 chain retrained with the CUDA-graph version (`scripts/r1_chain.sh`), each run replaying Zara's recorded
`args.json`: the base model `h_muon_mo` from scratch, then `w3_ft` and the four members from it ("full chain"). The
four fine-tunes were also retrained from Zara's own base model ("from Zara's base"), to separate the base's run-to-run
variation. Scored with her internal validation (raw, 3,816 cases); r1 here is the plain average of the four members
(her `ens_rt4`).

| Model | Zara | Full chain | From Zara's base |
|---|---|---|---|
| `rt_ft` | 0.8343 | 0.8354 | 0.8337 |
| `rt_mix` | 0.8335 | 0.8372 | 0.8339 |
| `rt_ftw3` | 0.8330 | 0.8382 | 0.8362 |
| `rt_nonoise` | 0.8286 | 0.8289 | 0.8288 |
| **r1 (average of the four)** | **0.8387** | **0.8405** | **0.8387** |

Every model lands within 0.0052 of Zara's score (the organisers allow 0.01 on the real score), and the r1 average within
0.0018. The base model's last-epoch held-out numbers match hers (DIII-D R²ψ 0.99770 against 0.99775, synthetic MAST
R²ψ 0.9943 against 0.9944).

Training time on the RTX PRO 6000: base 40 min (Zara's: 2.7 h), fine-tunes 10-15 min each; the whole chain 1.8
GPU-hours against her 7.3.

The production r1 file adds prediction-time steps on top of this average (a per-frame field correction chosen with
`rt_ft`, post-processing and a guard against boundaries crossing the centre column; `final/zara_mast/r1/r1_chain.sh`).
Those steps are prediction only and were not rerun here.

## Not done yet

- u4 has not been retrained end to end with the new code; its code passes the bit-for-bit and graph checks and was
  timed for one epoch.
- No A100 measurement of the new code.
- The CUDA graph uses standard PyTorch (`torch.cuda.make_graphed_callables`), no Triton or compiler. It has not been run
  on the A100 machine's older driver (535).
- These are changes to Zara's code: she needs to review and adopt them before they go into the reproduction package.

## Files

- `scripts/apply_speedups.py FOLDER [--graph]`: applies the edits to a copy of a code folder.
- `scripts/run.sh`, `scripts/run_args.sh`: run a recipe (from its recorded `args.json`) with Zara's data read-only.
- `scripts/check_boundary_wrap.py`, `scripts/check_graph_wrap.py`: the checks above.
- `scripts/eval_rt_member.sh`: Zara's internal validation for one model.
- `scripts/profile_rt_ft.py`, `scripts/launch_split.py`: launch and wait counts per step.
- `out/`: logs, checkpoints and scores of every run above (16 GB, mostly prediction files under `out/eval/`).
