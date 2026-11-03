# r1: training caches built on the CPU

## What these caches are

r1 (Zara Ercan's MAST model) does not train on the raw synthetic equilibria directly. Before training, every training frame
goes once through the model's own equilibrium solve, and the result is stored: the plasma boundary of that first pass, the
features derived from it, and for DIII-D a few auxiliary targets. These stored results are the "first-pass caches". r1 reads
ten of them:

| Cache (under `$R1_ROOT/cache/`) | Files | Used by |
|---|---:|---|
| `ch2/syn/mastsyn_synth_p7` | 93,148 | the base network `h_muon_mo` (synthetic MAST records MASTSYN) |
| `ch2/g4/caches/mastg4_synth_wd`, `mastg4rt_synth_wd` | 24,874 each | the fine-tunes (synthetic MAST records MASTG4, MASTG4RT) |
| `ch2/g4/caches/mastg4v3_synth_wd`, `mastg4v3rt_synth_wd` | 5,000 each | the fine-tunes (MASTG4V3, MASTG4V3RT) |
| `ch2/g4/caches/mast_test_iv_full_wdiv` | 1,206 | only to build the fine-tune caches: time features of the real public-test shots (inputs only) |
| `pegs/diiid_train_s2p7`, `pegs/diiid_train_p7` | 6,337 and 704 | every run (DIII-D training shots, every 2nd frame; holdout shots) |
| `upegs/aux_d3d_s2p7`, `upegs/aux_d3d_p7` | 6,337 and 704 | every run (DIII-D auxiliary targets) |

## Why the CPU

Zara built these caches on a GPU during the competition (about 19 GPU-hours, which would count toward the 24-hour training
limit). Chris Beckham rebuilt all of them with no GPU visible on 26-27 October 2026, using the faster boundary search described
in `../SPEEDUP.md` (bit-identical to Zara's search), and retrained r1 from them. Those retrained networks are the shipped r1
weights (`../predict/r1_files.json`). On Zara's internal validation set they score S 0.8404, against 0.8387 for the networks
trained on her GPU-built caches. Building the caches on the CPU makes them data preparation, which rule 6 asks us to declare
with its hardware and wall-clock time (below); it uses none of the 24 GPU-hours.

## How to get them

Either download the caches that the shipped weights were trained on (`python zara/mast/r1/fetch_r1_data.py --root $R1_ROOT`,
the default), or build them yourself:

```bash
python zara/mast/r1/fetch_r1_data.py --root $R1_ROOT --no-cpu-caches     # Zara's data bundle only
bash zara/mast/r1/link_data.sh $DATA                                      # official data for r1
bash zara/diii-d/link_data.sh $DATA                                       # official data for the DIII-D package
R1_ROOT=$R1_ROOT R1_PY=<fusion-eq python> R1_PY_STARTER=<fusion-equilibrium-starter python> \
  bash zara/mast/r1/cpu_caches/build_r1_caches.sh
```

`build_r1_caches.sh` runs every step with no GPU visible, writes straight into `$R1_ROOT`, and keeps a log per step and shard
in `$R1_ROOT/cache_build_logs/` together with the wall-clock time of each step (`times.tsv`). It stops if one of the ten
caches is already there and was not built by it (a downloaded copy, or Zara's GPU-built one): delete that folder first. If it
is interrupted, run it again and it continues. It is sized for a 48-core machine (about 7 hours). The last bits of each
solve depend on the CPU type, the number of threads per process and how frames are grouped into processes, so the defaults
(the processes and threads that built the shipped caches) are the ones that reproduce them; on the Final inputs, 4 threads
instead of 12 matched 2 of 1,205 files (`SUBMISSION_B.md`, job 11386). `R1_THREADS` changes the threads per process for a
machine with fewer cores, at the cost of byte-identity. `R1_SMOKE_SHOTS=12` runs the whole chain on 12 shots per cache as a
quick test.

Its steps:

0. Inputs that Zara's data bundle does not hold, all derived from the official data with code in this repository: the DIII-D
   per-shot input and Thomson caches of every training shot (`zara/diii-d/my_experiments/c1/nn/preprocess.py` and
   `preprocess_px.py`), their reference boundaries (`ch2/g4/fz/build_refs.py`) and EFIT profile labels (`xg/labels.py`; DIII-D
   labels are allowed), and the MAST public-test input cache (`c2/preprocess_mast.py`, inputs only). The bundle holds only the
   files r1's training opens; files already present are kept.
1. Base model's cache: `m2/upegs_fz/build_cache.py --machine MASTSYN --split synth --p-share 0.7 --tag _p7`, 6 processes of
   8 threads. It is written to `cache/m2/mastsyn_synth_p7` and moved to `cache/ch2/syn/mastsyn_synth_p7`, where the base
   network reads it.
2. Fine-tune caches: `ch2/g4/fz/build_cache.py --machine <M> --split synth --sol-factor 0.5 --p-share 0.7 --tag _wd` for MASTG4,
   MASTG4RT, MASTG4V3, MASTG4V3RT, with `INTR_WALL=0.20 INTR_FP=divert`, 6 processes of 8 threads each.
3. Real-MAST public-test cache: the same script with `--machine MAST --split test --tag _iv_full_wdiv`.
4. `ch2/g4/fix_time.py train <M>`: copies the time-history features of each fine-tune frame from the real shot it was cut from
   (step 3).
5. DIII-D first passes: `pegs/build_cache.py --machine DIII-D --split train --subset train --stride 2 --p-share 0.7 --tag _s2p7`
   (10 processes of 4 threads) and `--subset holdout --p-share 0.7 --tag _p7` (4 of 12).
6. DIII-D auxiliary targets: `ch2/g4/fz/build_aux.py --tag _s2p7` (6 of 8) and `--tag _p7 --subset holdout` (1 of 16).

## Hardware and time of the build behind the shipped weights

AWS cluster, nodes of type g7e: Intel Xeon Platinum 8559C, 96 vCPUs and about 1.9 TB of memory per node; each job had 48 CPUs
and no GPU visible. 26 October 19:21 UTC to 27 October 01:17 UTC, about 6 hours of wall-clock time, with the DIII-D training
first pass on a second node at the same time as the fine-tune caches. Times from the job logs (`/usr/bin/time` per process):

| Step | Processes | Wall-clock | Process-hours | CPU core-hours |
|---|---:|---:|---:|---:|
| 1. base cache | 6 | 0.63 h | 3.5 | 26.8 |
| 2. fine-tune MASTG4 | 6 | 0.76 h | 4.5 | 35.0 |
| 2. fine-tune MASTG4RT | 6 | 1.38 h | 8.1 | 55.5 |
| 2. fine-tune MASTG4V3 | 6 | 0.28 h | 1.5 | 11.5 |
| 2. fine-tune MASTG4V3RT | 6 | 0.27 h | 1.5 | 11.4 |
| 3. real-MAST test cache | 6 | 0.56 h | 3.3 | 25.8 |
| 4. time features | 1 | 0.16 h | 0.2 | about 0.2 |
| 5. DIII-D `_s2p7` | 10 | 2.24 h | 21.6 | 75.3 |
| 5. DIII-D `_p7` | 4 | 0.59 h | 2.3 | 23.0 |
| 6. auxiliary targets | 7 | 0.12 h | about 0.5 | not logged |
| **Total** | | **about 7.0 h back to back (6 h as run, two nodes)** | **about 47** | **about 265** |

The DIII-D `_s2p7` row is the run that finished the cache: an earlier start on the other node had built 472 of its 6,337
shots (7 %) before it was stopped, and the finishing run skipped those. A rebuild in one run therefore cannot match the
shipped `_s2p7` cache byte for byte: in a full rebuild, 592 of its 6,337 files and 578 of the matching auxiliary-target files
differ, and every other cache is identical (`SUBMISSION_B.md`, "Checks"). `pegs/build_cache.py` solves a shard's shots in
groups of 8, taken in order from the shots still to do, and the group sets the last bits of its solves; skipping the 472
moved later shots of each shard into other groups. Of the 592: 299 are among the 472 shots built on the other node, and 293
among the 4,139 shots the finishing run solved in a different group of 8 than a run in one go would; none of the 1,726 shots
it solved in the same group differs. The 578 auxiliary-target files split the same way (292 and 286, none in the same
group): each belongs to a shot whose first pass differs (14 such shots have identical auxiliary targets). Step 0 did not
run then (its inputs were already on our disk); from the checks below it takes about 20 minutes with 16 workers.

## Checks

On one node of a different type (AMD EPYC 7R13, 16 cores), one small shard of every step, run with the code in this
repository from an `R1_ROOT` laid out as above, against two references: the same step run with the code that built the
shipped caches (`zara_speedup` at `9a940ff`, with its output redirect), and the shipped caches themselves.

| Step | Shard | Against the as-run code, same machine | Against the shipped cache (built on Intel Xeon 8559C) |
|---|---|---|---|
| 1. base | first 48 shots of shard 0 of 6 | 48 of 48 files byte-identical | not byte-identical; no frame moves by more than 0.02 in elongation; median largest relative feature difference 4e-4 |
| 2. MASTG4, MASTG4RT | first 48 shots of shard 0 of 6 | 48 of 48 byte-identical, each | (after step 4) the 8 time-feature columns identical in 48 of 48 files |
| 3. real-MAST test | first 16 shots of shard 0 of 6 | 16 of 16 byte-identical | 1 frame in 883 moves by more than 0.02 in elongation |
| 5. DIII-D `_s2p7`, `_p7` | first 16 and 8 shots of shard 0 | 16 of 16 and 8 of 8 byte-identical | 2-4 % of frames have a feature off by more than 1 % of its spread, under 1 % by more than 10 % |
| 6. auxiliary targets | 16 and 8 shots, from the shipped DIII-D caches | | largest difference 2e-9 |
| 0. inputs | 3 holdout shots missing from the bundle; 6 MAST test shots | | DIII-D input, Thomson and reference caches byte-identical; labels equal to float32 rounding (relative 1e-7); MAST input cache byte-identical |

So the scripts and the copied code reproduce the computation that built the shipped caches exactly; on a different CPU the
equilibrium solve differs in the last bits, and a few frames converge to a slightly different boundary. The same happens when
the organisers rebuild the caches on their machine. `compare_caches.py NEW REF` makes these comparisons (byte for byte, then
array by array).

## Sources of the copied code

| In this repository | Copied from | Changes |
|---|---|---|
| `starter/my_experiments/m2/upegs_fz/` (`build_cache.py`, `data.py`, `embed.py`, `machine.py`, `physics.py`, `shots.py`) | Chris's workspace `zara_speedup` at `9a940ff` (Zara's `m2/upegs_fz` copied unchanged in `ad1bac0`, speed-ups in `0aa5894`) | `/fsx/zara.ercan` became the setting `R1_ROOT` |
| `starter/my_experiments/pegs/` (same six files) | `zara_speedup` at `9a940ff` (Zara's `pegs` copied unchanged in `21fae1c`, speed-ups in `93cd0de`) | `R1_ROOT` |
| `starter/my_experiments/ch2/g4/fix_time.py` | Zara's `fusion-equilibrium-challenge-starter/my_experiments/ch2/g4/fix_time.py` (SHA256 `eb928924…`, 13 October) | `R1_ROOT`; imports `embed` from this package's `ch2/g4/fz` (identical to Zara's) |
| `starter/my_experiments/xg/labels.py` | Zara's `my_experiments/xg/labels.py` (SHA256 `2f8c87e1…`, 5 October) | `R1_ROOT` |
| `starter/local_score.py`, `starter/fusion_scoring/metrics.py`, `starter/fusion_scoring/masks/d3d_envelope.npz` | the organisers' starter kit, as in `zara/diii-d/` (identical to Zara's copies) | none |

The fine-tune, test and auxiliary steps use `starter/my_experiments/ch2/g4/fz/`, which r1's training also uses. It differs from
the copy that built the shipped caches only in the CUDA-graph speed-up of training (inactive on the CPU) and in file paths.

## `as_run/`

The scripts that actually ran on our cluster (copied from `zara_speedup/scripts` at `9a940ff`, unchanged). They read and write
fixed paths in Chris's workspace and use Linux mount namespaces (`unshare -rm`) to put the new caches over Zara's cache
folders. They are kept as the record of what ran; `build_r1_caches.sh` replaces them.
