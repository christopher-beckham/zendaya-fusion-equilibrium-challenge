# Generation code of Zara's FreeGSNKE datasets

Rule 6 allows generated data in a reproduction package provided the code that generates it is included. This folder holds
that code for the eight datasets Zara Ercan generated with FreeGSNKE and that the package downloads (list, Hugging Face
locations and hardware: `chris/mast/DATASETS.md`, section "Zara's FreeGSNKE sets"). It is a record of what ran. Nothing here
needs to be run to reproduce the submissions: the package downloads the generated data, pinned by SHA256.

| Set | Folder | Training data in the package | Generator |
|---|---|---|---|
| 500 synthetic DIII-D cases | `zara_v1_d3d/` | Chris's cache `zara_v1_d3d` | `gen_d3d.py` |
| MASTSYN generation 1 | `mastsyn_gen1/` | r1's `cache/m2/multi/MASTSYN/src` (`mastsyn_*`) | `gen_m2.py` |
| MASTSYN generation 3 | `mastsyn_gen3/` | r1's `cache/m2/multi/MASTSYN/src` (`mastsyn3_*`) | `gen_m2.py` |
| gen4 5000 | `gen4_5000/` | `zara_g4_5000_limiter`; r1's `MASTG4`, `MASTG4RT` | `gen4.py` + `gen4_set.py` (gen4.8) |
| gen4 20000 | `gen4_20000/` | `zara_g4_20000_limiter`; r1's `MASTG4`, `MASTG4RT` | gen4.8 |
| gen4 5000v3 | `gen4_5000v3/` | r1's `MASTG4V3`, `MASTG4V3RT` | gen4.9 |
| gen4 5000v4 | `gen4_5000v4/` | `zara_g4_5000v4_limiter` | gen4.10 |
| gen4 1500v6 | `gen4_1500v6/` | `zara_g4_v6_limiter` | gen4.11 |
| gen4 1500v7 | `gen4_1500v7/` | `zara_g4_v7_limiter` | gen4.11 |

Shared code is in `shared/`; each set's section below names the exact files it used.

## Layout

| Path | Contents |
|---|---|
| `<set>/COMMANDS.txt` | the commands as they ran, with dates, Slurm job numbers, node and worker counts |
| `<set>/*.py`, `*.sbatch` | the generator files in the version each step used (file names carry the step or job) |
| `<set>/code_snapshot/` (gen4) | the `code/` folder of the set as it is on disk: `run_big.sbatch` copies `gen4.py`, `gen4_set.py` and `pool.py` into it when each generation job starts |
| `<set>/as_run/` | versions rebuilt from the change log where the file on disk is not the one that ran (see "How the versions were established") |
| `<set>/set_files/` | the set's own README, caveats, assembly summary and code checksums, copied from the set folder |
| `<set>/history/`, `shared/*/history/` | every recorded version of a generator file, and each change as it was applied (`patches/`) |
| `shared/generated_code/` | `/fsx/zara.ercan/more_data/generated/code`: machine descriptions (`mast_machine.py`, `d3d_machine.py`), labels and synthetic Thomson (`synth_common.py`), the MAST input pools (`gen_mast.py`, `gen_mast_v2.py`), quality control and release (`relabel_qc.py`, `qc_boundary.py`, `audit_datasets.py`, `make_release.py`) |
| `shared/generated_code/as_run/` | `gen_mast.py` as it built `mast_input_pool.npz`; `synth_common.py` as it was during the DIII-D solves |
| `shared/starter/data_fixes.py` | the starter kit's data-errata module, imported by `gen_d3d.py` (MIT, Sophelio) |
| `shared/m2/` | `convert_m2.py` (MASTSYN raw points to r1 records) and its CPU launcher |
| `shared/gen4/` | the seed-library builder, the release checks (`verify_set.py`, `check.py`, `export_meta.py`), the method description `README_SET500v2.md` |
| `shared/r1_convert/` | `convert_g4.py` (gen4 sets to r1 records), both versions that ran, and `fix_ipmax.py` |
| `provenance_checks.json` | every version check below, the SHA256 of each copied file's original, of the input pools, seed library and sampling plans, and the cache-against-set comparisons |
| `SHA256SUMS.txt` | SHA256 of every file in this folder |

Paths inside the code are the cluster's (`/fsx/zara.ercan/...`), as they ran. Environment: conda env `freegsnke`
(FreeGSNKE 3.1.1), specified in `zara/environment/freegsnke.yml` of the package; the conversion steps use `fusion-eq-gpu`.
Every generator ran on CPUs only, single-threaded processes, no GPU requested.

## How the versions were established

1. **Session logs.** The commands that wrote, patched and launched every generator are in the logs of the coding assistant
   (Claude Code) kept on the cluster under `~/.claude/projects/`. Each file's history was replayed from its creation
   through every recorded change (full writes, edits and patch scripts, applied in time order). For `gen_d3d.py`,
   `synth_common.py`, `mast_machine.py`, `d3d_machine.py`, `relabel_qc.py`, `qc_boundary.py`, `make_release.py`,
   `gen_m2.py`, `convert_m2.py` and `convert_g4.py` the replay ends byte-identical to the file on disk
   (`provenance_checks.json`, `checks`). That makes the intermediate versions trustworthy, and the log timestamps place
   each change before or after each run.
2. **Job start times against change times.** `sbatch` copies a job script when it is submitted, and a Python process keeps
   the code it imported at start (the gen4 worker pool forks, so restarted gen4 workers keep it too). The version that ran
   is therefore the last version before the job or process started. Start and end times come from Slurm `.out` files,
   worker logs and output-file times (job accounting before 22 October is no longer available).
3. **Copies made at run time.** For gen4, `run_big.sbatch` copies the generator into `<set>/code/` at every job start.
4. **Output checksums.** Each of Chris's training caches records the SHA256 of every source file it was built from. All of
   them equal the set files as released: 500 of 500 (`zara_v1_d3d`, both the release copy and the generator's own output),
   5,490 of 5,490 (gen4 5000), 22,560 of 22,560 (gen4 20000), 5,640 of 5,640 (5000v4), 1,700 of 1,700 (v6, v7).
5. **Frozen copy.** The private repository `zaraercan/fusion-eq-freeze-2026-10-21` (tag `freeze-2026-10-21`) holds
   byte-identical copies of the generator files on disk; `provenance_checks.json` (`in_freeze_repo`) lists the paths.

The starter repository's git history does not help here: `my_experiments/` is excluded by its `.gitignore`.

Two of the copied READMEs, `shared/generated_code/RELEASE_README.md` and `zara_v1_d3d/generated_workspace_README.md`, link
to a private web page (a viewer of the generated sets) that only Zara can open. They are kept byte-identical to their
originals (`SHA256SUMS.txt`). The page is not needed to rebuild or check any set.

## Rules checks on this code

- **No credentials.** Every file here was searched for Hugging Face tokens (`hf_...`), Weights & Biases keys and `.netrc`
  entries: none found. Nothing was redacted.
- **No MAST labels.** No code or settings file here refers to FAIR-MAST or MAST EFIT output. The MAST generators read
  only the Development MAST inputs (`mast_public_test`), and those files carry no label columns at all: their only
  `efit_*` columns are `efit_times` and `efit_grid_R/Z` (checked on the files). All labels in the synthetic sets are
  computed from FreeGSNKE solutions.
- **Demo shot, schema only.** `shared/generated_code/gen_mast.py`, which `gen_m2.py` imports for its solver class and input
  pool, holds the path of the starter kit's MAST demo shot (`parquet_data/mast_shot_28348.parquet`). It is read only in
  `gen_mast.assemble()`, to copy the column schema, when building Zara's 500-case synthetic MAST test set. MASTSYN never
  calls it, and no gen4 file refers to it.
- **DIII-D labels** (training data, allowed): `gen_d3d.py --validate` compares FreeGSNKE flux with the real DIII-D EFIT flux
  on 280 held-out frames, and the v7 li target comes from Chris's model fitted on DIII-D training labels.

## zara_v1_d3d (500 synthetic DIII-D cases)

- **Source:** `/fsx/zara.ercan/more_data/generated/code/gen_d3d.py`, with `d3d_machine.py`, `synth_common.py`,
  `relabel_qc.py`, `qc_boundary.py`, `make_release.py` from the same folder and the starter kit's `data_fixes.py`. Output
  `/fsx/zara.ercan/more_data/generated/d3d_synthetic/parquet`, released to
  `/fsx/zara.ercan/more_data/synthetic_test_sets/d3d/parquet`.
- **Command:** `zara_v1_d3d/COMMANDS.txt`. 14 processes on the login node, 5 October 22:58 to 23:10 UTC, a re-run of 294
  points on 6 October 09:07-09:09, label re-computation 09:19-09:24, final assembly 09:28-09:34.
- **Settings and seeds:** 4,096 scrambled-Sobol points (seed 2028) over 15 dimensions (`design_summary.json`), keep the 500
  accepted points with the lowest Sobol index; tolerance 1e-5, 30 s per point; outcomes 1,694 ok, 656 limited,
  1,746 not converged.
- **Inputs:** official `diii_d_train` inputs of the 704 held-out shots of Zara's C1 validation split (10 %, seed 0), pooled
  into `cache/d3d_input_pool.npz` (159,537 frames, inputs only); DIII-D coil geometry from the same files.
- **Versions:** `gen_d3d_generation_2026-10-05T2258.py` solved 3,802 of the 4,096 points;
  `gen_d3d_rerun_2026-10-06T0906.py` (reversed-current fix) re-solved the 294 points drawn from reversed-current frames;
  `gen_d3d_final_assembly_2026-10-06T0928.py` (= `gen_d3d.py`, the file on disk) wrote the 500 files. The raw solves used
  `shared/generated_code/as_run/synth_common_for_d3d_solves_2026-10-05.py`; the label re-computation used the current
  `synth_common.py`, `relabel_qc.py` and `qc_boundary.py`.
- **Conversion:** Chris's `chris/mast/multimachine/my_experiments/nn/synth_cache.py` (in the package) built the cache
  `zara_v1_d3d` from the release folder on 7 October. The cache's `manifest.json` records the source folder, the settings
  and all 500 source SHA256, which equal the released files.
- **Gaps:** the run was not a Slurm job, so its times come from log files only. Chris's exact `synth_cache.py` command line
  is not recorded; the manifest gives its source and settings.

## MASTSYN generation 1

- **Source:** `my_experiments/m2/gen/gen_m2.py` (imports `gen_mast.py`, `mast_machine.py`, `synth_common.py` from
  `shared/generated_code/`, all unchanged since before this run), launcher `gen_workers.sbatch`. Raw output
  `/fsx/zara.ercan/more_data/m2/mast_train/raw/wNN/point_XXXXXX.npz`.
- **Command:** `mastsyn_gen1/COMMANDS.txt`. Jobs 3600 (64 workers, 6 October 11:08-11:18), 3630 (56 workers, 11:18-11:27) and
  3690 (56 workers, 11:28-13:29), all on g7e8-spot-1.
- **Settings and seeds:** Sobol seed 31001, the first 131,072 points; shot-level split by SHA1 hash of the shot name, 10 % of
  shots kept out as 'val'; solenoid factor 0.5; tolerance 1e-5; 40 s per point.
- **Inputs:** `mast_input_pool.npz` (62,932 frames of the 1,206 `mast_public_test` shots, inputs only; built 5 October by
  `shared/generated_code/as_run/gen_mast_for_pool_2026-10-05T1511.py`) and `mast_input_pool_thomson.npz` (real Thomson
  levels, built by `gen_mast_v2.py`); the MAST machine description from `mast_machine.py`.
- **Versions:** `gen_m2_job3600_2026-10-06T1108.py`, `gen_m2_job3630_2026-10-06T1118.py` and
  `gen_m2_job3690_start_2026-10-06T1128.py`, with the launchers `gen_workers_job*.sbatch`. While 3690 ran, its watchdog
  restarted stalled workers, and restarted workers loaded the file as it was then. The changes after 11:08 (`history/`)
  add hang handling, diagnostics and command-line options whose defaults keep the generation unchanged; the solver,
  design, ranges and seeds are the same in every version from 11:08 on. `gen_m2_2026-10-06T1333_current.py` is the file
  on disk.
- **Conversion:** `shared/m2/convert_m2.py` (job 3692, about 13:30), two records per equilibrium (synthetic Thomson with
  realism noise, and a hybrid with the frame's real Thomson profile) into `cache/m2/multi/MASTSYN/src`.
- **Gaps:** 3,778 design points were written twice, once by the 64-worker job 3600 and again by the 56-worker jobs (a
  point's file path depends on the worker count), giving 133,779 point files for 130,001 distinct points; about 1,071
  points were never written because 3690 was stopped. 828 of the duplicated points converged both times, and their
  records share a file name, so 62,448 records were converted and 60,792 remain. Which of the two identical-input solves
  each surviving record came from is not recorded. Per-point times were not logged.

## MASTSYN generation 3

- **Source:** `gen_m2.py` as on disk (last changed 6 October 13:33, before this run) = `mastsyn_gen3/gen_m2.py`; launcher
  `gen_workers2_job3843_2026-10-06T1808.sbatch` (the file on disk minus a worker-relaunch guard added at 19:41, after the
  job ended). Raw output `/fsx/zara.ercan/more_data/m2/mast_train3/raw`.
- **Command:** `mastsyn_gen3/COMMANDS.txt`. Job 3843, 56 workers on g7e8-spot-1, 6 October 18:08-19:33.
- **Settings and seeds:** Sobol seed 780, first 65,536 points, vertical-coil jitter 0.99-1.01, axis-pressure factor 0.9-1.6,
  solenoid factor 0.5; 16,178 converged.
- **Inputs:** as generation 1.
- **Conversion:** `shared/m2/convert_m2.py --raw-name mast_train3 --prefix mastsyn3 --copies 2` (job 3846), 32,356 records.
- **Gaps:** none for the code. Per-point times were not logged.

## gen4 sets: common to all six

- **Source:** `my_experiments/m2/gen4/gen4.py` (solver, plasma and Thomson model), `gen4_set.py` (sampling plan, workers,
  assembly), `pool.py` (worker watchdog), launcher `slurm/run_big.sbatch`, assembly `slurm/assemble_set.sbatch`
  (`assemble_v6.sbatch`, `assemble_v7.sbatch` for v6 and v7). They import `mast_machine.py` and `synth_common.py` from
  `shared/generated_code/` (unchanged since 6 October). Output `/fsx/zara.ercan/more_data/m2/gen4_set<name>/`.
- **Inputs:** `mast_input_pool.npz`, the `mast_public_test` input files, the MAST machine description, and the seed library
  `cache/m2/gen4_seedlib.npz` (6,000 MASTSYN gen-1/gen-3 solutions used as starting guesses; builder and command in
  `shared/gen4/`). From v4 on, also the machine variant with a hidden vertical-field coil (`code_snapshot/machine_MAST_challenge_bz/`,
  written by `gen4.py` itself on first use from the MAST machine description).
  The sampling plan `candidates.json` of each set is built by `gen4_set.py --build-candidates` from these inputs and the
  arguments; SHA256 of each in `provenance_checks.json`.
- **Version evidence:** the `code/` copies made at job start, and the change log showing whether the generator changed
  between the first and the last job start of a set. Release checks after assembly: `shared/gen4/`.
- **Conversion to Chris's caches:** `chris/mast/multimachine/my_experiments/nn/synth_cache_seq.py` and
  `synth_shape_targets.py` (in the package), from `<name>_train` and `<name>_heldout`; checked by SHA256 (above).
- **Conversion to r1's records:** `shared/r1_convert/` (gen4 5000, 20000 and 5000v3 only).

## gen4 5000 (gen4.8, recipe v2)

- **Command:** `gen4_5000/COMMANDS.txt`. Jobs 6273-6275 (8 October 13:07), relaunched as 6938, 6939 (9 October 08:31),
  assembly 6940.
- **Settings and seeds:** 5,000 + 640 cases requested, 15 strata (Ip quintile x dIp/dt tercile) at the real share, per-case
  seed 100,000 + candidate id. One ramp-up stratum ran out of frames: 4,874 + 616 cases.
- **Versions:** `code_snapshot/` was copied by the relaunch. Three edits to `gen4_set.py` just before it (9 October
  08:30, case bookkeeping after the watchdog loop; no change to the solver or the sampling) are undone in
  `as_run/gen4_set_2026-10-08T1307_first_launch.py`, the version of the first launch. Cross-check: the same file results
  from the 500v2 smoke-test copy plus the two edits recorded before the launch. `gen4.py` and `pool.py` did not change
  between the two launches. `as_run/run_big_before_2026-10-12.sbatch` is the launcher of both launches.
- **Gaps:** the spot nodes of 6273/6274 were reclaimed overnight, so their logs are truncated.

## gen4 20000 (gen4.8, recipe v2)

- **Command:** `gen4_20000/COMMANDS.txt`. Jobs 7130 (12 October 09:14) and 7132-7134 (09:42), assembly 7135 (18:12-18:24).
  Jobs 7124-7127 on gpu2 never started.
- **Settings and seeds:** 20,000 + 2,560 cases, up to 6 realisations per real frame (`--n-real 6`), per-case seed
  1,000,000 + candidate id.
- **Versions:** `code_snapshot/` (copied 09:42:32; no change to the generator between 09:14 and 09:42). The assembly ran
  with `gen4_set.py` as edited at 11:30 for recipe v3 (`as_run/gen4_set_assembly_2026-10-12T1812.py`, identical to
  `gen4_5000v3/code_snapshot/gen4_set.py`). That edit had moved the "at least 3 labelled frames" rule under recipe v3 only,
  so this set includes 5,802 training and 654 held-out cases with 1 or 2 labelled frames. The files in use are the
  output of this assembly.

## gen4 5000v3 (gen4.9, recipe v3)

- **Command:** `gen4_5000v3/COMMANDS.txt`. Jobs 7144, 7145 (12 October 11:43-14:17), assembly 7146.
- **Settings and seeds:** 5,000 + 640 cases, `--n-real 2`, per-case seed 3,000,000 + candidate id; inner-wall-limited
  plasmas allowed, up-down coil jitter, Thomson structure variations.
- **Versions:** `code_snapshot/` (single launch). r1 uses `mast_g4s5000v3_train`, not the relabelled `v3b` copy.

## gen4 5000v4 (gen4.10, recipe v4)

- **Command:** `gen4_5000v4/COMMANDS.txt`. Jobs 8290-8292 (13 October 12:40-14:09), assembly 8293.
- **Settings and seeds:** 5,000 + 640 cases, `--n-real 2`, per-case seed 6,000,000 + candidate id; hidden vertical field,
  wide coil jitter, coil blends of two shots, wide current profiles, 80 profile groups.
- **Versions:** `code_snapshot/gen4.py`, `gen4_set.py`, `pool.py` (copied 12:41:24, last job start); the other files there
  were added at release. Chris's cache uses `mast_g4s5000v4_*`, not the relabelled `v4b` copy.

## gen4 1500v6 (gen4.11, recipe v6)

- **Command:** `gen4_1500v6/COMMANDS.txt`. Jobs 9017-9019 (15 October 15:31-16:03), assembly 9028.
- **Settings and seeds:** 1,500 + 200 cases, `--n-real 2`, per-case seed 12,000,000 + candidate id; one jitter factor per
  upper/lower coil pair, P4 factor in 3 equal-quota bins (45 strata), TF jitter 0.3 %, hidden field 0-20 mT.
- **Versions:** `code_snapshot/` was refreshed by hand at release (16:03:58); the last generator edits were at 15:30-15:31,
  before the submission at 15:31:21, and none followed, so the snapshot is the version that ran.

## gen4 1500v7 (gen4.11, recipe v7)

- **Command:** `gen4_1500v7/COMMANDS.txt`. `li_table.py` (16 October 13:09), jobs 9339-9341 (13:37) and 9361, 9362 (14:03),
  stopped at 14:41; assembly with `assemble_v7.sbatch` (14:42).
- **Settings and seeds:** v6 plus a per-frame li target; 1,500 + 200 cases, `--n-real 2`, per-case seed 13,000,000 +
  candidate id; hidden field -10 to +20 mT.
- **Inputs:** as v6, plus `li_targets.npz` from `li_table.py`, which applies Chris's `cond_model.pkl` (gradient-boosted
  trees fitted on DIII-D training labels) to input features of the real MAST frames (`mast/real_z.parquet`, built by
  Chris's `mast_z.py` / `zfeat.py` from `mast_public_test` inputs). The SHA256 recorded in `li_targets.json` for
  `cond_model.pkl`, `cond_model.json`, `zfeat.py`, `mast_z.py`, `fit_prior.py`, `gen_param2.py` and `cond_prior.py` equal
  the copies in `chris/mast/gsolve/recipes/tune2/`.
- **Versions:** `code_snapshot/` copied at 14:08:34 by the last two jobs; at 13:40 the copy made by the first three jobs
  had the same SHA256, and `SHA256SUMS_code.txt` (14:44) lists the same values.
- **Gaps:** `real_z.parquet` is not in the package (inputs only, rebuilt by `mast_z.py`); `li_targets.npz` is not copied
  here (9 MB, derived from inputs; SHA256 in `provenance_checks.json`).

## Other gaps

- `gen_mast.py`: replaying its log gives the file on disk except one line, `N_POINTS` 2048 against 4096, changed outside
  the logged sessions. `N_POINTS` sizes only Zara's synthetic MAST test set; `gen_m2.py` and the input-pool builder do not
  use it, and the pool builder's version is a complete file from the log.
- `gen_mast_v2.py` (builds `mast_input_pool_thomson.npz`) has no recorded history; its last change (5 October 15:47:14)
  is 13 s before the pool build, so the file on disk is the version that ran.
- Not included, by design: the input pools, the seed library, the sampling plans and `li_targets.npz` (derived from the
  official data or from the generators' own output; checksums recorded), and the raw and released data (on Hugging Face).
