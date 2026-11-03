# Reproducing submissions A and B

Team zenithon-zendaya (Christopher Beckham, Zara Ercan), Fusion Equilibrium Challenge Final.

This guide takes an empty GPU machine to two results:

1. **The submitted files, regenerated from the shipped weights.** Expected: byte-identical on NVIDIA RTX PRO 6000 GPUs, on
   which the submitted files were made. On another GPU type the last digits differ: on an A100 we estimate the change in S at
   about 1e-5 for the DIII-D files and 2e-5 for B's MAST file (6e-5 one standard deviation above), against the 1e-4 allowed (step 4).
2. **Every model retrained from scratch, then used to predict again.** Expected: within the organisers' tolerance of 0.01 in S.

Each step says what it does, why it is needed, how to run it, and how long it took us. The steps are shell scripts in
`steps/` and the two entrypoints `train.sh` and `predict.sh` at the top of the repository. Every script is short, and its
header lists the commands it runs, so each step can also be run by hand.

## The two submissions

| | Submission A | Submission B |
|---|---|---|
| Entry | FW-EDF58354, Challenge 1 | FW-41B8A13C, Challenge 2 |
| Predictions repository | `zenithon-zendaya/fec-final-sub1` @ `807b5104` | `zenithon-zendaya/fec-final-sub5` @ `3a2a2a26` |
| DIII-D file | recipe **v13** (Zara Ercan) | recipe **Ch1-min** (Zara Ercan) |
| MAST file | none | blend of **8 of Chris Beckham's models** and **Zara Ercan's r1** |
| GPU training, one A100 80 GB | 20.9 h | 17.8 h (Ch1-min 3.2 h, Chris's models 11.0 h, r1 3.7 h) |
| Details | `SUBMISSION_A.md` | `SUBMISSION_B.md` |

`SUBMISSION_A.md` and `SUBMISSION_B.md` describe the models and hold the compute declaration and the data-provenance
statement for each machine (rule 6). This guide covers how to run everything.

## What you need

| Item | Detail |
|---|---|
| GPU | one NVIDIA A100 80 GB (any 80 GB card works; the code caps itself at 80 GB) |
| CPU and memory | 12 or more cores, about 128 GB of RAM with 12 to 16 CPU workers. Memory grows with the number of workers (`NCPU` in `settings.sh`, default every core): `predict.sh A` peaked at about 125 GB with 16 workers; with 6 workers, every step stayed under 50 GB |
| Disk | about 750 GB free for both submissions and both checks (step 0); our A100 run peaked at about 650 GB |
| Software | Linux, an NVIDIA driver for CUDA 12.8, `git`, and Miniconda or Miniforge. Everything else is installed in step 1 |
| Network | access to github.com, huggingface.co, conda-forge and the PyTorch package index |
| Hugging Face token | not needed: every repository the package downloads from is public (Hugging Face may print a notice suggesting a token; it can be ignored) |

A and B can be run on the same machine, one after the other, or on two machines at the same time. On one machine, do not
start the training of A and B at the same time: both build the same DIII-D caches first.

## Step 0: two copies of the code, one per check

**What.** Two clones of the repository at the commit you were given: one for the shipped-weights check (step 4) and one for
retraining (steps 5 and 6), each with its own work folder. The software environments and the official data are shared.

**Why.** The shipped DIII-D networks are downloaded to the same paths where training writes its own networks, and training
skips any network that is already there. In one copy, training after the shipped-weights check would retrain nothing. The
scripts refuse to mix the two: `steps/4_shipped_weights.sh` marks its copy, and `prepare.sh`, `train.sh` and
`predict.sh --rebuilt` stop in a marked copy (and the other way round).

**How.** Every later step reads its folders and interpreters from one file, `settings.sh`. Set `WORK` (the copy's own work
folder), `ENVS` and `DATA` (shared), then source it, in every new shell:

```bash
BIG=/path/to/large/disk
for c in shipped retrain; do
  git clone https://github.com/christopher-beckham/zendaya-fusion-equilibrium-challenge.git $BIG/zendaya-$c
  git -C $BIG/zendaya-$c checkout <commit>
done

# a shell for the shipped-weights check (step 4)
cd $BIG/zendaya-shipped && export WORK=$BIG/work-shipped ENVS=$BIG/envs DATA=$BIG/data && source settings.sh

# a shell for retraining (steps 5 and 6)
cd $BIG/zendaya-retrain && export WORK=$BIG/work-retrain ENVS=$BIG/envs DATA=$BIG/data && source settings.sh
```

Each step below says which copy it runs in. Disk for both submissions and both copies: about 750 GB (650 GB measured).

## Step 1: build the software environments

**What.** Three conda environments, each pinned to exact package versions and builds:

| Environment | Used for |
|---|---|
| `fusion-eq-gpu` | DIII-D networks, r1 (Zara Ercan) |
| `fusion-equilibrium-starter` | the challenge starter kit's packages: DIII-D input caches, the organisers' validator, downloads |
| `fusion-eq` | Chris Beckham's MAST models |

**Why.** These are the environments that trained and ran the submitted models. Each is checked against its recorded package
list after it is built.

**How.** Once, in either copy (the environments are shared):

```bash
bash steps/1_environments.sh
```

**Expect.** About 15 minutes. Three lines ending in `identical to ...pip-freeze.txt`, then the torch version (2.11.0+cu128)
and the name of the GPU.

## Step 2: download the official challenge data

**What.** The organisers' Development dataset (DIII-D training shots, DIII-D and MAST public test inputs) and Final dataset
(DIII-D and MAST Final inputs), at pinned revisions.

**Why.** Submission A trains on the DIII-D training shots only. Submission B also uses the MAST public test inputs (never
labels) and predicts the MAST Final shots.

**How.** In both copies. The data is downloaded once into the shared `$DATA`; in the second copy the script checks the files
again and makes that copy's links.

```bash
bash steps/2_official_data.sh A          # DIII-D splits, about 93 GB
bash steps/2_official_data.sh B          # adds the MAST splits, about 100 GB in all
```

**Expect.** 20 minutes to 2 hours, depending on the connection (about 110 GB). One line per split with every file matching the SHA256 that Hugging Face publishes
(7,041 training shots, 874 and 879 DIII-D test shots, and for B the MAST test splits), then the links the code reads through.

## Step 3 (B only): MAST training data and Final inputs

**What.** Four parts, all on the CPU:

- a. Synthetic MAST equilibria that Chris's models train on (14.8 GB), with their checksums.
- b. Derived caches for Chris's models, built from the official data (coil pictures, feature tables, linear-decoder
  matrices).
- c. r1's training data (about 45 GB): Zara's synthetic MAST records and DIII-D inputs, and the first-pass caches r1 trains
  on, which were built on the CPU. Zara's original GPU-built caches in the same bundle are skipped, as is any pre-trained
  network: r1 starts from random weights.
- d. The inputs of the 1,205 Final MAST shots in the form Chris's models read.

**Why.** Chris's models and r1 train on synthetic MAST equilibria. These were generated from MAST inputs only (no MAST
labels) on CPUs; `SUBMISSION_B.md` gives the generators and their hardware and run time. The generated data is downloaded
because generating it again takes days of CPU time. "Optional: rebuild the generated data" below explains how to rebuild it.

**How.** In both copies; the shipped-weights copy does not need part c (r1's training data):

```bash
PARTS="a b d" bash steps/3_mast_data.sh  # shipped-weights copy
bash steps/3_mast_data.sh                # retraining copy
```

**Expect.** About 1 to 1.5 hours in the retraining copy, most of it parts b and c; about 30 minutes in the other.

## Step 4: regenerate the submitted files from the shipped weights

**What.** Download the trained weights that made the submitted files, run each submission's inference entrypoint on the Final
inputs, and compare with the submitted files.

**Why.** This is the first check: the shipped weights must reproduce the submitted score. On NVIDIA RTX PRO 6000 GPUs the
files are byte-identical; on other GPUs, see "Expect".

**How.** In the shipped-weights copy:

```bash
bash steps/4_shipped_weights.sh A        # DIII-D weights, 24.9 GB (B uses the same download)
bash steps/4_shipped_weights.sh B        # adds Chris's MAST checkpoints (2.9 GB), r1's networks and r1's Final-input cache (0.7 GB)

FEQ_CLEAN=1 bash predict.sh A $WORK/out/A_shipped
FEQ_CLEAN=1 bash predict.sh B $WORK/out/B_shipped

bash steps/5_compare.sh A $WORK/out/A_shipped
bash steps/5_compare.sh B $WORK/out/B_shipped
```

`FEQ_CLEAN=1` deletes each DIII-D network's intermediate predictions once the file is written (about 70 GB for v13).

Before its networks run, r1 solves an equilibrium for every frame of the Final MAST inputs (its "first-pass cache"). This is
CPU work on the inputs alone, with no learned parameters, and it takes about 2 hours on 96 CPU cores. Step 4 therefore
downloads the cache the submitted file was predicted with. To build it yourself instead, delete
`$R1_ROOT/cache/ch2/g4/caches/mast_private_iv_full_wdiv` before running `predict.sh B`. A build on the same CPU type (Intel Xeon
Platinum 8559C) with the default 8 shards of 12 threads was byte-identical. The solve's last bits depend on the CPU, on the
thread count (`R1_FP_THREADS`) and on how shots are grouped into shards. A build with 4 threads per shard matched 2 of the 1,205
files byte for byte; through r1 and the blend it moved `mast_private_test.npz` by at most 0.003 in ψ and 0.07 in βN (q95 did not
change), and the label-free bound on the change in S was 2.5e-4, above the 1e-4 allowed for the shipped weights. Use the
downloaded cache for the shipped-weights check.

**Expect.** About 2 hours per submission on one GPU with 16 CPU cores. Most of it is CPU work, so it takes longer with fewer
cores: 7 to 8 hours on an A100 machine whose 24 vCPUs were shared with two trainings. `predict.sh` runs the organisers' validator on each file and prints its
SHA256. `5_compare.sh` downloads the submitted files and compares:

| File | SHA256 of the submitted file |
|---|---|
| A: `diii_d_private_test.npz` | `3af1ebe0ee31cef2c0cd827434850e3a9fc5063b42fc2ebb2c873f73d9c1be27` |
| B: `diii_d_private_test.npz` | `86e6a3726ff4d7a050bf7560a2d6df19c49a4ca1ce817f704da7492ac0acc153` |
| B: `mast_private_test.npz` | `3300302e7c69325554ee182b103bc6defeec07495d284180c333d07963f13fb3` |

On NVIDIA RTX PRO 6000 GPUs, the GPU type the submitted files were made on, all three files were byte-identical in every run.
On another GPU type the networks' forward passes round differently in the last digits, so no file is byte-identical, and
`5_compare.sh` shows the size of the differences instead. On an A100 (`checks/a100-final/`):

| File | Largest difference ψ / q95 / βN | ψ R² against the submitted file | Estimated change in S |
|---|---|---|---|
| A: `diii_d_private_test.npz` | 0.011 / 0.032 / 0.013 | 0.99999994 | 1e-5 |
| B: `diii_d_private_test.npz` | 0.031 / 0.025 / 0.009 | 0.99999992 | 1e-5 |
| B: `mast_private_test.npz` | 0.003 / 0.115 / 0.033 | 0.9999999995 | 2e-5 (6e-5 one standard deviation above) |

The DIII-D estimates scale the change in S measured on labelled shots predicted twice by the size of these differences. No
labelled MAST data exists for that test, so the MAST estimate assumes the A100's rounding differences are unrelated to the
errors against the labels (`checks/a100-final/README.md`). A strict bound that assumes every difference moves the prediction
away from the labels gives 4.7e-4 and 5.7e-4 for the two DIII-D files and 1.5e-3 for the MAST file.

## Step 5: retrain every model from scratch

**What.** Two entrypoints, one after the other. `prepare.sh` does the CPU preprocessing, with no GPU visible. `train.sh`
then retrains every network of the submission from random weights, one after another, on one GPU.

| Submission | `prepare.sh` (CPU only) | `train.sh` (one GPU) |
|---|---|---|
| A | DIII-D preprocessing of v13: input caches, label-side targets, Ridge base model (about 5.5 to 6 h on 12 cores) | the 52 networks of v13: **20.9 GPU-hours** on an A100 (measured twice: 20.9 and 20.77) |
| B | DIII-D preprocessing of Ch1-min (about 4.8 h on 12 cores) | the 12 networks of Ch1-min (3.2 h), the 15 training runs behind Chris's 8 models (11.0 h), the 6 training runs of r1 (3.4 to 3.7 h): **17.8 GPU-hours** on an A100 (measured twice: 17.84 and 17.68) |

**Why.** This is the second check: the retrained models must score within 0.01 of the submitted S. Only GPU work counts
toward the 24 GPU-hours, so the CPU preprocessing has its own entrypoint. It can run on a machine without a GPU, as long as
`train.sh` then reads the same `$FEQ_CACHE`.

**How.** In the retraining copy:

```bash
bash prepare.sh A 2>&1 | tee $WORK/prepare_A.log
bash train.sh A   2>&1 | tee $WORK/train_A.log

bash prepare.sh B 2>&1 | tee $WORK/prepare_B.log
bash train.sh B   2>&1 | tee $WORK/train_B.log
```

Each step is skipped when its output already exists, so an interrupted run continues where it stopped when the same command
is run again.

**Expect.** At the end, `train.sh` prints the time of each part on this GPU. The DIII-D steps are also logged one per line,
marked CPU or GPU, in `$FEQ_CACHE/train_times.tsv` (A) and `$FEQ_CACHE/train_times_ch1min.tsv` (B). In `train.sh A`, the
CPU follow-up of the 5 out-of-fold networks (about 2.5 h) runs in the background while the GPU trains networks that do not read
it, so it adds no GPU time. On our A100 run (30 October to 1 November), `train.sh A` took 20 h 46 min and `train.sh B` 17 h 41
min, wall-clock (`checks/a100-final/README.md`).

## Step 6: predict with the retrained models

**What.** The same inference entrypoint as step 4, reading the networks that step 5 trained.

**How.** In the retraining copy:

```bash
FEQ_CLEAN=1 bash predict.sh A $WORK/out/A_rebuilt --rebuilt

$R1_PY zara/mast/r1/predict/fetch_r1.py --firstpass-only     # r1's first-pass cache of the Final MAST inputs (B only)
FEQ_CLEAN=1 bash predict.sh B $WORK/out/B_rebuilt --rebuilt

bash steps/5_compare.sh A $WORK/out/A_rebuilt
bash steps/5_compare.sh B $WORK/out/B_rebuilt
```

The `fetch_r1.py --firstpass-only` line downloads the same r1 input data as step 4 (579 MB, checked by SHA256), and no
networks. This data is computed from the Final MAST inputs alone, so it is the same for the shipped and the retrained
networks, and with it the difference from the submitted file comes from retraining alone. Without it, `predict.sh B`
builds it on the CPU first (about 2 hours on 96 CPU cores, about 190 core-hours in all; step 4 gives the small differences
this makes).

**Expect.** Retrained networks are not bit-identical to the shipped ones, so these files differ slightly from the submitted
ones. `5_compare.sh` reports the pooled ψ R² against the submitted file and the largest difference per quantity. Our A100
retrain (`checks/a100-final/`) gave:

| File | ψ R² against the submitted file | Largest difference ψ / q95 / βN |
|---|---|---|
| A: `diii_d_private_test.npz` | 0.9999988 | 0.018 / 0.37 / 0.059 |
| B: `diii_d_private_test.npz` | 0.9999981 | 0.026 / 0.42 / 0.096 |
| B: `mast_private_test.npz` | 0.99980 | 0.24 / 6.7 / 1.40 |

A retrain on another machine gives differences of the same size, not the same values. The organisers score with the Final
labels, which we do not have. Our evidence on the change in S: a label-free bound of at most 0.002 for both DIII-D files,
and a change of −0.0033 to +0.0007 for B's MAST blend on 15 labelled synthetic MAST sets (`SUBMISSION_B.md`, section "What
to expect when B is retrained"). On our A100 run, `predict.sh --rebuilt` took 3 h 15 min for A, and about 5 h for B.

## Optional: rebuild the generated data

The downloads in step 3 are generated data. Each can be rebuilt from this repository on CPUs only:

| Data | Rebuild | CPU time we measured |
|---|---|---|
| r1's first-pass training caches | `bash zara/mast/r1/cpu_caches/build_r1_caches.sh` after `python zara/mast/r1/fetch_r1_data.py --root $R1_ROOT --no-cpu-caches` (`zara/mast/r1/cpu_caches/README.md`) | about 6 h wall-clock on our CPU nodes, about 47 process-hours |
| Chris's gsolve sets (`cond1000`, `cond3_1000`) | the solver and recipes in `chris/mast/gsolve/` (`chris/mast/DATASETS.md`). The recipe scripts are records of the runs on our cluster and use its folder names; to run them elsewhere, change those folders | `chris/mast/DATASETS.md` |
| Chris's training caches of the synthetic sets | `python data/fetch.py --gsolve-raw`, then the cache builders (`chris/mast/DATASETS.md`), which are also records of the runs on our cluster and use its folder names | minutes |

Rebuilt r1 caches match the downloaded ones byte for byte only on the CPU type that built them (Intel Xeon Platinum 8559C).
On another CPU the boundary search can land a few frames on slightly different solutions; on an AMD EPYC test node no frame's
elongation moved by more than 0.02 (`zara/mast/r1/cpu_caches/README.md`).

## Where things are

| Path | What it holds |
|---|---|
| `settings.sh`, `steps/`, `train.sh`, `predict.sh` | everything this guide runs |
| `SUBMISSION_A.md`, `SUBMISSION_B.md` | the models, compute declaration, data provenance, files and checks of each submission |
| `data/` | download scripts and manifests (repository, revision, SHA256 per file) for the official data, Chris's synthetic data and Chris's weights |
| `zara/diii-d/` | Zara's DIII-D package: recipes v13 (A) and Ch1-min (B), weights manifest (`README.md`) |
| `zara/mast/r1/` | Zara's r1 MAST model: training, prediction, data manifest, CPU cache builder (`README.md`) |
| `chris/mast/` | Chris's MAST models: code, training records of every run, model definitions, checks (`README.md`) |
| `blend/` | the fixed weights of B's MAST blend, the script that applies them, and the code that chose them from Development-board scores |
| `zara/environment/`, `chris/environment/` | the pinned environment files |

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `source settings.sh first` | a new shell: set `WORK`, `ENVS`, `DATA` and run `source settings.sh` (step 0) |
| "This copy ... holds the shipped weights" or "... holds retrained networks" | a step run in the wrong copy; see step 0 |
| `401` or `Repository Not Found` from Hugging Face | a repository or revision the package does not use: every repository it downloads from is public; check the command against `REPRODUCE.md` |
| a Hugging Face download error (for example `CAS Client Error ... error decoding response body`), followed by tar's `This does not look like a tar archive` | the download dropped. `settings.sh` sets `HF_HUB_DISABLE_XET=1`, which avoids the Xet transfer that caused this in our test; check it is set (`echo $HF_HUB_DISABLE_XET`), then run the same step again. Most steps reuse files already downloaded after their SHA256 check; step 3 part c (r1's training data) unpacks its bundle as it downloads, so it starts again from the first part |
| `conda env create` stops at a terms-of-service prompt | run the step through `steps/1_environments.sh`, which sets the conda-forge channel variables |
| an environment `differs from ...pip-freeze.txt` | a package came from another channel; remove the environment folder under `$ENVS` and run step 1 again |
| `FileNotFoundError` under `$WORK/r1_data` | step 3 part c has not run, or stopped early (run it again: it checks every file) |
| `official data not linked` | step 2 was run for A only; run `bash steps/2_official_data.sh B` |
| out of memory in a CPU step | set `NCPU` lower before `source settings.sh` (default: every core; the DIII-D preprocessing needs about 8 GB per worker) |
