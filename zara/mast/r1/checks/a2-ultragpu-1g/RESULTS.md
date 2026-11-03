# r1 training time on one A100 80 GB (GCP a2-ultragpu-1g)

**r1 trains in 13,155 s = 3.65 h on one A100 80 GB**, 2.03 times the 6,474 s (1.80 h) of the RTX PRO 6000 Blackwell
reference, with the same code (an earlier version of this package's r1 code; `train.py` and the run arguments are unchanged
since, the other changes are file paths). All 6 runs exited 0. Every epoch takes 2.0 to 2.1 times as
long as on the RTX. Against Zara's original code on an A100 (734 s per `rt_ft` epoch), an epoch is 5.3 times faster.

One quality question, unrelated to timing: the last epoch of `rt_nonoise` ends with DIII-D held-out R2psi **-3.12**
(Zara: 0.99784), while its synthetic-MAST validation is unchanged. Real MAST predictions are not affected:
the spike is one DIII-D frame, in a part of the network r1 does not use on MAST (`RT_NONOISE_SPIKE_RESULTS.md`).

## Machine

| | |
|---|---|
| Instance | GCP `a2-ultragpu-1g`, under KVM |
| GPU | 1 × NVIDIA A100-SXM4-80GB, 400 W, driver 535.216.01 (CUDA 12.2), not shared |
| CPU | Intel Xeon @ 2.20 GHz, 12 vCPUs (6 cores × 2 threads, 1 socket), 38.5 MB L3 |
| RAM | 167 GB |
| Disk | 500 GB GCP persistent disk (all data and outputs), 445 GB free at the start |
| Kernel | 6.8.0-1015-gcp |
| Software | torch 2.11.0+cu128 from `chris/environment/environment.yml` (works on the 535 driver) |

Folders: `DATA=/root/fec_data/official`, `DEST=/root/fec_data/zara`, `OUT=/root/fec_runs/{quick,full}`, bundle download
in `/root/fec_data/bundle_dl`.

## Steps

| Step | Outcome | Time |
|---|---|---|
| 1. Environment | OK: torch 2.11.0+cu128, CUDA available, A100 seen | 4.5 min |
| 2. Official data | OK: `diii_d_train` 7,041/7,041 files match (88 GB), `mast_public_test` 1,206/1,206 (3.1 GB) | 16.8 min |
| 3. Zara's bundle | OK after two fixes on the data side (below) | 4.5 min download, 2.8 min unpack, extra archive under 1 min |
| 4. Links under `/fsx` | OK after a path fix (below) | |
| 5. Quick timing | OK on the second attempt (below) | 1,102 s + 640 s |
| 6. Full chain | OK: 6 runs, all exit 0, 20:03:39 to 23:42:54 UTC | 13,155 s |

### Problems and fixes (no code was changed)

1. **Step 3, token.** The first token tried (a fine-grained token of `beckhamc`) returned `404 Repository Not Found` for
   `beckhamc/zara-r1-a100-bundle`. A second token from Chris with access to the dataset worked.
2. **Step 3, missing files in revision `4fc240c6`.** 90,416 of the 93,148 entries in
   `zara.ercan/cache/ch2/syn/mastsyn_synth_p7/` are symlinks to `/fsx/zara.ercan/cache/m2/mastsyn_synth_p7/`, which was not
   in the bundle. Step 5 then failed in `h_muon_mo` after 141 s:
   `FileNotFoundError: [Errno 2] No such file or directory: '/fsx/zara.ercan/cache/ch2/syn/mastsyn_synth_p7/mastsyn3_000000.npz'`
   (log: `logs/quick/h_muon_mo.attempt1_missing_files.log`). Chris added `zara_r1_bundle_extra.tar` in revision
   `e2109f88`: checksum OK, extracted on top of the six parts, 90,416 files, no broken links left.
3. **Step 4, path.** The handoff links `$DATA/development/$s`, but `fetch_official.py` puts each split in
   `$DATA/development/data/$s`. The links point at the latter. `sudo` is not installed; the commands ran as root without it.
4. **Step 5, separate `rt_ft` timing.** While the fix for problem 2 was prepared, 2 epochs of `rt_ft` were timed with the
   unchanged `train_r1.sh quick`, in a separate folder with a placeholder `h_muon_mo.log` (a line `exit 0`) so the script
   skipped the base model (logs: `logs/quick_rt_ft_only/`, placeholder not committed). The extra archive was downloaded
   and extracted during that run's data loading, before its first epoch. Epochs: 143 s, 141 s.

## Step 5: quick timing

| Run | Epoch 1 (graph capture) | Epoch 2 | RTX per epoch | A100 / RTX | Whole run |
|---|---|---|---|---|---|
| `h_muon_mo` (2 epochs, no warm-up) | 143 s | 139 s | 67-73 s | about 2.0 | 1,102 s |
| `rt_ft` (2 epochs) | 142 s | 140 s | 69 s | 2.03 | 640 s |
| `rt_ft`, separate run | 143 s | 141 s | 69 s | 2.04 | 651 s |

Epoch times are `time_s` in `hist.json`; the `== epoch` lines in the logs are 2-4 s longer.

## Step 6: full chain

| Run | Epochs | A100 per epoch, mean [range] | RTX per epoch | A100 whole run | RTX whole run | A100 / RTX (whole run) | Outside epochs (loading, set-up) |
|---|---|---|---|---|---|---|---|
| `h_muon_mo` | 28 (2 warm-up) | 140.7 s [139-143] after warm-up; warm-up 127, 124 s | 67-73 s | 4,779 s | 2,393 s | 2.00 | 870 s |
| `w3_ft` | 10 | 141.6 s [140-144] | about 70 s | 1,798 s | 858 s | 2.10 | 382 s |
| `rt_ftw3` | 6 | 142.6 s [142-144] | about 69 s | 1,237 s | 590 s | 2.10 | 381 s |
| `rt_ft` | 10 | 139.2 s [136-146] | 69 s | 1,781 s | 872 s | 2.04 | 389 s |
| `rt_mix` | 10 | 140.5 s [138-143] | about 70 s | 1,828 s | 892 s | 2.05 | 423 s |
| `rt_nonoise` | 10 | 137.0 s [134-138] | about 70 s | 1,732 s | 869 s | 1.99 | 362 s |
| **r1 total** | | | | **13,155 s = 3.65 h** | **6,474 s = 1.80 h** | **2.03** | 2,807 s |

- **Per epoch:** 2.0 to 2.1 times the RTX in every run. GPU utilisation during training is mostly 61-74%, at most 83%
  (`*.gpu.csv`, 5th to 95th percentile of the non-zero samples), which fits training that is still partly limited by the
  CPU; this machine has 12 vCPUs at 2.20 GHz.
- **Data loading:** about 14.5 min for `h_muon_mo` (RTX: about 6 min), with the GPU at 0% until training starts
  (`h_muon_mo.gpu.csv`, 20:03:39 to about 20:17:30 UTC); in step 5 the DIII-D data took 102 s of this and the 93,148 MASTSYN
  files most of the rest. About 6-7 min for each of the other runs (RTX: about 1 min). Loading is 21% of the A100 total.
- **Against Zara's original code:** `rt_ft` per epoch 734 s on an A100 (handoff) against 139 s here, 5.3 times faster.
  The whole chain with the original code on an A100 was not measured; 7.3 h × 2.17 gives about 15.8 h, so about 4.3
  times faster.
- **Competition limit:** 3.65 h of the 24 GPU-hours allowed.

## Peak GPU memory

| Run | `[capmem]` peak allocated / reserved | `nvidia-smi` peak (`*.gpu.csv`) |
|---|---|---|
| every run of steps 5 and 6 | 42.5 GB / 45.8-45.9 GB | 46,765-46,777 MiB |

The same as the RTX reference (42.5 GB / 45.3 GB reserved). No out-of-memory errors.

## Held-out results against Zara's recorded runs

| Run | Final R2psi, A100 | Final R2psi, Zara |
|---|---|---|
| `h_muon_mo` | 0.99793 | 0.99775 |
| `w3_ft` | 0.99805 | 0.99699 |
| `rt_ftw3` | 0.99762 | 0.99780 |
| `rt_ft` | 0.99596 | 0.99713 |
| `rt_mix` | 0.99799 | 0.99760 |
| `rt_nonoise` | **-3.12316** | 0.99784 |

`rt_nonoise` tracks Zara's run through epoch 8 (0.9966-0.9980 against her 0.9968-0.9978), then at epoch 9 (the last) the
held-out psi loss jumps from about 1 to 1,570 and R2psi falls to -3.12. The other held-out numbers of that epoch stay
normal (axis dR 16.6 mm, q95 0.948, βN kin 0.975), no batch was skipped, and the weights were finite (`train.py` checks
before saving), which points at a few validation frames with a very wrong psi rather than a broken network. Zara's runs
show the same kind of one-epoch dip: her `rt_ft` falls to R2psi 0.6964 at epoch 5 and her `rt_mix` to 0.9755 at epoch 1,
and both recover the next epoch. Here the dip landed on the last epoch, so nothing follows it.

The synthetic-MAST validation of epoch 9 is identical to epoch 8 (MASTG4RT 0.9886, MASTG4V3RT 0.9906), so the failure
shows on DIII-D only. `predict.py` loads `model.pt`, the last epoch, so an r1 built from this chain uses the epoch-9
model. Whether it is also wrong on some real MAST shots was not tested here: the MAST public-test input cache is not in
the bundle. A later check on these checkpoints (`RT_NONOISE_SPIKE.md`, results in `RT_NONOISE_SPIKE_RESULTS.md`) found
that the spike comes from one DIII-D frame, in a part of the network that r1 does not use on MAST: MAST predictions are
not affected.

## Files

- `logs/full/`: step 6, `<run>.log`, `<run>.gpu.csv` (GPU memory and utilisation every 10 s), `<run>/hist.json`, `driver.log`.
- `logs/quick/`: step 5 (second attempt), and the failed first attempt `h_muon_mo.attempt1_missing_files.*`.
- `logs/quick_rt_ft_only/`: the separate 2-epoch `rt_ft` timing.
- `RT_NONOISE_SPIKE.md`: handoff for checking whether the `rt_nonoise` spike reaches real MAST.
