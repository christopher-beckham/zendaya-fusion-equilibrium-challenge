# Synthetic test sets: generator workspace

This folder holds the code and intermediate files that produce the synthetic test sets. **The data to use, and its documentation, are in `/fsx/zara.ercan/more_data/synthetic_test_sets/`** (see its README).

## Contents

| Path | What it is |
|---|---|
| `code/` | Generators, quality control, audit, scoring and release scripts |
| `machines/` | FreeGSNKE machine descriptions built from the challenge geometry (MAST, DIII-D) and the SPARC base equilibria |
| `cache/` | Input pools drawn from the challenge data (MAST public-test inputs, DIII-D held-out shots) |
| `<set>_synthetic/raw/` | The outcome of every Sobol design point, including rejected ones |
| `<set>_synthetic/parquet/`, `*_500.npz` | The assembled sets (copied into the release folder) |
| `qc_logs/`, `<set>_synthetic/logs/` | Run logs |
| `audit_report.json` | The latest audit (copied into the release folder) |
| `explorer/` | Source of the explorer page (https://claude.ai/artifact/GCh8nzUPbPP183dgwWBMmz) |
| `sparc_public/` | The SPARCPublic g-files converted to a table; not part of the test sets |

## Code

| Script | Role |
|---|---|
| `mast_machine.py`, `d3d_machine.py` | Machine descriptions from the challenge coil geometry |
| `gen_mast.py`, `gen_d3d.py`, `gen_machine.py` (SPARC, ITER) | Sobol designs, solves, assembly |
| `synth_common.py` | Labels from a solved equilibrium, synthetic Thomson |
| `relabel_qc.py`, `qc_boundary.py` | Quality control: re-solve check, scorer boundaries, label recomputation |
| `audit_datasets.py` | The 49-check audit |
| `score_folder.py` | Competition-metric scoring on any folder |
| `make_release.py`, `RELEASE_README.md` | Builds the release folder (refuses unless the audit passes) |
| `build_explorer_data.py` | Data for the explorer page |
| `convert_sparc.py`, `regen_mast_thomson.py`, `lcfs_check.py` | SPARCPublic conversion and diagnostics |

## Regenerating

Run in the `freegsnke` conda env with `OMP_NUM_THREADS=1` per worker (14 workers shown). Workers skip design points already in `raw/`, so interrupted runs resume.

```bash
cd code
# MAST
python mast_machine.py && python gen_mast.py --build-pool --nproc 14
for k in $(seq 0 13); do python gen_mast.py --worker $k --n-workers 14 & done; wait
# DIII-D
python d3d_machine.py && python gen_d3d.py --build-pool --nproc 14
for k in $(seq 0 13); do python gen_d3d.py --worker $k --n-workers 14 & done; wait
for k in $(seq 0 13); do python gen_d3d.py --validate 280 --worker $k --n-workers 14 & done; wait
python gen_d3d.py --merge-validation
# SPARC (base equilibria first) and ITER
for k in $(seq 0 11); do python gen_machine.py --machine SPARC --build-bases --worker $k --n-workers 12 & done; wait
python gen_machine.py --machine SPARC --merge-bases
for m in SPARC ITER; do for k in $(seq 0 13); do python gen_machine.py --machine $m --worker $k --n-workers 14 & done; wait; done
# quality control, assembly, audit, release
for s in mast d3d sparc iter; do for k in $(seq 0 13); do python relabel_qc.py --set $s --worker $k --n-workers 14 & done; wait; done
python gen_mast.py --assemble && python gen_d3d.py --assemble
python gen_machine.py --machine SPARC --assemble && python gen_machine.py --machine ITER --assemble
python audit_datasets.py --sets mast d3d sparc iter --resolve 25
python make_release.py            # remove the existing release folder first to rebuild
python build_explorer_data.py
```

Each design takes 15–35 minutes and the quality-control pass about 20 minutes on 14 cores.
