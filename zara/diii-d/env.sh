#!/bin/bash
# Paths and interpreters of the Challenge 1 (DIII-D) pipeline. Source it (the entrypoints do); override any variable before.
#   FEQ_ROOT        this package (the repository root: challenge_data/, fusion_scoring/, my_experiments/)
#   FEQ_CACHE       caches, intermediate files and predictions (about 150 GB for training, 15 GB per test split)
#   FEQ_PY          python of the GPU environment (environment/fusion-eq-gpu.yml: python 3.11, torch 2.11 + CUDA 12.8)
#   FEQ_PY_STARTER  python of the starter environment used to build the c1 input caches (environment/fusion-equilibrium-starter.yml)
#   FEQ_WORKERS     CPU worker processes for the CPU steps
# Data: raw parquet files of each split in $FEQ_ROOT/challenge_data/data/<split>/ (Hugging Face Sophelio/fusion-equilibrium-challenge,
# configs diii_d_train, diii_d_public_test, diii_d_private_test).
export FEQ_ROOT=${FEQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export FEQ_CACHE=${FEQ_CACHE:-$FEQ_ROOT/cache}
export FEQ_PY=${FEQ_PY:-python}
export FEQ_PY_STARTER=${FEQ_PY_STARTER:-$FEQ_PY}
export FEQ_WORKERS=${FEQ_WORKERS:-16}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation; no numerical effect
