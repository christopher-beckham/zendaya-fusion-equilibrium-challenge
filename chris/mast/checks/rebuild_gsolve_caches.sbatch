#!/bin/bash
#SBATCH --job-name=fec-rebuild-gsolve
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/chris.beckham/fec_upload/jobs/rebuild_gsolve-%j.out
set -euo pipefail
hostname; nproc
/usr/bin/time -v /fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python -u /fsx/chris.beckham/fec_upload/jobs/rebuild_gsolve_caches.py 16
