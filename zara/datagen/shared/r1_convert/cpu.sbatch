#!/bin/bash
#SBATCH --job-name=ch2-intr-cpu
#SBATCH --partition=g7e8-spot,gpu1,gpu2,gpu4,g6e1
#SBATCH --exclude=g7e8-spot-[9-10]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4/logs/%x-%j.out
# usage: sbatch [-J name --cpus-per-task N --mem M] slurm/cpu.sbatch <env> <workdir> <command...>   (CPU only)
set -euo pipefail
source /fsx/zara.ercan/env.sh
conda activate $( [ "$1" = fusion-equilibrium-starter ] && echo $1 || echo /fsx/zara.ercan/conda/envs/$1 ); shift
cd "$1"; shift
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%T)"
"$@"
echo "end $(date -u +%FT%T)"
