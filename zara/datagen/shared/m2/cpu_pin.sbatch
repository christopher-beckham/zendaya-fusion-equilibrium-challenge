#!/bin/bash
#SBATCH --job-name=m2-cpu
#SBATCH --partition=g7e8-spot
#SBATCH --nodelist=g7e8-spot-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/logs/%x-%j.out
# usage: sbatch [--cpus-per-task=N --mem=M -J name] cpu_pin.sbatch <env> <workdir> <command...>   (pinned CPU job, no GPU)
set -euo pipefail
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/$1; shift
cd "$1"; shift
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$@"
