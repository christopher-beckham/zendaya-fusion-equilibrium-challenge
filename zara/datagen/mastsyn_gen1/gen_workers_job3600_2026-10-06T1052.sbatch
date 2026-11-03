#!/bin/bash
#SBATCH --job-name=m2-gen
#SBATCH --partition=g7e8-spot
#SBATCH --nodelist=g7e8-spot-1
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/logs/%x-%j.out
# usage: sbatch --cpus-per-task=N --mem=M gen_workers.sbatch <pool> <n_workers> <n_points>
#   runs n_workers single-threaded gen_m2.py workers on the already-running node (no new node boots)
set -euo pipefail
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/freegsnke
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
POOL=$1; NW=$2; NP=$3
for k in $(seq 0 $((NW-1))); do
  python gen_m2.py --pool $POOL --worker $k --n-workers $NW --n-points $NP > /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/logs/gen_${POOL}_w${k}.log 2>&1 &
done
wait
python gen_m2.py --summary --pool $POOL
