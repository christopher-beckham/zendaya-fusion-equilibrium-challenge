#!/bin/bash
#SBATCH --job-name=m2-gen4big
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4/logs/%x-%j.out
# usage: sbatch -p <part> [-w node] --cpus-per-task=N --mem=M run_big.sbatch <node_tag> <n_workers> <out_dir> <n_train> <n_heldout> <name> [recipe]
# Same recipe as the 500-case set (gen4_set.py), larger quotas; candidates are claimed through files, so several jobs can run.
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/freegsnke
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
python -m py_compile gen4.py gen4_set.py || exit 1
ARGS="--out $3 --n-train $4 --n-heldout $5 --name $6 ${7:+--recipe $7}"
mkdir -p $3
( flock 9; [ -f $3/candidates.json ] || python gen4_set.py --build-candidates $ARGS ) 9>$3/.build.lock
mkdir -p $3/code && cp gen4.py gen4_set.py pool.py $3/code/
python gen4_set.py --run --node-tag $1 --n-workers $2 $ARGS > logs/big_$1_$SLURM_JOB_ID.log 2>&1
