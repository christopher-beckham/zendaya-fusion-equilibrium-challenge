#!/bin/bash
#SBATCH --job-name=m2-gen4set-asm
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4/logs/%x-%j.out
# usage: sbatch -p <part> [-w node] assemble_set.sbatch <out_dir> <n_train> <n_heldout> <name> <recipe> <tag>
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/freegsnke
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4
export PYTHONWARNINGS=ignore
D=$1; T=$6
python gen4_set.py --assemble --out $D --n-train $2 --n-heldout $3 --name $4 --recipe $5 > logs/assemble_$T.log 2>&1 || exit 1
python verify_set.py $D/$4_train $D/$4_heldout > logs/verify_$T.log 2>&1
python check.py $D/$4_train --procs 4 > logs/check_train_$T.log 2>&1
python check.py $D/$4_heldout --procs 4 > logs/check_heldout_$T.log 2>&1
(cd $D && for s in $4_train $4_heldout; do sha256sum $s/*.parquet; done > SHA256SUMS.txt)
