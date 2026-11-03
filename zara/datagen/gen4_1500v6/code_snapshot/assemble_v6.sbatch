#!/bin/bash
#SBATCH --job-name=m2-gen4v6-asm
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4/logs/%x-%j.out
# usage: sbatch -p <part> [-w node] assemble_v6.sbatch <out_dir> <n_train> <n_heldout> <name>
# Recipe v6 release: the accepted cases up to every quota in <name>_{train,heldout}/, then loader checks, per-frame
# metadata, the v6 checks against real MAST inputs (train and held-out), the realism checks and checksums.
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/freegsnke
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen4
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
D=$1; NAME=$4
L=logs/v6rel_${NAME}
python gen4_set.py --assemble --out $D --n-train $2 --n-heldout $3 --name $NAME --recipe v6 > ${L}_assemble.log 2>&1 || exit 1
python verify_set.py $D/${NAME}_train $D/${NAME}_heldout > ${L}_verify.log 2>&1
for s in train heldout; do
  python export_meta.py $D/${NAME}_$s --procs 16 > ${L}_meta_$s.log 2>&1
  python v6_stats.py $D/${NAME}_$s --procs 16 > ${L}_stats_$s.log 2>&1
done
python check.py $D/${NAME}_train --procs 16 > ${L}_check_train.log 2>&1
(cd $D && for s in ${NAME}_train ${NAME}_heldout; do sha256sum $s/*.parquet; done > SHA256SUMS_${NAME}.txt)
echo "release $NAME done"
