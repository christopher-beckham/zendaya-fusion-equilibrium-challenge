#!/bin/bash
#SBATCH -p g7e8-spot
#SBATCH --gres=gpu:1
#SBATCH -c 48
#SBATCH --mem=400G
# predict.sh B (MAST part) on the pooled synthetic split, then the score per set.  sbatch --export=C=shipped|rebuilt
set -euo pipefail
S=/fsx/chris.beckham/synval; cd $S/repo-$C
export WORK=$S/work-$C ENVS=/fsx/chris.beckham/fec_pkgtest/work/envs DATA=$S/DATA NCPU=48; source settings.sh
if [ $C = rebuilt ]; then                    # the CPU-built inputs are shared with the shipped copy
  rm -rf $MAST_FINAL_CACHE && ln -sfn $S/work-shipped/mast_final_cache $MAST_FINAL_CACHE
  ln -sfn $S/work-shipped/r1_data/cache/ch2/g4/caches/mast_private_iv_full_wdiv $R1_ROOT/cache/ch2/g4/caches/mast_private_iv_full_wdiv
fi
export R1_WORK=$S/r1work-$C; mkdir -p $R1_WORK; ln -sfn $S/r1work/inputs $R1_WORK/inputs
REB=""; [ $C = rebuilt ] && REB=--rebuilt
PARTS=mast bash predict.sh B $S/out-$C $REB
$ENVS/fusion-equilibrium-starter/bin/python -I $S/scripts/score.py $S $S/out-$C/mast_private_test.npz $S/score_$C.json
