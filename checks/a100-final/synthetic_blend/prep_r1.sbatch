#!/bin/bash
#SBATCH -p g7e8-spot
#SBATCH -c 96
#SBATCH --mem=600G
#SBATCH -J synval_r1_firstpass
#SBATCH -o /fsx/chris.beckham/synval/prep_r1.log
# Steps 1-2 of zara/mast/r1/predict/predict_r1.sh (input cache, first-pass cache), same commands and settings.
set -euo pipefail
S=/fsx/chris.beckham/synval; cd $S/repo-shipped
export WORK=$S/work-shipped ENVS=/fsx/chris.beckham/fec_pkgtest/work/envs DATA=$S/DATA; source settings.sh
HERE=$S/repo-shipped/zara/mast/r1/predict; ME=$(cd $HERE/../starter/my_experiments && pwd); G4=$ME/ch2/g4
PQ=$DATA/final/data/mast_private_test; K=private; W=$S/r1work; INP=$W/inputs; FP=$R1_ROOT/cache/ch2/g4/caches/mast_${K}_iv_full_wdiv; TH=12
mkdir -p $W; NS=$(ls $PQ | grep -c '\.parquet$'); echo "[$(date -u +%T)] $NS shots"
(cd $ME/c2 && CUDA_VISIBLE_DEVICES= $R1_PY_STARTER preprocess_mast.py --data-dir $PQ --out $INP --workers 48)
echo "[$(date -u +%T)] input cache: $(ls $INP | grep -c '^mast_shot.*\.npz$')"
[ "$(ls $INP | grep -c '^mast_shot.*\.npz$')" -eq "$NS" ] || { echo "input cache incomplete"; exit 3; }
pids=""
for s in 0 1 2 3 4 5 6 7; do
  (cd $G4/fz && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH INTR_WALL=0.20 INTR_FP=divert \
     $R1_PY $G4/lbfit/split_run_any.py $G4/fz $K $INP build_cache.py --machine MAST --split $K --sol-factor 0.5 \
     --p-share 0.7 --tag _iv_full_wdiv --check --shard $s --n-shards 8 > $W/first_pass_$s.log 2>&1) & pids="$pids $!"
done
for p in $pids; do wait $p; done
echo "[$(date -u +%T)] first-pass cache: $(ls $FP | grep -c '\.npz$') of $NS"
[ "$(ls $FP | grep -c '\.npz$')" -eq "$NS" ] || exit 3
