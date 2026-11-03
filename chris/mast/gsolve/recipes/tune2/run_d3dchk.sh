#!/bin/bash
# usage: run_d3dchk.sh TAG KNOBS_JSON [COND_JSON] [NSHARD] [THREADS]
cd /fsx/chris.beckham/gsolve_data/tune2
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
TAG=$1; KN=$2; COND=${3:-}; NS=${4:-4}; TH=${5:-5}
for s in $(seq 0 $((NS-1))); do
  OMP_NUM_THREADS=$TH ./venv/bin/python gen_d3d.py --frames d3d/val.parquet --knobs "$KN" ${COND:+--cond $COND} \
     --out d3dchk/${TAG}_$s.npz --shard $s/$NS > logs/d3dchk_${TAG}_$s.log 2>&1 &
done
wait
OMP_NUM_THREADS=1 ./venv/bin/python d3d_score.py --tag $TAG --workers $((NS*TH)) > logs/d3dscore_$TAG.log 2>&1
echo DONE $TAG
