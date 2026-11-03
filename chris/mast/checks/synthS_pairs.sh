#!/bin/bash
# Synthetic-MAST S (g4_20000) of NAME CKPT pairs read from a file, as train_mm.py's --synthS scoring ran it, on cores $1.
# Skips pairs whose JSON already exists.   synthS_pairs.sh CORES PAIRS_FILE
set -u
CORES=$1; PAIRS=$(readlink -f "$2")
L=/fsx/chris.beckham/fec_upload/jobs
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
cd /fsx/chris.beckham/github/fec-final/chris/mast/starter/my_experiments/nn/synth_S
while read name ck; do
  [ -s $L/synthS_${name}.json ] && continue
  CUDA_VISIBLE_DEVICES= taskset -c $CORES nice -n 10 $PY -u ensemble_synth.py --out $L/synthS_${name}.json --sets g4_20000 \
    --threads 4 --workers 6 $name $ck > $L/synthS_${name}.log 2>&1
  echo "EXIT $?" >> $L/synthS_${name}.log
done < $PAIRS
