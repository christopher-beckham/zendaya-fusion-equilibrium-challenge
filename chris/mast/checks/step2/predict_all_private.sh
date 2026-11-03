#!/bin/bash
# predict_model.py for all 13 models on mast_private_test (the Final inputs), two GPUs.
set -u
P=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python; O=/fsx/chris.beckham/fec_upload/step2/pred_private
export FEC_DATA_ROOT=/fsx/chris.beckham/fec_official/final FEC_CACHE=/fsx/chris.beckham/fec_cache/final
mkdir -p $O; cd /fsx/chris.beckham/github/fec-final
worker() {
  local g=$1 c=$2; shift 2
  for m in "$@"; do
    [ -s $O/$m/mast_private_test.npz ] && continue
    CUDA_VISIBLE_DEVICES=$g taskset -c $c $P -u chris/mast/predict_model.py $m --split mast_private_test --out $O --workers 6 > $O/$m.log 2>&1
    echo "EXIT $?" >> $O/$m.log
  done
}
worker 2 36-41 Z Z9qb Z15 Z2 Z8 V7C Z6b &
worker 3 42-47 Z18 FA Z3 Z7 Z17 FC &
wait
echo "all done"
