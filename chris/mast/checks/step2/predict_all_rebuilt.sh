#!/bin/bash
# After the private predictions: all 13 models on mast_public_test from the RETRAINED checkpoints (retrain.py's
# RUN-rebuild/last.pt), then each compared with the original model's file (pooled R² and max |Δ| of ψ, q95, βN).
set -u
until grep -q "all done" /fsx/chris.beckham/fec_upload/step2/predict_all_private.out 2>/dev/null; do sleep 60; done
P=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python; O=/fsx/chris.beckham/fec_upload/step2/pred_rebuilt
W=/fsx/chris.beckham/github/fec-final/chris/mast/multimachine/my_experiments/nn/runs
mkdir -p $O; cd /fsx/chris.beckham/github/fec-final
worker() {
  local g=$1 c=$2; shift 2
  for m in "$@"; do
    [ -s $O/$m/mast_public_test.npz ] && continue
    CUDA_VISIBLE_DEVICES=$g taskset -c $c $P -u chris/mast/predict_model.py $m --rebuilt --weights $W --out $O --workers 6 > $O/$m.log 2>&1
    echo "EXIT $?" >> $O/$m.log
  done
}
worker 2 36-41 Z Z9qb Z15 Z2 Z8 V7C Z6b &
worker 3 42-47 Z18 FA Z3 Z7 Z17 FC &
wait
for m in Z Z2 Z3 Z6b Z7 Z8 Z9qb Z15 Z17 Z18 FA FC V7C; do
  $P chris/mast/checks/compare_predictions.py /fsx/chris.beckham/fec_upload/step2/pred_public/$m/mast_public_test.npz $O/$m/mast_public_test.npz --name $m
done
echo "all done"
