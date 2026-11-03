#!/bin/bash
# Prepares $FEQ_CACHE for inference from the shipped weights (CPU only, about 5 minutes with 16 cores):
#
#   python fetch_weights.py        (checkpoints into this package, inference artifacts into $FEQ_CACHE)
#   bash prepare_inference.sh
#   bash predict_ch1.sh <split>    (or predict_ch1safe.sh / predict_ch1min.sh)
#
# The q95 / βN networks recompute their label normalisation from the training labels when they load, so inference reads the
# per-shot training cache (nn/diii_d_train, built here from the official training data, byte-identical to the one the networks
# were trained on) and the official training split (challenge_data/data/diii_d_train/). Skipped when already built.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
E=$FEQ_ROOT/my_experiments; NN=$E/c1/nn; C=$FEQ_CACHE
NTR=$(ls $FEQ_ROOT/challenge_data/data/diii_d_train/*.parquet | wc -l)
for f in nn/ridge_full.npz ml/ch1s/inp_rfull_train.npz ml/ch1q/mapf_poof.npz; do
  [ -e $C/$f ] || { echo "missing $C/$f: run fetch_weights.py with FEQ_CACHE set first"; exit 1; }
done
if [ -d $C/nn/diii_d_train ] && [ "$(ls $C/nn/diii_d_train | grep -c '\.npz$')" -ge "$NTR" ]; then
  echo "== training cache: done, skip"
else
  echo "== $(date -u +%FT%TZ) training cache ($NTR shots)"
  cd $FEQ_ROOT && $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/diii_d_train --out $C/nn/diii_d_train --workers $FEQ_WORKERS
fi
echo "== $(date -u +%FT%TZ) ready for predict_*.sh"
