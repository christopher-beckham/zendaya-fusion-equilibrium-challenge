#!/bin/bash
# Step 4: download the trained weights that made the submitted files, each checked by SHA256 against its pinned manifest.
#   bash steps/4_shipped_weights.sh A|B
# Both: DIII-D networks and training-side inference files, zenithon-zendaya/diii-d-model-weights @ c2e8c4a3 (24.9 GB; it
#       holds every DIII-D recipe): networks into zara/diii-d/my_experiments/, inference files into $FEQ_CACHE
#       (zara/diii-d/fetch_weights.py, zara/diii-d/weights_manifest.json).
# B:    Chris's 30 MAST checkpoints, zenithon-zendaya/mast-model-weights @ 8616a376 (2.9 GB), into $FEC_WEIGHTS
#       (data/fetch_weights.py, data/weights_manifest.json); r1's four networks and three data files,
#       zenithon-zendaya/zara-r1-weights @ 5ca03a54, into $R1_WEIGHTS/runs and $R1_ROOT/cache (zara/mast/r1/predict/fetch_r1.py,
#       r1_files.json); and r1's first-pass cache of the Final MAST inputs, beckhamc/zara-r1-cpu-caches @ d9be687e (579 MB),
#       into $R1_ROOT/cache (r1_final_firstpass.json): built on the CPU from the Final inputs only, the cache the submitted
#       file was predicted with. Without it, predict.sh B builds it again (about 2 hours on 96 CPU cores).
# The shipped DIII-D networks land where training writes its own, so this copy of the repository is then for the shipped-weights
# check only: prepare.sh, train.sh and predict.sh --rebuilt refuse to run in it (REPRODUCE.md, step 0).
set -euo pipefail
source "$(dirname "$0")/../settings.sh"
case "${1:-}" in A|B) ;; *) echo "usage: $0 A|B"; exit 2 ;; esac
if [ -e $REPO/.retrained ] || [ -e $FEQ_CACHE/.retrained ]; then
  echo "This copy (or its \$FEQ_CACHE) holds retrained networks: the shipped weights would overwrite them."
  echo "Use another copy of the repository and another WORK folder for the shipped-weights check (REPRODUCE.md)."; exit 3
fi
mkdir -p $FEQ_CACHE && touch $REPO/.shipped_weights $FEQ_CACHE/.shipped_weights
(cd $REPO/zara/diii-d && $ENVS/fusion-eq-gpu/bin/python fetch_weights.py)
if [ "$1" = B ]; then
  $PY $REPO/data/fetch_weights.py --root $FEC_WEIGHTS
  $R1_PY $REPO/zara/mast/r1/predict/fetch_r1.py --root $R1_WEIGHTS
fi
