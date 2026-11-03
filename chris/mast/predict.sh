#!/usr/bin/env bash
# Predict all 13 of Chris's MAST models on one MAST test set and post-process them.
#
#   FEC_DATA_ROOT=... FEC_CACHE=... bash chris/mast/predict.sh SPLIT OUT_DIR [--rebuilt] [--weights DIR]
#
# SPLIT: mast_public_test or mast_private_test. FEC_CACHE must hold that split's inputs
# (build_mast_test_inputs.sh SPLIT). Weights: --weights DIR (folder of run folders) or $FEC_WEIGHTS; --rebuilt uses
# train.sh's retrained checkpoints instead. Output: OUT_DIR/<model>/SPLIT.npz. Z runs before Z6b (Z6b takes Z's q95).
set -euo pipefail
SPLIT=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
for m in Z Z2 Z3 Z6b Z7 Z8 Z9qb Z15 Z17 Z18 FA FC V7C; do
  echo "[$(date -u +%T)] $m"
  $PY -u "$HERE/predict_model.py" "$m" --split "$SPLIT" --out "$OUT" "$@"
done
echo "[$(date -u +%T)] done"
