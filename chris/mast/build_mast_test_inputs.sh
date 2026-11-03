#!/usr/bin/env bash
# Build the MAST test-set inputs the models and post-processing read, for one split, into $FEC_CACHE.
#
#   FEC_DATA_ROOT=<folder with data/SPLIT/> FEC_CACHE=<cache> bash chris/mast/build_mast_test_inputs.sh SPLIT [WORKERS]
#
# $FEC_CACHE must already hold features_v2/pictures_mast.npz (the coil pictures, built once from the public data).
# Writes, as the original public-test caches were built:
#   features_v5_SPLIT.{npy,parquet}                     build_features.py --only-config SPLIT
#   features_v2/pictures_mast_sol0.5.npz,
#   features_v5_SPLIT_sol0.5.{npy,parquet}              build_mast_sol_half.py (solenoid current halved)
#   features_v5_SPLIT_sol0.5_limiter.{npy,parquet}      the same with FEC_MAST_MASK=limiter
#   mast_inputs[_SPLIT].npz                             build_mast_inputs.py
#   thomson_raw_mast[_SPLIT].npz, thomson_raw_ne_mast[_SPLIT].npz   build_thomson_raw.py --mast-only
# (the _SPLIT suffix only for splits other than mast_public_test). CPU only.
set -euo pipefail
SPLIT=$1; W=${2:-8}
: "${FEC_DATA_ROOT:?set FEC_DATA_ROOT}" "${FEC_CACHE:?set FEC_CACHE}"
NN=$(cd "$(dirname "$0")" && pwd)/multimachine/my_experiments/nn
PY=${PY:-python}
export FEC_MAST_SPLIT=$SPLIT CUDA_VISIBLE_DEVICES=
cd "$NN"
echo "[$(date -u +%T)] features_v5 for $SPLIT"
env -u FEC_MAST_MASK $PY -u build_features.py --version v5 --only-config "$SPLIT" --workers "$W" \
  --data-root "$FEC_DATA_ROOT" --cache "$FEC_CACHE"
echo "[$(date -u +%T)] solenoid halved"
env -u FEC_MAST_MASK $PY -u build_mast_sol_half.py --workers "$W"
echo "[$(date -u +%T)] solenoid halved, limiter region"
FEC_MAST_MASK=limiter $PY -u build_mast_sol_half.py --workers "$W"
echo "[$(date -u +%T)] coil currents and Ip"
env -u FEC_MAST_MASK $PY -u build_mast_inputs.py --workers "$W"
echo "[$(date -u +%T)] Thomson Te and ne"
env -u FEC_MAST_MASK $PY -u build_thomson_raw.py --mast-only --workers "$W"
env -u FEC_MAST_MASK $PY -u build_thomson_raw.py --mast-only --quantity ne --workers "$W"
echo "[$(date -u +%T)] done"
