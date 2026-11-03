#!/usr/bin/env bash
# Build every derived cache Chris's models train and predict from, out of the official challenge data. CPU only.
#
#   FEC_DATA_ROOT=<challenge data folder> FEC_CACHE=<empty cache folder> bash chris/mast/build_caches.sh [WORKERS]
#
# FEC_DATA_ROOT holds data/diii_d_train, data/diii_d_public_test and data/mast_public_test. In order:
#   build_cache.py           DIII-D train memmaps (shots, frames, coils, psi, scal, phase), shards, score subset
#   build_features.py v5     coil pictures (features_v2/pictures_{d3d,mast}.npz) and features_v5 for train and tests
#   build_thomson_raw.py     raw Thomson Te and ne, DIII-D train and MAST public test
#   lindec_build.py          linear-decoder Green's matrices, DIII-D and MAST (limiter region)
#   build_mast_test_inputs.sh mast_public_test   MAST test inputs the training monitors read
# The synthetic training caches come from data/fetch.py (FEC_CACHE/synth). Normalisation statistics are written by
# train_mm.py on first use. Takes about 20 minutes with 24 workers.
set -euo pipefail
W=${1:-8}
: "${FEC_DATA_ROOT:?set FEC_DATA_ROOT}" "${FEC_CACHE:?set FEC_CACHE}"
HERE=$(cd "$(dirname "$0")" && pwd)
NN=$HERE/multimachine/my_experiments/nn
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=
mkdir -p "$FEC_CACHE"
cd "$NN"
echo "[$(date -u +%T)] DIII-D train cache"
$PY -u build_cache.py --cache "$FEC_CACHE" --data-root "$FEC_DATA_ROOT" --workers "$W"
echo "[$(date -u +%T)] coil pictures and features_v5"
env -u FEC_MAST_MASK $PY -u build_features.py --version v5 --workers "$W" --cache "$FEC_CACHE" --data-root "$FEC_DATA_ROOT"
echo "[$(date -u +%T)] Thomson Te and ne"
env -u FEC_MAST_MASK -u FEC_MAST_SPLIT $PY -u build_thomson_raw.py --workers "$W"
env -u FEC_MAST_MASK -u FEC_MAST_SPLIT $PY -u build_thomson_raw.py --quantity ne --workers "$W"
echo "[$(date -u +%T)] linear-decoder matrices"
env -u FEC_MAST_MASK $PY -u lindec_build.py --cache "$FEC_CACHE" --machine d3d
FEC_MAST_MASK=limiter $PY -u lindec_build.py --cache "$FEC_CACHE" --machine mast
echo "[$(date -u +%T)] MAST public-test inputs"
PY=$PY bash "$HERE/build_mast_test_inputs.sh" mast_public_test "$W"
echo "[$(date -u +%T)] done"
