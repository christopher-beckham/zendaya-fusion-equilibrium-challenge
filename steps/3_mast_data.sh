#!/bin/bash
# Step 3 (submission B only): the MAST training data and the inputs of the Final MAST shots. CPU only.
#   a. Synthetic MAST training data of Chris's models (zenithon-zendaya/mast-synthetic-data, 14.8 GB) into $FEC_CACHE/synth,
#      checked by SHA256 (data/fetch.py, data/manifest.json).
#   b. Derived caches of Chris's models, built from the official data (chris/mast/build_caches.sh; about 20 min on 24 cores).
#   c. r1's training data into $R1_ROOT: Zara's data bundle and the CPU-built first-pass caches, checked by SHA256
#      (zara/mast/r1/fetch_r1_data.py, zara/mast/r1/r1_data.json; about 45 GB).
#   d. Inputs of the 1,205 Final MAST shots for Chris's models, in their own cache folder $MAST_FINAL_CACHE, which reuses
#      the coil pictures built in b (chris/mast/build_mast_test_inputs.sh).
# Every part skips work that is already done. To check only the shipped weights (step 4), c can be left out:
# PARTS="a b d" bash steps/3_mast_data.sh.
set -euo pipefail
source "$(dirname "$0")/../settings.sh"
PARTS=${PARTS:-a b c d}; W=$NCPU
has() { [[ " $PARTS " == *" $1 "* ]]; }
if has a; then echo "[$(date -u +%T)] 3a. synthetic MAST training data"; $PY $REPO/data/fetch.py --root $FEC_CACHE; fi
if has b; then
  if [ -f $FEC_CACHE/.build_caches_done ]; then echo "3b. derived caches: done before"
  else echo "[$(date -u +%T)] 3b. derived caches"; bash $REPO/chris/mast/build_caches.sh $W && touch $FEC_CACHE/.build_caches_done; fi
fi
if has c; then echo "[$(date -u +%T)] 3c. r1 training data"; $R1_PY $REPO/zara/mast/r1/fetch_r1_data.py --root $R1_ROOT; fi
if has d; then
  echo "[$(date -u +%T)] 3d. inputs of the Final MAST shots"
  mkdir -p $MAST_FINAL_CACHE/features_v2
  for f in pictures_mast.npz pictures_d3d.npz; do ln -sfn $FEC_CACHE/features_v2/$f $MAST_FINAL_CACHE/features_v2/$f; done
  for f in $FEC_CACHE/lindec_G_*.npz; do [ -e "$f" ] && ln -sfn $f $MAST_FINAL_CACHE/; done
  FEC_DATA_ROOT=$DATA/final FEC_CACHE=$MAST_FINAL_CACHE bash $REPO/chris/mast/build_mast_test_inputs.sh mast_private_test $W
fi
echo "[$(date -u +%T)] step 3 done"
