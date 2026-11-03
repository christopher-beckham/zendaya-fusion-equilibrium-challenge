#!/bin/bash
# Rebuild a pipeline-C candidate from its kept -base folder with the fec-final copies of apply_cc.py and
# apply_bn_variant.py, then compare with the original candidate byte for byte.
#   postproc_check.sh OUT_TAG BASE_DIR CANDIDATE_DIR      (env: extra variables to test, e.g. FEC_CACHE=...)
set -euo pipefail
OUT=/fsx/chris.beckham/fec_upload/step2/$1; BASE=$2; CAND=$3
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
Q=/fsx/chris.beckham/github/fec-final/chris/mast/starter/my_experiments/nn/q95_study
mkdir -p $OUT; rm -rf $OUT/q $OUT/final
export CUDA_VISIBLE_DEVICES= FEC_MAST_MASK=limiter
taskset -c 36-47 $PY -u $Q/H_v4/apply_cc.py $BASE $OUT/q --factor 0.932 --workers 12 > $OUT/apply_cc.log 2>&1
taskset -c 36-47 $PY -u $Q/F_betaN/apply_bn_variant.py $OUT/q $OUT/final --variant V2 --workers 12 > $OUT/apply_bn.log 2>&1
cmp $OUT/final/mast_public_test.npz $CAND/mast_public_test.npz && echo "MATCH: $1 mast_public_test.npz identical to $(basename $CAND)"
