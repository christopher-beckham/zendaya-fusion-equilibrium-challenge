#!/usr/bin/env bash
# Build a full candidate from checkpoints: ensemble -> centre-column q95 rule (×0.932) -> V2 βN; check and upload.
# usage: [GPUS="0 1"] [UPLOAD=0] build_qb.sh TAG CKPT [CKPT ...]   -> submissions/<TAG>qb-... (nothing is submitted)
set -u
TAG=$1; shift
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter; NN=$REPO/my_experiments/nn; S=$REPO/submissions
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python; export FEC_MAST_MASK=limiter
B=${TAG}-base; QB=${TAG}qb-q95cc0932-bnV2-limiter-c1
( cd $NN && GPUS="${GPUS:-0 1 2 3}" NO_UPLOAD=1 bash build_ensemble.sh $B "$@" > $S/build_${TAG}.log 2>&1 ) || { echo "BUILD FAILED"; tail -5 $S/build_${TAG}.log; exit 1; }
( cd $NN/q95_study/H_v4 && CUDA_VISIBLE_DEVICES= $PY apply_cc.py $S/$B $S/${B}-q --factor 0.932 --workers 12 | tail -1 )
( cd $NN/q95_study/F_betaN && CUDA_VISIBLE_DEVICES= $PY apply_bn_variant.py $S/${B}-q $S/$QB --variant V2 --workers 12 | tail -2 )
rm -rf $S/${B}-q
$PY - <<PYEOF
import numpy as np
z = np.load("$S/$QB/mast_public_test.npz"); q = np.concatenate([z[k] for k in z.files if k.endswith("_q95")])
b = np.concatenate([z[k] for k in z.files if k.endswith("_betaN")])
print("$QB", sum(k.endswith("_psirz") for k in z.files), "shots", sorted({str(z[k].dtype) for k in z.files}),
      "non-finite", sum(int((~np.isfinite(z[k])).sum()) for k in z.files), "q95>12.44 %.2f%%" % (100 * (q > 12.44).mean()),
      "betaN 10/50/90", np.percentile(b, [10, 50, 90]).round(2))
PYEOF
if [ "${UPLOAD:-1}" = 1 ]; then
  ( cd $NN && HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash upload_candidate.sh $QB 2>&1 | grep -E "MATCH|MISMATCH|ERROR|Error" ); ls $S/$QB/$QB.zip
fi
