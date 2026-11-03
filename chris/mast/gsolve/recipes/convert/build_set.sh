#!/usr/bin/env bash
# gsolve npz -> gen4 layout -> synth cache (limiter) + shape targets -> held-out practice set.
#   build_set.sh NAME HELDOUT_THR TAG SETNAME NPZ...
#   e.g. build_set.sh s1000 50 gsolve_s1000 g4_gs1000 /fsx/chris.beckham/gsolve_data/s1000/s1000_part{0,1,2}.npz
set -euo pipefail
NAME=$1; THR=$2; TAG=$3; SET=$4; shift 4
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
OUT=/fsx/chris.beckham/gsolve_data/$NAME/gen4_layout
W=${WORKERS:-8}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/fsx/chris.beckham/gsolve_copy FEC_SYNTH_HALF=${FEC_SYNTH_HALF:-4}
cd /fsx/chris.beckham/gsolve_data/convert
[ -n "${SKIP_CONVERT:-}" ] || $PY -u convert_gsolve.py --npz "$@" --out $OUT --heldout-thr $THR --workers $W
$PY -u sanity.py $OUT
cd /fsx/chris.beckham/github/fec-multimachine/my_experiments/nn
export FEC_MAST_MASK=limiter
$PY -u synth_cache_seq.py --src $OUT/train --heldout-src $OUT/heldout --tag ${TAG}_limiter --workers $W
$PY -u synth_shape_targets.py $TAG --workers $W
cd /fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/my_experiments/nn/synth_S
$PY - <<EOF
import json
p='extra_sets.json'; d=json.load(open(p))
d["$SET"]={"limiter_cache":"${TAG}_limiter","parquet":"$OUT/heldout"}
open(p,'w').write(json.dumps(d,indent=1))
EOF
$PY build_sets.py subset $SET
$PY scorer.py truth $SET
SYNTHS_WORKERS=$W $PY scorer.py perfect $SET | grep -v RESULT | head -2
