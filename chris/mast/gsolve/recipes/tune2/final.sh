#!/bin/bash
# Final stage: tuned 1,000-shot set (same shots as s1000, every 2nd frame, 3 shards) -> per-slice-noise conversion
# (convert_perslice.py) -> cache + shape targets + practice set (build_set.sh with SKIP_CONVERT=1), then the test-panel
# comparison (gsolve defaults through the original conversion vs tuned through the per-slice conversion, 3 fresh
# batches each), the conversion-path check and the label table.
#   final.sh [KNOBS_JSON]      (default: the knobs of the incumbent in incumbent_s6.json)
# Env overrides (for a dry run on a few shots; defaults = the real deliverable):
#   NAME=tuned1000 TAG=gsolve_tuned1000 SET=g4_gst1000 NSHOTS=0 (0 = all 1,000) READY=.../tune/FINAL_READY
#   SKIP_EXTRA=0 (1 = skip the test-panel / pipeline / label steps) GEN_THREADS=14 CONV_WORKERS=40 BUILD_WORKERS=28
set -euo pipefail
cd /fsx/chris.beckham/gsolve_data/tune
NAME=${NAME:-tuned1000}; TAG=${TAG:-gsolve_tuned1000}; SET=${SET:-g4_gst1000}; NSHOTS=${NSHOTS:-0}
READY=${READY:-/fsx/chris.beckham/gsolve_data/tune/FINAL_READY}; SKIP_EXTRA=${SKIP_EXTRA:-0}
GEN_THREADS=${GEN_THREADS:-14}; CONV_WORKERS=${CONV_WORKERS:-40}; BUILD_WORKERS=${BUILD_WORKERS:-28}
OUT=/fsx/chris.beckham/gsolve_data/$NAME          # build_set.sh reads $OUT/gen4_layout
mkdir -p $OUT
ts() { echo "[$(date -u +%H:%M:%S)] $*"; }
if [ $# -ge 1 ]; then KN=$(realpath "$1"); else
  KN=$OUT/final_knobs.json
  ./venv/bin/python -c "import json; d=json.load(open('incumbent_s6.json')); json.dump(d['knobs'], open('$KN','w'), indent=1); print('incumbent', d['tag'], d['loss_mean'])"
fi
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
PY=./venv/bin/python
$PY -c "import json,tune; json.dump(tune.defaults() | json.load(open('$KN')), open('$OUT/knobs.json','w'), indent=1)"
ts "knobs $(cat $OUT/knobs.json | tr -d '\n ')"
SHOTS=${SHOTS_FILE:-/fsx/chris.beckham/gsolve_data/s1000/shots.txt}
if [ "$NSHOTS" -gt 0 ]; then head -n $NSHOTS $SHOTS > $OUT/shots.txt; SHOTS=$OUT/shots.txt; fi
ts "generate"
for s in 0 1 2; do
  sed -n "$((s+1))~3p" $SHOTS > $OUT/shard_$s.txt
  if [ ! -f $OUT/${NAME}_part$s.npz ] && [ ! -f $OUT/raw_part$s.npz ]; then
    OMP_NUM_THREADS=$GEN_THREADS nohup $PY gen_param.py --files-from $OUT/shard_$s.txt --out $OUT/raw_part$s.npz --knobs $OUT/knobs.json \
      --every 2 --seed $((5000 + s)) > $OUT/gen_$s.log 2>&1 &
  fi
done
wait
for s in 0 1 2; do
  [ -f $OUT/${NAME}_part$s.npz ] || $PY - <<P
import numpy as np
z = dict(np.load("$OUT/raw_part$s.npz"))
np.savez("$OUT/${NAME}_part$s.npz", **{k: v for k, v in z.items() if v.ndim > 0})
P
done
rm -f $OUT/raw_part*.npz
ts "convert"
# per-slice conversion into the layout build_set.sh expects, then build_set.sh without its own conversion
if [ ! -f $OUT/gen4_layout/convert_summary.json ]; then
  OMP_NUM_THREADS=1 $PY -u convert_perslice.py --npz $OUT/${NAME}_part0.npz $OUT/${NAME}_part1.npz $OUT/${NAME}_part2.npz \
    --out $OUT/gen4_layout --heldout-thr 50 --workers $CONV_WORKERS --knobs $OUT/knobs.json > $OUT/convert.log 2>&1
fi
ts "build"
ST=/fsx/chris.beckham/fusion_cache/synth/${TAG}_limiter/shape_targets.npz
(rc=0; SKIP_CONVERT=1 WORKERS=$BUILD_WORKERS bash /fsx/chris.beckham/gsolve_data/convert/build_set.sh $NAME 50 $TAG $SET \
   $OUT/${NAME}_part0.npz $OUT/${NAME}_part1.npz $OUT/${NAME}_part2.npz > $OUT/build.log 2>&1 || rc=$?; echo "build exit $rc" >> $OUT/build.log
 if [ -f $ST ]; then echo $ST > $READY; ts "READY written ($READY)"; else ts "NO shape targets: $ST"; fi) &
if [ "$SKIP_EXTRA" != 1 ]; then
  for s in 10 11 12; do
    OMP_NUM_THREADS=16 $PY tune.py eval --tag test_base_s$s --seed $s --panel test --stage test --shards 4 --threads 4 --knobs '{}' --perslice 0 || echo "test_base_s$s failed"
    OMP_NUM_THREADS=16 $PY tune.py eval --tag test_tuned_s$s --seed $s --panel test --stage test --shards 4 --threads 4 --knobs $OUT/knobs.json --perslice 1 || echo "test_tuned_s$s failed"
  done
  OMP_NUM_THREADS=16 $PY pipeline_check.py $OUT/knobs.json pipe_check_tuned --seed 10 > pipeline_check.log 2>&1 || echo "pipeline check failed" >> pipeline_check.log
fi
wait
ts "build done: $(tail -1 $OUT/build.log)"
if [ "$SKIP_EXTRA" != 1 ]; then $PY compare_labels.py --tuned $OUT || echo "compare_labels failed"; fi
ts "FINAL DONE"
