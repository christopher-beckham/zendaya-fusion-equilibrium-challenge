#!/bin/bash
# Two workers (one per GPU) take runs from rebuild_queue.txt in order; each run is timed and its GPU memory sampled.
# A run already finished (log ends with "EXIT 0") is skipped, so the script can be restarted after a crash.
#   rebuild_queue.sh GPU [GPU ...]   (one worker per GPU)
set -u
L=/fsx/chris.beckham/fec_upload/jobs
Q=$L/rebuild_queue.txt
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
REC=/fsx/chris.beckham/github/fec-final/chris/mast/run_records
cd /fsx/chris.beckham/github/fec-final
next() {   # atomically claim the next unclaimed run
  ( flock 9
    for r in $(cat $Q); do
      [ -e $L/claim_$r ] && continue
      touch $L/claim_$r || { echo "CLAIM_FAILED"; break; }; echo $r; break
    done ) 9>$L/rebuild_queue.lock
}
worker() {
  local gpu=$1 run
  while run=$(next); [ -n "$run" ]; do
    if [ "$run" = CLAIM_FAILED ]; then echo "[$(date -u +%F_%T)] gpu $gpu cannot write claim files; worker stops" >> $L/rebuild_queue.out; return; fi
    if tail -n 1 $L/rebuild_${run}.log 2>/dev/null | grep -q "^EXIT 0"; then continue; fi
    rm -rf chris/mast/multimachine/my_experiments/nn/runs/${run}-rebuild
    local st=""; [ -f $REC/$run/best_mast.json ] || st="--stop-epoch 30"
    nvidia-smi -i $gpu --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 > $L/gpumem_${run}.csv 2>&1 &
    local smi=$!
    echo "[$(date -u +%F_%T)] gpu $gpu start $run" >> $L/rebuild_queue.out
    local t0=$SECONDS
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES=$gpu $PY -u chris/mast/retrain.py $run $st > $L/rebuild_${run}.log 2>&1
    local rc=$?; echo "EXIT $rc" >> $L/rebuild_${run}.log; kill $smi
    if [ $rc -ne 0 ] && [ $SECONDS -lt $((t0 + 60)) ]; then echo "[$(date -u +%F_%T)] gpu $gpu $run failed within a minute; worker stops" >> $L/rebuild_queue.out; return; fi
    echo "[$(date -u +%F_%T)] gpu $gpu end $run exit $rc" >> $L/rebuild_queue.out
  done
}
for g in "$@"; do worker $g & done
wait
echo "[$(date -u +%F_%T)] queue done" >> $L/rebuild_queue.out
