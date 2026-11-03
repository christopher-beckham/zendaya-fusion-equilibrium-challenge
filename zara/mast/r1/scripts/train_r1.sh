#!/bin/bash
# train_r1.sh full OUT_ROOT [PYTHON]
# Trains Zara's r1 MAST model from scratch with the faster code in zara/mast/r1/starter, one run after another on one GPU,
# replaying each run's recorded arguments (zara/mast/r1/recorded/<run>/args.json). Only --init changes, to point at this
# chain's own outputs: h_muon_mo (28 epochs, random start) -> w3_ft -> rt_ftw3, and h_muon_mo -> rt_ft, rt_mix, rt_nonoise.
# Needs R1_ROOT (Zara's data with the CPU-built caches: zara/mast/r1/fetch_r1_data.py) and the official data linked by
# zara/mast/r1/link_data.sh.
# Each run writes OUT_ROOT/<run>/ (model.pt, hist.json) and OUT_ROOT/<run>.log; GPU memory is sampled to OUT_ROOT/<run>.gpu.csv.
set -uo pipefail
MODE=$1; OUT=$2; PY=${3:-python}
R=$(cd $(dirname $0)/.. && pwd); ME=$R/starter/my_experiments
[ -d "${R1_ROOT:?set R1_ROOT, the r1 data folder (zara/mast/r1/README.md)}/cache" ] || { echo "no cache/ under R1_ROOT=$R1_ROOT"; exit 2; }
[ -d $R/starter/challenge_data/data/diii_d_train ] || { echo "official data not linked: bash zara/mast/r1/link_data.sh DATA_ROOT"; exit 2; }
export R1_ROOT
CAPMEM=$ME/ml/final/capmem.py; RUNS=$R/recorded; mkdir -p $OUT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_MEM_CAP_GB=80
run() {  # name code_dir intr_wall [key=value overrides]
  local n=$1 fz=$2 iw=$3; shift 3
  if grep -q "^exit 0" $OUT/$n.log 2>/dev/null; then echo "skip $n (done)"; return 0; fi
  local args; args=$($PY $R/scripts/args_to_cli.py $fz/train.py $RUNS/$n/args.json "$@") || return 1
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 > $OUT/$n.gpu.csv &
  local smi=$!
  echo "$(date -u +%FT%TZ) start $n"
  local t0=$(date +%s)
  (cd $fz && INTR_WALL=$iw $PY $CAPMEM train.py --out $OUT/$n $args) > $OUT/$n.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  echo "exit $rc wall $(( $(date +%s) - t0 )) s" >> $OUT/$n.log
  echo "$(date -u +%FT%TZ) end $n: exit $rc, $(( $(date +%s) - t0 )) s"
  return $rc
}
SYN=$ME/ch2/syn/fz; G4=$ME/ch2/g4/fz
case $MODE in
  full)
    run h_muon_mo $SYN "" || exit 1
    run w3_ft $G4 0.20 init=$OUT/h_muon_mo/model.pt || exit 1
    run rt_ftw3 $G4 0.20 init=$OUT/w3_ft/model.pt || exit 1
    for m in rt_ft rt_mix rt_nonoise; do run $m $G4 0.20 init=$OUT/h_muon_mo/model.pt || exit 1; done ;;
  *) echo "MODE must be full"; exit 2 ;;
esac
echo "per-epoch times (s):"
for f in $OUT/*/hist.json; do $PY -c "import json,sys;h=json.load(open('$f'));print('  $(basename $(dirname $f)):',[round(e['time_s']) for e in h])"; done
