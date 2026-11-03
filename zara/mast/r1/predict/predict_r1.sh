#!/bin/bash
# r1 (Zara's MAST model) on the Final MAST inputs: writes OUT_DIR/r1/mast_private_test.npz, the r1 member file the blend reads.
#
#   bash zara/mast/r1/predict/predict_r1.sh OUT_DIR RUNS_DIR
#
# RUNS_DIR holds rt_ft, rt_mix, rt_ftw3 and rt_nonoise (model.pt each): the shipped weights (fetch_r1.py) or the output of
# zara/mast/r1/scripts/train_r1.sh full. Steps:
#   1. input cache of the split (CPU, no GPU visible): starter/my_experiments/c2/preprocess_mast.py
#   2. first-pass cache (CPU, no GPU visible), 8 shards at once with R1_FP_THREADS threads each (12, as in the submitted run):
#      starter/my_experiments/ch2/g4/fz/build_cache.py, written to $R1_ROOT/cache/ch2/g4/caches/mast_<split>_iv_full_wdiv
#      The solve's last bits change with the CPU, the thread count and the shard grouping: keep 12 threads and 8 shards to
#      match the submitted run (REPRODUCE.md, step 4).
#   3. r1_chain.sh (GPU and CPU)
# Settings: R1_ROOT (Zara's data folder), FEC_DATA_ROOT (folder holding data/mast_private_test), R1_PY (fusion-eq-gpu), R1_PY_STARTER
# (fusion-equilibrium-starter), R1_SPLIT (split key, default private), R1_WORK (scratch, default OUT_DIR/r1_work).
# Data read under R1_ROOT: README.md in this folder.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ME=$(cd $HERE/../starter/my_experiments && pwd); G4=$ME/ch2/g4
mkdir -p "${1:?output folder}"; OUT=$(cd "$1" && pwd); RUNS=$(cd "${2:?member runs folder}" && pwd)
PQ=${FEC_DATA_ROOT:?set FEC_DATA_ROOT}/data/mast_private_test; K=${R1_SPLIT:-private}
W=${R1_WORK:-$OUT/r1_work}; INP=$W/inputs; FP=${R1_ROOT:?set R1_ROOT, the r1 data folder}/cache/ch2/g4/caches/mast_${K}_iv_full_wdiv; export R1_ROOT
NS=$(ls $PQ | grep -c '\.parquet$'); TH=${R1_FP_THREADS:-12}
mkdir -p $W
echo "[$(date -u +%T)] r1: $NS Final MAST shots, members from $RUNS"
if [ "$(ls $INP 2>/dev/null | grep -c '^mast_shot.*\.npz$')" -ne "$NS" ]; then
  echo "[$(date -u +%T)] r1: input cache (CPU)"
  (cd $ME/c2 && CUDA_VISIBLE_DEVICES= ${R1_PY_STARTER:-python} preprocess_mast.py --data-dir $PQ --out $INP --workers ${R1_WORKERS:-16})
fi
if [ "$(ls $FP 2>/dev/null | grep -c '\.npz$')" -ne "$NS" ]; then
  echo "[$(date -u +%T)] r1: first-pass cache (CPU, 8 shards)"
  pids=""
  for s in 0 1 2 3 4 5 6 7; do
    (cd $G4/fz && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH INTR_WALL=0.20 INTR_FP=divert \
       ${R1_PY:-python} $G4/lbfit/split_run_any.py $G4/fz $K $INP build_cache.py --machine MAST --split $K --sol-factor 0.5 \
       --p-share 0.7 --tag _iv_full_wdiv --check --shard $s --n-shards 8 > $W/first_pass_$s.log 2>&1) & pids="$pids $!"
  done
  for p in $pids; do wait $p; done
  [ "$(ls $FP | grep -c '\.npz$')" -eq "$NS" ] || { echo "first-pass cache incomplete"; exit 3; }
fi
bash $HERE/r1_chain.sh $K $INP $W/chain $FP $PQ $RUNS
mkdir -p $OUT/r1 && cp $W/chain/r1_mast.npz $OUT/r1/mast_private_test.npz
echo "[$(date -u +%T)] r1: $OUT/r1/mast_private_test.npz"
