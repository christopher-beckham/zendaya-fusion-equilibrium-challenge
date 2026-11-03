#!/bin/bash
# DIII-D CPU preprocessing of one recipe, run before its training entrypoint: per-shot input caches, label-side targets, the C1
# physics + Ridge base, and the q95 / betaN inputs that need no trained network (x_b targets, input export of the Ridge base map
# and first-pass features, Group H history of the scalar networks, out-of-fold lists). CPU only: every step runs with no GPU
# visible. Rule 6 asks for this work to be declared with its hardware and wall-clock; it does not count toward the 24 GPU-hours.
#
#   bash build_caches.sh <ch1|ch1safe|ch1min>     then     bash train_<recipe>.sh
#
# (ch1 = v13.) Each step is skipped when its output exists, so an interrupted run resumes. Wall-clock time per step goes to the
# recipe's step-time file ($FEQ_CACHE/train_times.tsv, train_times_ch1safe.tsv or train_times_ch1min.tsv), marked CPU; the
# training entrypoint appends its own steps to the same file. The CPU steps that read a trained network's output (r3c_oofqc_*,
# r3c_mapf) stay in the training entrypoint; train_ch1.sh runs them in the background while the GPU trains other networks.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
REC=${1:?recipe: ch1, ch1safe or ch1min}
case $REC in
  ch1)     LOG=$FEQ_CACHE/train_times.tsv;;
  ch1safe) LOG=$FEQ_CACHE/train_times_ch1safe.tsv;;
  ch1min)  LOG=$FEQ_CACHE/train_times_ch1min.tsv;;
  *) echo "unknown recipe $REC (ch1, ch1safe or ch1min)"; exit 2;;
esac
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; W=$FEQ_WORKERS; C=$FEQ_CACHE
mkdir -p $C/nn/px $C/ml/ch1h/hist $C/ml/ch1c $C/ml/ch1s $C/ml/ch1q/hist_ch1h   # every cache folder the steps write to
[ -f $LOG ] || printf "step\tseconds\tfinished_utc\tkind\n" > $LOG
run() {   # run <name> <done-test command> <command ...>: skip when done, else time it
  local name=$1 test=$2 kind=GPU; shift 2; [ "$1" = cpu ] && kind=CPU
  if eval "$test"; then echo "== $name: done, skip"; return 0; fi
  echo "== $(date -u +%FT%TZ) $name ($kind)"; local t0=$(date +%s)
  "$@"
  printf "%s\t%s\t%s\t%s\n" "$name" "$(( $(date +%s) - t0 ))" "$(date -u +%FT%TZ)" "$kind" >> $LOG
}
cpu() { CUDA_VISIBLE_DEVICES= "$@"; }   # CPU-only step: no GPU is visible to it, and train_times*.tsv marks it CPU
has() { [ -e "$1" ]; }
ndone() { [ -d "$1" ] && [ "$(ls "$1" | grep -c '\.npz$')" -ge "$2" ]; }
NTR=$(ls $FEQ_ROOT/challenge_data/data/diii_d_train/*.parquet | wc -l)
NTE=$(ls $FEQ_ROOT/challenge_data/data/diii_d_public_test/*.parquet | wc -l)
HTR=$C/ml/ch1h/hist/diii_d_train

# ---------------------------------------------------------------- 1. input caches and label-side targets
# (the public-test input caches are needed by the q95 / betaN input export, which writes train and public-test parts)
cd $FEQ_ROOT
run cache_nn_train "ndone $C/nn/diii_d_train $NTR" cpu $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/diii_d_train \
    --out $C/nn/diii_d_train --workers $W
run cache_nn_test "ndone $C/nn/diii_d_public_test $NTE" cpu $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/diii_d_public_test \
    --out $C/nn/diii_d_public_test --workers $W --no-targets
cd $NN
run cache_px_train "has $C/nn/px/diii_d_train/meta.json" cpu $FEQ_PY_STARTER preprocess_px.py --split diii_d_train --workers $W
run cache_px_test "has $C/nn/px/diii_d_public_test/meta.json" cpu $FEQ_PY_STARTER preprocess_px.py --split diii_d_public_test --workers $W
cd $ML/ch1h
run cache_hist_train "has $HTR/meta.json" cpu $FEQ_PY hist_features.py --split diii_d_train --workers $W
[ $REC = ch1min ] || run cache_hist_test "has $C/ml/ch1h/hist/diii_d_public_test/meta.json" cpu $FEQ_PY hist_features.py --split diii_d_public_test --workers $W
cd $ML/ch1c
run targets_cons "ndone $C/ml/ch1c/cons_train $NTR" cpu $FEQ_PY ref_train.py --workers $W
run targets_geo "ndone $C/ml/ch1c/geo_train $NTR" cpu $FEQ_PY geo_train.py --workers $W

# ---------------------------------------------------------------- 2. C1 physics + Ridge base on all training shots
cd $FEQ_ROOT
run ridge_fit "has $C/nn/c1_model_full_phys.pkl" cpu $FEQ_PY_STARTER $E/c1/train_parallel.py --data-dir challenge_data/data/diii_d_train \
    --physics field,shaf,eddy --workers $W --out $C/nn/c1_model_full_phys.pkl
run ridge_export "has $C/nn/ridge_full.npz" cpu $FEQ_PY_STARTER $NN/export_ridge.py $C/nn/c1_model_full_phys.pkl $C/nn/ridge_full.npz

# ---------------------------------------------------------------- 3. q95 / betaN inputs that need no trained network
cd $ML/ch1s
if [ $REC = ch1min ]; then XB=d3_xb; EXP=d3_export; else XB=r3c_xb; EXP=r3c_export; fi
run $XB "has $C/ml/ch1s/xb_all.npz" cpu $FEQ_PY xb_targets.py
# input export: Ridge base map and first-pass features of every training and public-test frame (torch on all CPU cores)
run $EXP "has $C/ml/ch1s/inp_rfull_train.npz" cpu env OMP_NUM_THREADS=$W MKL_NUM_THREADS=$W $FEQ_PY export_inputs.py \
    --ridge $C/nn/ridge_full.npz --tag rfull
if [ $REC != ch1min ]; then
  cd $ML/ch1q
  run r3c_hist_ch1h_train "has $C/ml/ch1q/hist_ch1h/diii_d_train/meta.json" cpu $FEQ_PY hist_features_ch1h.py --split diii_d_train --workers $W
  run r3c_hist_ch1h_test "has $C/ml/ch1q/hist_ch1h/diii_d_public_test/meta.json" cpu $FEQ_PY hist_features_ch1h.py --split diii_d_public_test --workers $W
  run r3c_hist_train "has $C/ml/ch1q/hist_train.npz" cpu $FEQ_PY hist_feats.py --part train --workers $W
  run r3c_hist_test "has $C/ml/ch1q/hist_test.npz" cpu $FEQ_PY hist_feats.py --part test --workers $W
  run r3c_hist_merge "has $C/ml/ch1q/hist_ch1hq_train.npz" cpu $FEQ_PY hist_from_ch1h.py --with-q --src $C/ml/ch1q/hist_ch1h
  mkdir -p lists
  run r3c_folds "has lists/poof4_pred.txt" cpu $FEQ_PY make_folds.py
fi
touch $C/build_caches_$REC.done
echo "== $(date -u +%FT%TZ) CPU preprocessing of $REC complete; next: bash train_$REC.sh; times in $LOG"
