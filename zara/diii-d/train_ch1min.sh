#!/bin/bash
# Challenge 1 (DIII-D) training entrypoint of the Ch1-min recipe (the DIII-D file of the Challenge 2 submission): rebuilds every
# trained component from the official DIII-D training data, on one GPU, step after step. Official DIII-D data only.
#
#   bash build_caches.sh ch1min && bash train_ch1min.sh         (data: challenge_data/data/diii_d_train/ and challenge_data/data/diii_d_public_test/)
#
# Each step is skipped when its output exists, so an interrupted run resumes. Wall-clock time per step goes to
# $FEQ_CACHE/train_times_ch1min.tsv. Afterwards: bash predict_ch1min.sh <split> for any inputs-only split.
# Trained components (all on the 7,041 training shots, no holdout), a subset of v13 (train_ch1.sh) with the same commands:
#   C1 physics + Ridge base (CPU); 6 Group H psi networks (v13 seeds 1-6) on a 30-epoch schedule instead of 60; 3 sequence heads
#   (as v13); 3 variant-D q95 / betaN networks (v13's prod_xbmix_D, prod_ctx5_D, prod_ctx5sm_D). No functional heads (the
#   psi networks' own shape outputs serve as calibration targets, "B1"), no out-of-fold networks, no R3c MLP / sequence nets.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; W=$FEQ_WORKERS; C=$FEQ_CACHE
mkdir -p $C/nn/px $C/ml/ch1h/hist $C/ml/ch1c $C/ml/ch1s   # every cache folder the steps write to
LOG=$C/train_times_ch1min.tsv; [ -f $LOG ] || printf "step\tseconds\tfinished_utc\tkind\n" > $LOG
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
BASE="--head free --epochs 60 --batch 2048 --width 2048 --depth 4 --lam-scal 0.2 --lam-axis 1.0 --w-band 10 --w-band-sigma 1.5 \
--px $C/nn/px/diii_d_train --firstpass --curves --opt muon --muon-lr 1e-3 --muon-adjust match_rms_adamw --muon-wd 0.1"
PROD="--holdout 0 --ridge $C/nn/ridge_full.npz --cache $C/nn/diii_d_train"
BASE_S="--targets q95,betaN,xb --pos --opt muon --muon-wd 0.1 --epochs 40"
EP_PSI=30                                                     # the Ch1-min psi schedule (v13: 60)
EPX=""; [ -n "${FEQ_EPOCHS:-}" ] && { EPX="--epochs $FEQ_EPOCHS"; EP_PSI=$FEQ_EPOCHS; }   # smoke tests only

# ---------------------------------------------------------------- 1-2. input caches, label-side targets, C1 physics + Ridge base
# (CPU, before this script: bash build_caches.sh ch1min)
[ -e $C/build_caches_ch1min.done ] || { echo "run bash build_caches.sh ch1min first (the CPU preprocessing of this recipe)"; exit 1; }

# ---------------------------------------------------------------- 3. psi networks (30-epoch schedule) and sequence heads (GPU)
cd $ML/ch1h
for n in 1 2 3 4 5 6; do
  R=runs/min_e30_psi_s$n
  run psi_e30_$n "has $R/model.pt" $FEQ_PY train_c1h.py $BASE --lam-grad 1.0 --grad-w band --aux-cons 0.2 --aux-geo --hist $HTR \
      --hist-causal --init-seed $n $PROD --epochs $EP_PSI --out $R
done
for n in 1 2 3; do
  run seqhead_$n "has runs/prod_seqh_s$n/model.pt" $FEQ_PY train_c1h.py $BASE --scalars-only --aux-cons 1.0 --aux-geo --hist $HTR \
      --seq --init-seed $n $PROD $EPX --out runs/prod_seqh_s$n
done

# ---------------------------------------------------------------- 4. variant-D q95 / betaN networks (GPU; input export and x_b targets: build_caches.sh)
cd $ML/ch1s
mkdir -p runs
mapfile -t D3 < <(cat plans/prodD_{a,b,c}.txt | grep -E "^(prod_xbmix_D|prod_ctx5_D|prod_ctx5sm_D) ")
for line in "${D3[@]}"; do
  read -r name args <<< "$line"
  args=${args//@FEQ_CACHE@/$C}
  args=$(echo "$args" | sed -E 's#--xbmix ,[^ ]*#--xbmix ,#')         # wall fractions of training frames come from true maps
  run d3_$name "has runs/$name/model.pt" $FEQ_PY train_s.py $BASE_S $args --no-test $EPX --out runs/$name
done
echo "== $(date -u +%FT%TZ) training complete; times in $LOG"
