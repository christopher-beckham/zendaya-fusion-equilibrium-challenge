#!/bin/bash
# Challenge 1 (DIII-D) training entrypoint of the Ch1-safe recipe: rebuilds every trained component from the official DIII-D
# training data, on one GPU, step after step. Official DIII-D data only (no other machine, no synthetic data).
#
#   bash build_caches.sh ch1safe && bash train_ch1safe.sh        (data: challenge_data/data/diii_d_train/ and challenge_data/data/diii_d_public_test/)
#
# Each step is skipped when its output exists, so an interrupted run resumes. Wall-clock time per step goes to
# $FEQ_CACHE/train_times_ch1safe.tsv. Afterwards: bash predict_ch1safe.sh <split> for any inputs-only split.
# The v13 commands (train_ch1.sh) with two changes: the 12 psi networks and 8 functional heads train on a 40-epoch schedule
# instead of 60, and the q95 / betaN recipe is R3c-lite (the 6 MLP of plans/prod_hm6.txt + the 6 sequence networks, no
# plans/prod_hm12_*, no variant-D networks). Trained components (all on the 7,041 training shots, no holdout):
#   C1 physics + Ridge base (CPU), 12 Group H psi networks (40 epochs), 8 functional heads (40 epochs), 3 sequence heads (as
#   v13), and the q95 / betaN networks: 5 out-of-fold psi networks (map features for training frames), 6 MLP + 6 sequence
#   scalar networks.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; W=$FEQ_WORKERS; C=$FEQ_CACHE
mkdir -p $C/nn/px $C/ml/ch1h/hist $C/ml/ch1c $C/ml/ch1s $C/ml/ch1q/oof $C/ml/ch1q/hist_ch1h   # every cache folder the steps write to
LOG=$C/train_times_ch1safe.tsv; [ -f $LOG ] || printf "step\tseconds\tfinished_utc\tkind\n" > $LOG
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
EPX=""; [ -n "${FEQ_EPOCHS:-}" ] && EPX="--epochs $FEQ_EPOCHS"   # smoke tests only: override every epoch count
EP40="--epochs ${FEQ_EPOCHS:-40}"                                 # the Ch1-safe schedule of the psi networks and heads (v13: 60)

# ---------------------------------------------------------------- 1-2. input caches, label-side targets, C1 physics + Ridge base
# (CPU, before this script: bash build_caches.sh ch1safe)
[ -e $C/build_caches_ch1safe.done ] || { echo "run bash build_caches.sh ch1safe first (the CPU preprocessing of this recipe)"; exit 1; }

# ---------------------------------------------------------------- 3. psi networks and functional heads (40 epochs), sequence heads (GPU)
cd $ML/ch1h
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  R=runs/safe_e40_psi_s$n; X=""; [ $n -ge 7 ] && X="--muon-wd 0.2"
  run psi_e40_$n "has $R/model.pt" $FEQ_PY train_c1h.py $BASE --lam-grad 1.0 --grad-w band --aux-cons 0.2 --aux-geo --hist $HTR \
      --hist-causal $X --init-seed $n $PROD $EP40 --out $R
done
for n in 1 2 3; do
  run seqhead_$n "has runs/prod_seqh_s$n/model.pt" $FEQ_PY train_c1h.py $BASE --scalars-only --aux-cons 1.0 --aux-geo --hist $HTR \
      --seq --init-seed $n $PROD $EPX --out runs/prod_seqh_s$n
done
cd $ML/ch1c
for n in 1 2 3; do
  run head_scog_e40_$n "has runs/safe_e40_scog_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --scalars-only --aux-cons 1.0 --aux-geo \
      --init-seed $n $PROD $EP40 --out runs/safe_e40_scog_s$n
  run head_scog_ctx_e40_$n "has runs/safe_e40_scog_ctx_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --scalars-only --aux-cons 1.0 --aux-geo \
      --ctx --init-seed $n $PROD $EP40 --out runs/safe_e40_scog_ctx_s$n
done
for n in 1 2; do
  run head_auxg_e40_$n "has runs/safe_e40_auxg_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --lam-grad 1.0 --grad-w band --aux-cons 0.2 \
      --aux-geo --init-seed $n $PROD $EP40 --out runs/safe_e40_auxg_s$n
done

# ---------------------------------------------------------------- 4. q95 / betaN networks of recipe R3c-lite
# (input export and history features: build_caches.sh)
cd $ML/ch1q
for i in 0 1 2 3 4; do
  run r3c_oof_$i "has $C/ml/ch1q/oof/poof$i/model.pt" $FEQ_PY train_c1q.py $BASE --lam-grad 1.0 --grad-w band --ridge $C/nn/ridge_full.npz \
      --cache $C/nn/diii_d_train --init-seed 1 --eval-every 20 --train-list lists/poof${i}_train.txt --pred-list lists/poof${i}_pred.txt \
      $EPX --out $C/ml/ch1q/oof/poof$i
  run r3c_oofqc_$i "has $C/ml/ch1q/qc_poof$i.npz" cpu $FEQ_PY qcurve_oof.py --fold poof$i --workers $W
done
run r3c_mapf "has $C/ml/ch1q/mapf_poof.npz" cpu $FEQ_PY assemble_oof.py --prefix poof
mkdir -p runs
for f in plans/prod_hm6.txt plans/prod_seq_{1,2,3,4,5,6}.txt; do   # R3c-lite: no plans/prod_hm12_*, no variant-D networks
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$C}
    args=$(echo "$args" | sed -E 's# --test-(qc|ens) [^ ]+##g')       # test-time inputs: predict_ch1safe.sh supplies them
    run r3c_$name "has runs/$name/model.pt" $FEQ_PY train_sq.py $BASE_S $args --no-test $EPX --out runs/$name
  done < $f
done
echo "== $(date -u +%FT%TZ) training complete; times in $LOG"
