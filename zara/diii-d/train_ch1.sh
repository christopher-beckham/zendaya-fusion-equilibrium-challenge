#!/bin/bash
# Challenge 1 (DIII-D) training entrypoint: rebuilds every trained component of the submitted v13 pipeline from the official
# DIII-D training data, on one GPU, step after step. Official DIII-D data only (no other machine, no synthetic data).
#
#   bash build_caches.sh ch1 && bash train_ch1.sh            (data: challenge_data/data/diii_d_train/ and challenge_data/data/diii_d_public_test/)
#
# Each step is skipped when its output exists, so an interrupted run resumes. Wall-clock time per step goes to
# $FEQ_CACHE/train_times.tsv. Afterwards: bash predict_ch1.sh <split> for any inputs-only split.
# Every step in the foreground trains a network on the GPU. The only CPU work in this file (r3c_oofqc_*, r3c_mapf: q-curves and
# map features of the out-of-fold networks' maps) runs in the background with no GPU visible, while the GPU trains networks
# that do not read it, so the GPU does not wait for it; its steps are marked CPU in train_times.tsv.
# Trained components (all on the 7,041 training shots, no holdout):
#   C1 physics + Ridge base (CPU), 12 Group H psi networks, 8 functional heads, 3 sequence heads (psi map and calibration
#   targets), and the q95 / betaN networks of recipe R3c: 5 out-of-fold psi networks (map features for training frames),
#   12 MLP + 6 sequence scalar networks, 6 variant-D scalar networks.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; W=$FEQ_WORKERS; C=$FEQ_CACHE
mkdir -p $C/nn/px $C/ml/ch1h/hist $C/ml/ch1c $C/ml/ch1s $C/ml/ch1q/oof $C/ml/ch1q/hist_ch1h   # every cache folder the steps write to
LOG=$C/train_times.tsv; [ -f $LOG ] || printf "step\tseconds\tfinished_utc\tkind\n" > $LOG
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

# ---------------------------------------------------------------- 1-2. input caches, label-side targets, C1 physics + Ridge base
# (CPU, before this script: bash build_caches.sh ch1)
[ -e $C/build_caches_ch1.done ] || { echo "run bash build_caches.sh ch1 first (the CPU preprocessing of this recipe)"; exit 1; }

# ---------------------------------------------------------------- 3. out-of-fold psi networks of recipe R3c (GPU), first
cd $ML/ch1q
for i in 0 1 2 3 4; do
  run r3c_oof_$i "has $C/ml/ch1q/oof/poof$i/model.pt" $FEQ_PY train_c1q.py $BASE --lam-grad 1.0 --grad-w band --ridge $C/nn/ridge_full.npz \
      --cache $C/nn/diii_d_train --init-seed 1 --eval-every 20 --train-list lists/poof${i}_train.txt --pred-list lists/poof${i}_pred.txt \
      $EPX --out $C/ml/ch1q/oof/poof$i
done
# Their CPU follow-up runs in the background (low priority, no GPU visible) while the GPU trains step 4, which does not read it;
# only the MLP and sequence scalar networks of step 5 do, and they wait for it.
BGLOG=$C/train_ch1_background.log
( for i in 0 1 2 3 4; do
    run r3c_oofqc_$i "has $C/ml/ch1q/qc_poof$i.npz" cpu nice -n 19 $FEQ_PY qcurve_oof.py --fold poof$i --workers $W
  done
  run r3c_mapf "has $C/ml/ch1q/mapf_poof.npz" cpu nice -n 19 $FEQ_PY assemble_oof.py --prefix poof
) > $BGLOG 2>&1 &
BG=$!
killtree() { local c; for c in $(pgrep -P $1); do killtree $c; done; kill $1 2>/dev/null || true; }   # children first
trap 'killtree $BG' EXIT                                          # this script ends early: stop the background job too
echo "== $(date -u +%FT%TZ) r3c_oofqc_*, r3c_mapf (CPU) started in the background, log $BGLOG"

# ---------------------------------------------------------------- 4. psi networks, functional heads, sequence heads (GPU)
cd $ML/ch1h
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if [ $n -le 6 ]; then R=runs/prod_auxghc_s$n; X=""; else R=runs/prod_auxghcw2_s$n; X="--muon-wd 0.2"; fi
  run psi_$n "has $R/model.pt" $FEQ_PY train_c1h.py $BASE --lam-grad 1.0 --grad-w band --aux-cons 0.2 --aux-geo --hist $HTR \
      --hist-causal $X --init-seed $n $PROD $EPX --out $R
done
for n in 1 2 3; do
  run seqhead_$n "has runs/prod_seqh_s$n/model.pt" $FEQ_PY train_c1h.py $BASE --scalars-only --aux-cons 1.0 --aux-geo --hist $HTR \
      --seq --init-seed $n $PROD $EPX --out runs/prod_seqh_s$n
done
cd $ML/ch1c
for n in 1 2 3; do
  run head_scog_$n "has runs/prod_scog_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --scalars-only --aux-cons 1.0 --aux-geo \
      --init-seed $n $PROD $EPX --out runs/prod_scog_s$n
  run head_scog_ctx_$n "has runs/prod_scog_ctx_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --scalars-only --aux-cons 1.0 --aux-geo \
      --ctx --init-seed $n $PROD $EPX --out runs/prod_scog_ctx_s$n
done
for n in 1 2; do
  run head_auxg_$n "has runs/prod_auxg_s$n/model.pt" $FEQ_PY train_c1c.py $BASE --lam-grad 1.0 --grad-w band --aux-cons 0.2 \
      --aux-geo --init-seed $n $PROD $EPX --out runs/prod_auxg_s$n
done

# ---------------------------------------------------------------- 5. q95 / betaN networks of recipe R3c (GPU)
# (input export, history features: build_caches.sh; out-of-fold map features: the background job of step 3)
t0=$(date +%s)
wait $BG || { echo "background CPU step failed (log $BGLOG):"; tail -20 $BGLOG; exit 1; }
trap - EXIT
cat $BGLOG
echo "== $(date -u +%FT%TZ) background CPU steps done; the GPU waited $(( $(date +%s) - t0 )) s for them"
cd $ML/ch1q
mkdir -p runs
for f in plans/prod_hm6.txt plans/prod_hm12_{1,2,3,4,5,6}.txt plans/prod_seq_{1,2,3,4,5,6}.txt; do
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$C}
    args=$(echo "$args" | sed -E 's# --test-(qc|ens) [^ ]+##g')       # test-time inputs: predict_ch1.sh supplies them
    run r3c_$name "has runs/$name/model.pt" $FEQ_PY train_sq.py $BASE_S $args --no-test $EPX --out runs/$name
  done < $f
done
cd $ML/ch1s
mkdir -p runs
for f in plans/prodD_{a,b,c}.txt; do
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$C}
    args=$(echo "$args" | sed -E 's#--xbmix ,[^ ]*#--xbmix ,#')         # wall fractions of training frames come from true maps
    run r3c_$name "has runs/$name/model.pt" $FEQ_PY train_s.py $BASE_S $args --no-test $EPX --out runs/$name
  done < $f
done
echo "== $(date -u +%FT%TZ) training complete; times in $LOG"
