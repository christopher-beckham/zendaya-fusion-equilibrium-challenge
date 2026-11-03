#!/bin/bash
# Challenge 1 (DIII-D) inference entrypoint of the Ch1-safe recipe on any inputs-only DIII-D split:
#
#   bash predict_ch1safe.sh <split>       e.g. diii_d_private_test (raw parquet files in challenge_data/data/<split>/)
#
# Output: $FEQ_CACHE/pred/ch1safe/<split>/sub/<split>.npz (shot_XXXX_psirz / _q95 / _betaN, shots in sorted parquet order, psi
# float32). Same steps as predict_ch1.sh (v13) with the Ch1-safe networks (train_ch1safe.sh):
#   1. input caches of the split (shared with v13: they depend on the inputs only)
#   2. 12 psi networks and 8 functional heads (40-epoch schedule, runs/safe_e40_*) and 3 sequence heads predict the split
#   3. psi map = mean of the 12 psi networks; per-frame scorer functionals of the mean map and the 12 members
#   4. q95 / betaN "R3c-lite": the 6 MLP (plans/prod_hm6.txt) and 6 sequence scalar networks of v13 on this map + the psi
#      ensemble's own scalars + the q profile of the map, fixed blend (no variant-D networks)
#   5. calibration of the map, rule c4 (the 8 heads as targets, the sequence heads as alternates)
#   6. submission file
# Map-dependent intermediate files carry the key <split>_ch1safe, so the recipes can share one $FEQ_CACHE.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
SPLIT=$1; K=${SPLIT}_ch1safe
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; P=$FEQ_CACHE/pred/ch1safe/$SPLIT; W=$FEQ_WORKERS
Q=$ML/ch1q; C=$FEQ_CACHE/ml/ch1q; NNC=$FEQ_CACHE/nn/$SPLIT
NSHOT=$(ls $FEQ_ROOT/challenge_data/data/$SPLIT/*.parquet | wc -l)
step() { echo "== $(date -u +%FT%TZ) $*"; }
done_dir() { [ -d "$1" ] && [ "$(ls "$1" | grep -c '\.npz$')" -ge "$NSHOT" ]; }
mkdir -p $P
step "Ch1-safe, split $SPLIT: $NSHOT shots"

step "1. input caches"
done_dir $FEQ_CACHE/nn/$SPLIT || (cd $FEQ_ROOT && $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/$SPLIT \
    --out $FEQ_CACHE/nn/$SPLIT --workers $W --no-targets)
done_dir $FEQ_CACHE/nn/px/$SPLIT || (cd $NN && $FEQ_PY_STARTER preprocess_px.py --split $SPLIT --workers $W)
done_dir $FEQ_CACHE/ml/ch1h/hist/$SPLIT || (cd $ML/ch1h && $FEQ_PY hist_features.py --split $SPLIT --workers $W)

step "2. networks"
PSI=""; HEADS=""; SEQ=""
for R in ch1h/runs/safe_e40_psi_s{1,2,3,4,5,6,7,8,9,10,11,12} ch1c/runs/safe_e40_scog_s{1,2,3} ch1c/runs/safe_e40_auxg_s{1,2} \
         ch1c/runs/safe_e40_scog_ctx_s{1,2,3} ch1h/runs/prod_seqh_s{1,2,3}; do
  N=$(basename $R)
  done_dir $P/$N || $FEQ_PY $ML/final/infer_c1h.py --run $ML/$R --cache $NNC --px $FEQ_CACHE/nn/px/$SPLIT \
      --hist $FEQ_CACHE/ml/ch1h/hist/$SPLIT --out $P/$N
  case $R in ch1h/runs/safe_e40_psi_*) PSI="$PSI$P/$N ";; ch1c/*) HEADS="$HEADS$P/$N ";; *) SEQ="$SEQ$P/$N ";; esac
done

step "3. mean map and frame functionals"
done_dir $P/map || (cd $ML/ch1 && $FEQ_PY integrate.py --out $P/map --split -1 --psi $PSI)
[ -f $P/frames.npz ] || (cd $ML/ch1k && $FEQ_PY mfunc.py --preds $P/map $PSI --out $P/frames.npz --workers $W)

step "4. q95 / betaN (R3c-lite)"
GH=$P/map
cd $Q; mkdir -p runs $C/hist_ch1h
[ -f $C/inp_rfull_$SPLIT.npz ] || $FEQ_PY export_inputs_q.py --split $SPLIT
[ -f $C/hist_$SPLIT.npz ] || $FEQ_PY hist_feats.py --split $SPLIT --workers $W
[ -f $C/hist_ch1h/$SPLIT/meta.json ] || $FEQ_PY hist_features_ch1h.py --split $SPLIT --workers $W
[ -f $C/hist_ch1hq_$SPLIT.npz ] || $FEQ_PY hist_from_ch1h.py --split $SPLIT --with-q --src $C/hist_ch1h
[ -f $C/qc_${K}_gh.npz ] || $FEQ_PY qcurve_any.py --preds $GH --cache $NNC --out $C/qc_${K}_gh.npz --workers $W
BASE="--targets q95,betaN,xb --pos --opt muon --muon-wd 0.1 --epochs 40"
sq_pred() {   # plan file -> predict each network on the split, print q:nf_<key>_<name> specs
  local f=$1 name args
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$FEQ_CACHE}
    done_dir runs/nf_${K}_$name/preds_test || $FEQ_PY train_sq.py $BASE $args --load runs/$name/model.pt \
        --test-inp $C/inp_rfull_$SPLIT.npz --test-hist $C/hist_ch1hq_$SPLIT.npz --test-qc $C/qc_${K}_gh.npz \
        --test-ens $GH --out runs/nf_${K}_$name >&2
    printf ",q:nf_${K}_$name"
  done < $f
}
QQ=$(sq_pred plans/prod_hm6.txt); QQ=${QQ#,}
SS=""; for i in 1 2 3 4 5 6; do SS="$SS$(sq_pred plans/prod_seq_$i.txt)"; done; SS=${SS#,}
done_dir runs/nf_${K}_R3clite/preds_test || $FEQ_PY fin_test.py --out runs/nf_${K}_R3clite --test-cache $NNC --ens $GH \
    --qc $C/qc_${K}_gh.npz --q $QQ --b $QQ --x $QQ --q3 $SS --b3 $SS --w3q 0.3 --w3b 0.5 --wq 0.1,0.5,0.4 --wb 0.3,0.7 --e-bn-range 0,6
SCAL=$Q/runs/nf_${K}_R3clite/preds_test

step "5. calibration (rule c4)"
if ! done_dir $P/final; then
  cd $ML/ch1k
  $FEQ_PY targets.py --frames $P/frames.npz --cache $NNC --heads $HEADS --heads-alt $SEQ --rule rules/c4.json --out $P/targets
  rm -rf $P/final.tmp
  $FEQ_PY fchain.py --preds $P/map --targets $P/targets --out $P/final.tmp --steps k,a,l --scal $SCAL --workers $W --retry --anchor --max-axis-mm 30
  mv $P/final.tmp $P/final
fi

step "6. submission file"
mkdir -p $P/sub
cd $ML/ch1 && $FEQ_PY build_sub_ch1.py --preds $P/final --d3d-only --no-validate --config $SPLIT --out $P/sub
if [ "${FEQ_CLEAN:-0}" = 1 ]; then   # optional: drop the per-network predictions (about 3 GB each), keep map / final / sub
  rm -rf $PSI $HEADS $SEQ
  step "removed the per-network predictions (FEQ_CLEAN=1)"
fi
step "done: $P/sub/$SPLIT.npz"
