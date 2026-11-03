#!/bin/bash
# Challenge 1 (DIII-D) inference entrypoint of the Ch1-min recipe (the DIII-D file of the Challenge 2 submission) on any
# inputs-only DIII-D split:
#
#   bash predict_ch1min.sh <split>        e.g. diii_d_private_test (raw parquet files in challenge_data/data/<split>/)
#
# Output: $FEQ_CACHE/pred/ch1min/<split>/sub/<split>.npz (shot_XXXX_psirz / _q95 / _betaN, shots in sorted parquet order, psi
# float32). Same steps as predict_ch1.sh (v13) with the Ch1-min networks (train_ch1min.sh):
#   1. input caches of the split (shared with v13: they depend on the inputs only)
#   2. 6 psi networks (runs/min_e30_psi_s1-6) and 3 sequence heads predict the split (ml/final/infer_c1h.py)
#   3. psi map = mean of the 6 psi networks; per-frame scorer functionals of the mean map and the 6 members
#   4. q95 / betaN "D3": the 3 variant-D networks + the psi ensemble's own scalars + the q profile of the map, fixed weights
#      (q95 0.1 / 0.35 / 0.55, betaN 0.4 / 0.6)
#   5. calibration of the map, rule c4, with the psi networks' own shape outputs as targets (B1) and the sequence heads as
#      alternates
#   6. submission file
# Map-dependent intermediate files carry the key <split>_ch1min, so v13 and Ch1-min can share one $FEQ_CACHE.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
SPLIT=$1; K=${SPLIT}_ch1min
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; P=$FEQ_CACHE/pred/ch1min/$SPLIT; W=$FEQ_WORKERS
Q=$ML/ch1q; C=$FEQ_CACHE/ml/ch1q; NNC=$FEQ_CACHE/nn/$SPLIT
NSHOT=$(ls $FEQ_ROOT/challenge_data/data/$SPLIT/*.parquet | wc -l)
step() { echo "== $(date -u +%FT%TZ) $*"; }
done_dir() { [ -d "$1" ] && [ "$(ls "$1" | grep -c '\.npz$')" -ge "$NSHOT" ]; }
mkdir -p $P
step "Ch1-min, split $SPLIT: $NSHOT shots"

step "1. input caches"
done_dir $FEQ_CACHE/nn/$SPLIT || (cd $FEQ_ROOT && $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/$SPLIT \
    --out $FEQ_CACHE/nn/$SPLIT --workers $W --no-targets)
done_dir $FEQ_CACHE/nn/px/$SPLIT || (cd $NN && $FEQ_PY_STARTER preprocess_px.py --split $SPLIT --workers $W)
done_dir $FEQ_CACHE/ml/ch1h/hist/$SPLIT || (cd $ML/ch1h && $FEQ_PY hist_features.py --split $SPLIT --workers $W)

step "2. networks"
PSI=""; SEQ=""
for R in ch1h/runs/min_e30_psi_s{1,2,3,4,5,6} ch1h/runs/prod_seqh_s{1,2,3}; do
  N=$(basename $R)
  done_dir $P/$N || $FEQ_PY $ML/final/infer_c1h.py --run $ML/$R --cache $NNC --px $FEQ_CACHE/nn/px/$SPLIT \
      --hist $FEQ_CACHE/ml/ch1h/hist/$SPLIT --out $P/$N
  case $N in min_e30_psi_*) PSI="$PSI$P/$N ";; *) SEQ="$SEQ$P/$N ";; esac
done
HEADS=$PSI                                                     # B1: the psi members' own shape outputs

step "3. mean map and frame functionals"
done_dir $P/map || (cd $ML/ch1 && $FEQ_PY integrate.py --out $P/map --split -1 --psi $PSI)
[ -f $P/frames.npz ] || (cd $ML/ch1k && $FEQ_PY mfunc.py --preds $P/map $PSI --out $P/frames.npz --workers $W)

step "4. q95 / betaN (D3)"
GH=$P/map
cd $Q; mkdir -p runs $C/hist_ch1h
[ -f $C/inp_rfull_$SPLIT.npz ] || $FEQ_PY export_inputs_q.py --split $SPLIT
[ -f $C/hist_$SPLIT.npz ] || $FEQ_PY hist_feats.py --split $SPLIT --workers $W
[ -f $C/hist_ch1h/$SPLIT/meta.json ] || $FEQ_PY hist_features_ch1h.py --split $SPLIT --workers $W
[ -f $C/hist_ch1hq_$SPLIT.npz ] || $FEQ_PY hist_from_ch1h.py --split $SPLIT --with-q --src $C/hist_ch1h
[ -f $C/qc_${K}_gh.npz ] || $FEQ_PY qcurve_any.py --preds $GH --cache $NNC --out $C/qc_${K}_gh.npz --workers $W
BASE="--targets q95,betaN,xb --pos --opt muon --muon-wd 0.1 --epochs 40"
DD=""
mapfile -t D3 < <(cat ../ch1s/plans/prodD_{a,b,c}.txt | grep -E "^(prod_xbmix_D|prod_ctx5_D|prod_ctx5sm_D) ")
for line in "${D3[@]}"; do
  read -r name args <<< "$line"
  args=${args//@FEQ_CACHE@/$FEQ_CACHE}
  args=$(echo "$args" | sed "s#--xbmix ,[^ ]*#--xbmix ,$C/qc_${K}_gh.npz#")
  done_dir runs/nf_${K}_$name/preds_test || $FEQ_PY train_sq.py $BASE $args --load ../ch1s/runs/$name/model.pt \
      --test-inp $C/inp_rfull_$SPLIT.npz --out runs/nf_${K}_$name
  DD="$DD,q:nf_${K}_$name"
done
DD=${DD#,}
done_dir runs/nf_${K}_D3/preds_test || $FEQ_PY fin_test.py --out runs/nf_${K}_D3 --test-cache $NNC --ens $GH --qc $C/qc_${K}_gh.npz \
    --q $DD --b $DD --x $DD --wq 0.1,0.35,0.55 --wb 0.4,0.6 --e-bn-range 0,6
SCAL=$Q/runs/nf_${K}_D3/preds_test

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
  rm -rf $PSI $SEQ
  step "removed the per-network predictions (FEQ_CLEAN=1)"
fi
step "done: $P/sub/$SPLIT.npz"
