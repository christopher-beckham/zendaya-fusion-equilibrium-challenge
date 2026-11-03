#!/bin/bash
# Challenge 1 (DIII-D) inference entrypoint: the submitted v13 pipeline on any inputs-only DIII-D split.
#
#   bash predict_ch1.sh <split>        e.g. diii_d_private_test (raw parquet files in challenge_data/data/<split>/)
#
# Output: $FEQ_CACHE/pred/<split>/sub/<split>.npz (shot_XXXX_psirz / _q95 / _betaN, shots in sorted parquet order, psi float32)
# and the per-shot predictions $FEQ_CACHE/pred/<split>/final/<stem>.npz. Steps (one GPU for step 2 and the R3c networks, CPU
# otherwise; every step is deterministic, finished steps are skipped on a rerun):
#   1. input caches of the split: c1/nn per-shot inputs, Group A physics inputs, Group H long-history inputs
#   2. the 23 trained networks (my_experiments/ml/final/lists/v13_networks.txt) predict the split (ml/final/infer_c1h.py)
#   3. psi map = mean of the 12 Group H psi networks; per-frame scorer functionals of the mean map and the 12 members
#   4. q95 / betaN: recipe R3c (ml/final/r3c_predict.sh; 24 scalar networks + q profile of the map + fixed blend)
#   5. calibration of the map, rule c4 (ml/ch1k targets.py + fchain.py: gated elongation stretch about the X-point, axis
#      shift, li calibration with the li range guard), q95 / betaN from step 4
#   6. submission file (ml/ch1/build_sub_ch1.py --config <split>)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
SPLIT=$1
E=$FEQ_ROOT/my_experiments; ML=$E/ml; NN=$E/c1/nn; P=$FEQ_CACHE/pred/$SPLIT; W=$FEQ_WORKERS
NSHOT=$(ls $FEQ_ROOT/challenge_data/data/$SPLIT/*.parquet | wc -l)
step() { echo "== $(date -u +%FT%TZ) $*"; }
done_dir() { [ -d "$1" ] && [ "$(ls "$1" | grep -c '\.npz$')" -ge "$NSHOT" ]; }
mkdir -p $P
step "split $SPLIT: $NSHOT shots"

step "1. input caches"
done_dir $FEQ_CACHE/nn/$SPLIT || (cd $FEQ_ROOT && $FEQ_PY_STARTER $NN/preprocess.py --data-dir challenge_data/data/$SPLIT \
    --out $FEQ_CACHE/nn/$SPLIT --workers $W --no-targets)
done_dir $FEQ_CACHE/nn/px/$SPLIT || (cd $NN && $FEQ_PY_STARTER preprocess_px.py --split $SPLIT --workers $W)
done_dir $FEQ_CACHE/ml/ch1h/hist/$SPLIT || (cd $ML/ch1h && $FEQ_PY hist_features.py --split $SPLIT --workers $W)

step "2. networks"
for R in $(cat $ML/final/lists/v13_networks.txt); do
  N=$(basename $R)
  done_dir $P/$N || $FEQ_PY $ML/final/infer_c1h.py --run $ML/$R --cache $FEQ_CACHE/nn/$SPLIT --px $FEQ_CACHE/nn/px/$SPLIT \
      --hist $FEQ_CACHE/ml/ch1h/hist/$SPLIT --out $P/$N
done
PSI=$(grep -E "prod_auxghc" $ML/final/lists/v13_networks.txt | while read R; do printf "%s " $P/$(basename $R); done)
HEADS=$(grep -E "ch1c/runs" $ML/final/lists/v13_networks.txt | while read R; do printf "%s " $P/$(basename $R); done)
SEQ=$(grep -E "prod_seqh" $ML/final/lists/v13_networks.txt | while read R; do printf "%s " $P/$(basename $R); done)

step "3. mean map and frame functionals"
done_dir $P/map || (cd $ML/ch1 && $FEQ_PY integrate.py --out $P/map --split -1 --psi $PSI)
[ -f $P/frames.npz ] || (cd $ML/ch1k && $FEQ_PY mfunc.py --preds $P/map $PSI --out $P/frames.npz --workers $W)

step "4. q95 / betaN (R3c)"
R3C=$ML/ch1q/runs/nf_${SPLIT}_R3c/preds_test
done_dir $R3C || bash $ML/final/r3c_predict.sh $SPLIT $P/map

step "5. calibration (rule c4)"
if ! done_dir $P/final; then
  cd $ML/ch1k
  $FEQ_PY targets.py --frames $P/frames.npz --cache $FEQ_CACHE/nn/$SPLIT --heads $HEADS --heads-alt $SEQ --rule rules/c4.json --out $P/targets
  rm -rf $P/final
  $FEQ_PY fchain.py --preds $P/map --targets $P/targets --out $P/final --steps k,a,l --scal $R3C --workers $W --retry --anchor --max-axis-mm 30
fi

step "6. submission file"
mkdir -p $P/sub
cd $ML/ch1 && $FEQ_PY build_sub_ch1.py --preds $P/final --d3d-only --no-validate --config $SPLIT --out $P/sub
if [ "${FEQ_CLEAN:-0}" = 1 ]; then   # optional: drop the per-network predictions (about 3 GB each), keep map / final / sub
  for R in $(cat $ML/final/lists/v13_networks.txt); do rm -rf $P/$(basename $R); done
  step "removed the per-network predictions (FEQ_CLEAN=1)"
fi
step "done: $P/sub/$SPLIT.npz"
