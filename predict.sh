#!/usr/bin/env bash
# Inference entrypoint: regenerates one submission's prediction files on the Final inputs, from the shipped (or retrained)
# weights.
#
#   bash predict.sh A OUT_DIR [--rebuilt]     # OUT_DIR/diii_d_private_test.npz (DIII-D recipe v13)
#   bash predict.sh B OUT_DIR [--rebuilt]     # OUT_DIR/diii_d_private_test.npz (Ch1-min) and OUT_DIR/mast_private_test.npz
#
# Without --rebuilt: the shipped weights (steps/4_shipped_weights.sh). With --rebuilt: the networks train.sh trained.
# Settings: settings.sh (source it first). DIII-D: zara/diii-d/predict_ch1.sh or predict_ch1min.sh. MAST (B only): each of
# Chris's models through chris/mast/predict_model.py on the Final inputs ($DATA/final, $MAST_FINAL_CACHE; $FEC_WEIGHTS, or
# retrain.py's chris/mast/multimachine/my_experiments/nn/runs with --rebuilt), r1 through
# zara/mast/r1/predict/predict_r1.sh ($R1_WEIGHTS/runs, or $R1_OUT with --rebuilt), then blend/run_blend.py.
# For B, PARTS="diii-d mast" by default; either part alone regenerates only its file.
# Runs the organisers' validator on every file written and prints its SHA256. Step by step: REPRODUCE.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUB=${1:-}; [[ "$SUB" == A || "$SUB" == B ]] || { echo "usage: bash predict.sh A|B OUT_DIR [--rebuilt]" >&2; exit 2; }
shift
for v in WORK FEQ_CACHE DATA PY MAST_FINAL_CACHE FEC_WEIGHTS R1_ROOT R1_WEIGHTS R1_OUT; do   # every folder and interpreter comes from settings.sh
  [ -n "${!v:-}" ] || { echo "$v is not set: source settings.sh first" >&2; exit 2; }
done
source $HERE/zara/diii-d/env.sh
mkdir -p "${1:?output folder}"; OUT=$(cd "$1" && pwd); shift
REBUILT=""; [ "${1:-}" = "--rebuilt" ] && REBUILT=--rebuilt
if [ -n "$REBUILT" ] && { [ -e $HERE/.shipped_weights ] || [ -e $FEQ_CACHE/.shipped_weights ]; }; then
  echo "--rebuilt, but this copy holds the shipped weights (steps/4_shipped_weights.sh). Use the copy that ran train.sh."; exit 3
fi
if [ -z "$REBUILT" ] && [ ! -e $HERE/.shipped_weights ]; then
  echo "No shipped weights in this copy: run bash steps/4_shipped_weights.sh $SUB first, or add --rebuilt for the retrained networks."; exit 3
fi

predict_A() {
  echo "[$(date -u +%T)] submission A: DIII-D recipe v13"
  [ -n "$REBUILT" ] || bash $HERE/zara/diii-d/prepare_inference.sh   # shipped weights: training cache + artifacts check
  bash $HERE/zara/diii-d/predict_ch1.sh diii_d_private_test
  cp $FEQ_CACHE/pred/diii_d_private_test/sub/diii_d_private_test.npz $OUT/diii_d_private_test.npz
}

predict_B() {
  local WEIGHTS=$HERE/blend/weights_chris8r1.json PARTS=${PARTS:-diii-d mast}
  if [[ " $PARTS " == *" diii-d "* ]]; then
    echo "[$(date -u +%T)] submission B: DIII-D recipe Ch1-min"
    [ -n "$REBUILT" ] || bash $HERE/zara/diii-d/prepare_inference.sh   # shipped weights: training cache + artifacts check
    bash $HERE/zara/diii-d/predict_ch1min.sh diii_d_private_test
    cp $FEQ_CACHE/pred/ch1min/diii_d_private_test/sub/diii_d_private_test.npz $OUT/diii_d_private_test.npz
  fi
  if [[ " $PARTS " == *" mast "* ]]; then
    export FEC_DATA_ROOT=$DATA/final FEC_CACHE=$MAST_FINAL_CACHE   # the Final MAST shots and their inputs (steps/3_mast_data.sh)
    local R1_RUNS=$R1_WEIGHTS/runs; [ -n "$REBUILT" ] && R1_RUNS=$R1_OUT
    local MAST_RUNS=$FEC_WEIGHTS; [ -n "$REBUILT" ] && MAST_RUNS=$HERE/chris/mast/multimachine/my_experiments/nn/runs   # retrain.py's output
    local MODELS
    MODELS=$(${PY:-python} -c "import json; w = json.load(open('$WEIGHTS')); print(' '.join(sorted({k for q in ('maps', 'q95', 'betaN') for k, v in w[q].items() if v > 0})))")
    echo "[$(date -u +%T)] submission B: MAST blend $(basename $WEIGHTS) of $MODELS"
    for m in ${MODELS/r1/}; do
      echo "[$(date -u +%T)] MAST model $m"
      ${PY:-python} -u $HERE/chris/mast/predict_model.py $m --split mast_private_test --out $OUT/members --weights $MAST_RUNS $REBUILT
    done
    echo "[$(date -u +%T)] MAST model r1"
    bash $HERE/zara/mast/r1/predict/predict_r1.sh $OUT/members $R1_RUNS
    ${PY:-python} $HERE/blend/run_blend.py --weights $WEIGHTS --members $OUT/members --split mast_private_test \
        --out $OUT/mast_private_test.npz
  fi
}

predict_$SUB
for f in $OUT/*_private_test.npz; do
  c=$(basename $f .npz)
  (cd $HERE/zara/diii-d && $FEQ_PY_STARTER validate_submission.py $f --config $c | tail -2)
done
sha256sum $OUT/*_private_test.npz
echo "[$(date -u +%T)] submission $SUB: files in $OUT"
