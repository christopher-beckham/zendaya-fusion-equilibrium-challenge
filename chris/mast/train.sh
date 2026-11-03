#!/usr/bin/env bash
# Retrain every training run behind Chris's 13 MAST models, one after another on one GPU.
#
#   FEC_DATA_ROOT=... FEC_CACHE=... bash chris/mast/train.sh [RUN ...]
#
# FEC_CACHE must hold the derived caches (build_caches.sh) and synth/ (data/fetch.py --root $FEC_CACHE).
# Each run is retrain.py RUN: the recorded arguments, stopped at the epoch its checkpoint came from, no scoring jobs.
# Output: multimachine/my_experiments/nn/runs/RUN-rebuild/last.pt (predict.sh --rebuilt reads these).
# With no RUN given, all 30 runs in models.json. Measured on one RTX PRO 6000: 17.9 h for all 30 (checks/rebuilds).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${FEC_CACHE:?set FEC_CACHE (settings.sh)}" "${FEC_DATA_ROOT:?set FEC_DATA_ROOT (settings.sh)}"
PY=${PY:-python}
if [ $# -eq 0 ]; then
  set -- $($PY -c "import json; m = json.load(open('$HERE/models.json')); print(' '.join(dict.fromkeys(r for k, v in m.items() if not k.startswith('_') for r, _ in v['members'])))")
fi
for run in "$@"; do
  stop=""
  [ -f "$HERE/run_records/$run/best_mast.json" ] || stop="--stop-epoch 30"
  echo "[$(date -u +%T)] $run"
  $PY -u "$HERE/retrain.py" "$run" $stop
done
echo "[$(date -u +%T)] done"
