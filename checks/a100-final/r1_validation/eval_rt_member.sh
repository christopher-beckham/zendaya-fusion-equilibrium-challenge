#!/bin/bash
# eval_rt_member.sh OUT_BASE NAME (RUN_DIR | preds:PRED_DIR) [more members ...]
# Zara's internal validation (ch2/g4/slurm/eval_rt.sh) for one member, run with her unchanged code from her tree and her data
# mounted read-only: predict.py on the real-Thomson held-out split (args "--no-harm --betaN kinfill", as for rt_ft) ->
# ensemble.py -> to_score.py -> score_folder.py (limiter mask, subsets h20k/h5k/h5kv3). Writes OUT_BASE/NAME/score_raw.json.
# Call through unshare -rm. "preds:DIR" skips the prediction and scores existing member predictions.
set -euo pipefail
OB=$1; N=$2; shift 2
PY=${PY:-/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python}
for d in /fsx/zara.ercan /fsx/hackathon_offsite; do mount --bind $d $d && mount -o remount,ro,bind $d; done
touch /fsx/zara.ercan/.rotest 2>/dev/null && { echo "WRITE NOT BLOCKED"; exit 4; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
H=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4
E=/fsx/zara.ercan/cache/ch2/g4/eval
O=$OB/$N; mkdir -p $O
T0=$(date +%s)
P=""
for SRC in "$@"; do                    # several members: predicted one by one, then averaged by ensemble.py (as eval_rt.sh)
  if [[ $SRC == preds:* ]]; then P="$P ${SRC#preds:}"; continue; fi
  M=$OB/members/$(basename $(dirname $SRC))_$(basename $SRC)
  if [ ! -f $M/done ]; then
    cd $H/fz; INTR_WALL=0.20 $PY $H/datarec/run_rt.py predict.py --run $SRC --machine MAST --split g4hrt --sol-factor 0.5 --tag _iv_wd2 \
      --out $M --no-harm --betaN kinfill --batch 128 && touch $M/done
  fi
  P="$P $M/preds"
done
echo "predict done $(( $(date +%s) - T0 )) s"
cd $H/fz; $PY ensemble.py --out $O/raw --preds $P
cd $H; $PY to_score.py $O/raw $O/raw.npz
SUB="h20k=$E/stems_h20k.txt,h5k=$E/stems_h5k.txt,h5kv3=$E/stems_h5kv3.txt"
$PY score_folder.py --data $E/data --machine mastlim --pred $O/raw.npz --subsets $SUB --workers 16 --ref-cache $E/refs_mastlim.pkl --json $O/score_raw.json
echo "S all: $($PY -c "import json;print(json.load(open('$O/score_raw.json'))['all']['S'])")  total $(( $(date +%s) - T0 )) s"
