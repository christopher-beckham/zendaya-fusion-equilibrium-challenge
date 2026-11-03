#!/usr/bin/env bash
# Training entrypoint: retrains every network of one submission from scratch, one after another, on one GPU. Run
# prepare.sh first (the CPU preprocessing); the DIII-D training scripts stop at once if it has not run.
#
#   bash train.sh A      # submission A (Challenge 1): the 52 networks of DIII-D recipe v13          (20.9 GPU-hours on an A100)
#   bash train.sh B      # submission B (Challenge 2): the 12 networks of DIII-D recipe Ch1-min,       (17.8 GPU-hours on an A100)
#                        #   the 15 training runs behind Chris's 8 MAST models, the 6 training runs of r1
#
# A: zara/diii-d/train_ch1.sh. Its foreground steps are the network trainings; the CPU follow-up of its 5 out-of-fold networks
#    runs in the background, with no GPU visible, while the GPU trains networks that do not read it.
# B: zara/diii-d/train_ch1min.sh, then chris/mast/train.sh with the runs of blend/weights_chris8r1.json, then
#    zara/mast/r1/scripts/train_r1.sh full into $R1_OUT.
# Settings: settings.sh (source it first). At the end it prints the time of each part: the DIII-D steps come from
# $FEQ_CACHE/train_times*.tsv (each step marked CPU or GPU), the MAST parts are the wall-clock of their training runs.
# Each step is skipped when its output exists, so an interrupted run resumes. Step by step: REPRODUCE.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
for v in WORK FEQ_CACHE PY FEC_CACHE FEC_DATA_ROOT R1_ROOT R1_OUT R1_PY; do   # every folder and interpreter comes from settings.sh
  [ -n "${!v:-}" ] || { echo "$v is not set: source settings.sh first" >&2; exit 2; }
done
if [ -e $HERE/.shipped_weights ] || [ -e $FEQ_CACHE/.shipped_weights ]; then
  echo "This copy (or its \$FEQ_CACHE) holds the shipped weights (steps/4_shipped_weights.sh): training here would skip the DIII-D"
  echo "networks that are already there. Use another copy of the repository and another WORK folder (REPRODUCE.md)."; exit 3
fi
touch $HERE/.retrained $FEQ_CACHE/.retrained
SUMMARY=""

diiid() {   # diiid <recipe script> <step-time file>: train, then add its GPU and CPU step times to the summary
  local tsv=$FEQ_CACHE/$2 n0=0
  [ -f $tsv ] && n0=$(wc -l < $tsv)
  bash $HERE/zara/diii-d/$1
  SUMMARY+=$(awk -F'\t' -v n0=$n0 -v name=$1 'NR > n0 && NR > 1 { s[$4] += $2 }
    END { printf "  %-34s GPU steps %6.2f h   CPU steps in the background %5.2f h\n", name, s["GPU"] / 3600, s["CPU"] / 3600 }' $tsv)$'\n'
}
timed() {   # timed <label> <command ...>: run, and add its wall-clock to the summary
  local label=$1 t0=$(date +%s); shift
  "$@"
  SUMMARY+=$(awk -v s=$(( $(date +%s) - t0 )) -v name="$label" 'BEGIN { printf "  %-34s GPU training %6.2f h (wall-clock)", name, s / 3600 }')$'\n'
}

train_A() {
  echo "[$(date -u +%T)] submission A: DIII-D recipe v13"
  diiid train_ch1.sh train_times.tsv
}

train_B() {
  local WEIGHTS=$HERE/blend/weights_chris8r1.json
  echo "[$(date -u +%T)] submission B: DIII-D recipe Ch1-min, MAST blend $(basename $WEIGHTS)"
  diiid train_ch1min.sh train_times_ch1min.tsv
  local RUNS
  RUNS=$(${PY:-python} - $WEIGHTS $HERE/chris/mast/models.json <<'PY'
import json, sys
w, m = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
models = sorted({k for q in ("maps", "q95", "betaN") for k, v in w[q].items() if v > 0})
runs = []
for n in [k for k in models if k != "r1"]:     # the runs behind each model (r1 is trained by zara/mast/r1)
    for src in (n, m[n].get("q95_from")):
        if src:
            runs += [r for r, _ in m[src]["members"]]
print(" ".join(dict.fromkeys(runs)))
PY
)
  echo "[$(date -u +%T)] MAST runs: $RUNS"
  timed "chris/mast/train.sh (15 runs)" bash $HERE/chris/mast/train.sh $RUNS
  echo "[$(date -u +%T)] r1: base model, w3_ft and the four members"
  timed "r1 train_r1.sh full (6 runs)" bash $HERE/zara/mast/r1/scripts/train_r1.sh full $R1_OUT $R1_PY
}

case "${1:-}" in
  A) train_A ;;
  B) train_B ;;
  *) echo "usage: bash train.sh A|B" >&2; exit 2 ;;
esac
echo "[$(date -u +%T)] submission $1: training done. Time per part (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)):"
printf "%s" "$SUMMARY"
