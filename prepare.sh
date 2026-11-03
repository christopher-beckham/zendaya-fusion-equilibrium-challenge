#!/usr/bin/env bash
# Preprocessing entrypoint: the CPU work that comes before training. No GPU is visible to any step, and none of it counts
# toward the 24 GPU-hours (rule 6 asks for it to be declared with its hardware and wall-clock: SUBMISSION_A.md, SUBMISSION_B.md).
# It can run on a machine without a GPU, as long as that machine writes to the same $FEQ_CACHE that train.sh then reads.
#
#   bash prepare.sh A    # DIII-D recipe v13: zara/diii-d/build_caches.sh ch1      (about 5.5 h on 12 cores)
#   bash prepare.sh B    # DIII-D recipe Ch1-min: zara/diii-d/build_caches.sh ch1min (about 4.8 h on 12 cores)
#
# What it builds: per-shot input caches of the DIII-D shots, label-side training targets, the C1 physics + Ridge base model, and
# the q95 / betaN inputs that need no trained network (zara/diii-d/build_caches.sh lists every step). B's MAST data is prepared
# by steps/3_mast_data.sh. Settings: settings.sh (source it first). Step times: $FEQ_CACHE/train_times*.tsv, marked CPU.
# Each step is skipped when its output exists, so an interrupted run resumes.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
for v in WORK FEQ_CACHE; do   # every folder and interpreter comes from settings.sh
  [ -n "${!v:-}" ] || { echo "$v is not set: source settings.sh first" >&2; exit 2; }
done
if [ -e $HERE/.shipped_weights ] || [ -e $FEQ_CACHE/.shipped_weights ]; then
  echo "This copy (or its \$FEQ_CACHE) holds the shipped weights (steps/4_shipped_weights.sh): training here would skip the DIII-D"
  echo "networks that are already there. Use another copy of the repository and another WORK folder (REPRODUCE.md)."; exit 3
fi
mkdir -p $FEQ_CACHE && touch $HERE/.retrained $FEQ_CACHE/.retrained
export CUDA_VISIBLE_DEVICES=
case "${1:-}" in
  A) REC=ch1 ;;
  B) REC=ch1min ;;
  *) echo "usage: bash prepare.sh A|B" >&2; exit 2 ;;
esac
t0=$(date +%s)
echo "[$(date -u +%T)] submission $1: DIII-D CPU preprocessing ($REC), no GPU visible"
bash $HERE/zara/diii-d/build_caches.sh $REC
echo "[$(date -u +%T)] submission $1: preprocessing done in $(( ($(date +%s) - t0) / 60 )) min (CPU only); next: bash train.sh $1"
