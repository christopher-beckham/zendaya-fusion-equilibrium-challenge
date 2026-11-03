#!/usr/bin/env bash
# 2026-10-15: extra variants on Zara's recipe v6 (gen4_set1500v6, realistic coil inputs), beside run_new_set.sh's
# 4 seeds of Z9's data + v6 merged (scr-mastZ15-v6mix-seed0..3 on GPUs 0-3). Z9 recipe, one run per GPU:
#   V6W-seed0/1  Z9's data (20k + 5k + v4) at weight 4 plus v6 as its own set at weight 1 (v6 sampled far more often)
#   V6R          20k + 5k + v6 merged (v6 in place of v4)
#   V6O          v6 only
# usage: GPUS="4 5 6 7" bash run_v6_extra.sh
set -u
WT=/fsx/chris.beckham/github/fec-multimachine; MM=$WT/my_experiments/nn; R=$MM/runs
C=/fsx/chris.beckham/fusion_cache/synth; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
read -ra GPUS <<< "${GPUS:-4 5 6 7}"
export FEC_MAST_MASK=limiter
M=zara_g4_20k5kv6
[ -f $C/${M}_limiter/shape_targets.npz ] || ( cd $MM && CUDA_VISIBLE_DEVICES= $PY synth_merge.py ${M}_limiter zara_g4_20000_limiter \
    zara_g4_5000_limiter zara_g4_v6_limiter 2>&1 | tail -1 )
ntr() { $PY -c "import json;print(json.load(open('$C/$1_limiter/manifest.json'))['split_counts']['train'])"; }
sb() { echo $(( (640 * $1 + 79149) / 158299 )); }      # ~2 passes per epoch, as for Z9 (640 at 158,299 frames)
SBR=$(sb $(ntr $M)); SBO=$(sb $(ntr zara_g4_v6))
echo "v6-for-v4 batch $SBR, v6-only batch $SBO"
BASE="--mast-mask limiter --epochs 20 --score-every 3 --synthS-every 3 --synthS-sets g4_20000 --synthS-metric g4_20000 \
--mast-checks-every 0 --lindec-ip-floor 150 --lindec-exact-ip-mast --history-frames 1 --thomson-noise 0.6 \
--exclude-shot d3d_shot_955c35c962 --ema-halflife 3 --ema-start 5 --width 1024 --d3d-floor 0.85 --d3d-weight 0.25 \
--synth zara_v1_d3d:1.0 --synth-shape grad=1.0,jtor=0.2"
NAMES=(scr-V6W-z9data-v6w1-seed0 scr-V6W-z9data-v6w1-seed1 scr-V6R-20k5kv6 scr-V6O-v6only)
ARGS=("--synth zara_g4_20k5kv4:4.0 --synth zara_g4_v6:1.0 --synth-batch 640 --seed 0"
      "--synth zara_g4_20k5kv4:4.0 --synth zara_g4_v6:1.0 --synth-batch 640 --seed 1"
      "--synth $M:4.0 --synth-batch $SBR --seed 2"
      "--synth zara_g4_v6:4.0 --synth-batch $SBO --seed 3")
for i in 0 1 2 3; do n=${NAMES[$i]}; g=${GPUS[$((i % ${#GPUS[@]}))]}; mkdir -p $R/$n
  ( cd $WT && CUDA_VISIBLE_DEVICES=$g $PY my_experiments/nn/train_mm.py --name $n $BASE ${ARGS[$i]} >> $R/$n/train.log 2>&1 ) &
  echo "$n on GPU $g: ${ARGS[$i]}"; done
wait; echo done
