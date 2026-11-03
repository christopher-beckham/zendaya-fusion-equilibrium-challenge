#!/usr/bin/env bash
# 2026-10-17: train on a tune3 conditional gsolve set (DIII-D flux-map-supervised recipe, gsolve_data/TUNE3_BRIEF.md)
# once its flag and limiter cache exist. Z15/Z18 recipe, one run per GPU (GPU 4 is Zara's, never used):
#   P1, P2, P6  Z15's data (mixv6) + new set merged (seeds 0, 1, 2)    P3  new set only (clean test, like Z17)
#   P4, P7      Z15's data (weight 4) + new set as its own set (weight 1), seeds 3, 4
#   P5          Z9's data (20k5kv4) + new set merged (like Z18), seed 0
# then practice scores and candidates (q95 rule + V2 βN). Nothing is submitted.
# usage: run_tune3_train.sh FLAGFILE GSET PREFIX     e.g. .../tune3/EARLY_READY gsolve_cond2z1000 E
set -u
FLAG=$1; G=$2; P=$3
WT=/fsx/chris.beckham/github/fec-multimachine; MM=$WT/my_experiments/nn; R=$MM/runs
NN=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/my_experiments/nn; SS=$NN/synth_S
S=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions
C=/fsx/chris.beckham/fusion_cache/synth; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
export FEC_MAST_MASK=limiter
log() { echo "[$(date -u +%H:%M)] $*"; }
until [ -f $FLAG ] && [ -f $C/${G}_limiter/shape_targets.npz ]; do sleep 120; done
log "$G ready"
merge() { [ -f $C/$1_limiter/shape_targets.npz ] || ( cd $MM && CUDA_VISIBLE_DEVICES= $PY synth_merge.py $1_limiter "${@:2}" 2>&1 | tail -1 ); }
M1=zara_g4_mixv6_${G}; M2=zara_g4_20k5kv4_${G}
merge $M1 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter zara_g4_v6_limiter ${G}_limiter &
merge $M2 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter ${G}_limiter &
wait
ntr() { $PY -c "import json;print(json.load(open('$C/$1_limiter/manifest.json'))['split_counts']['train'])"; }
sb() { echo $(( (640 * $1 + 79149) / 158299 )); }
S1=$(sb $(ntr $M1)); S2=$(sb $(ntr $M2)); SG=$(sb $(ntr $G)); log "batches $S1 $S2 $SG"
BASE="--mast-mask limiter --epochs 20 --score-every 3 --synthS-every 3 --synthS-sets g4_20000 --synthS-metric g4_20000 \
--mast-checks-every 0 --lindec-ip-floor 150 --lindec-exact-ip-mast --history-frames 1 --thomson-noise 0.6 \
--exclude-shot d3d_shot_955c35c962 --ema-halflife 3 --ema-start 5 --width 1024 --d3d-floor 0.85 --d3d-weight 0.25 \
--synth zara_v1_d3d:1.0 --synth-shape grad=1.0,jtor=0.2"
train() { local n=$1 g=$2; shift 2; mkdir -p $R/$n
  ( cd $WT && CUDA_VISIBLE_DEVICES=$g $PY my_experiments/nn/train_mm.py --name $n $BASE "$@" >> $R/$n/train.log 2>&1 ) &
  log "$n on GPU $g: $*"; }
N1=scr-${P}1-z15mix-s0; N2=scr-${P}2-z15mix-s1; N3=scr-${P}3-only; N4=scr-${P}4-z15w1-s3
N5=scr-${P}5-z9mix-s0; N6=scr-${P}6-z15mix-s2; N7=scr-${P}7-z15w1-s4
train $N1 0 --synth $M1:4.0 --synth-batch $S1 --seed 0
train $N2 1 --synth $M1:4.0 --synth-batch $S1 --seed 1
train $N3 2 --synth $G:4.0 --synth-batch $SG --seed 2
train $N4 3 --synth zara_g4_mixv6:4.0 --synth $G:1.0 --synth-batch 680 --seed 3
train $N5 5 --synth $M2:4.0 --synth-batch $S2 --seed 0
train $N6 6 --synth $M1:4.0 --synth-batch $S1 --seed 2
train $N7 7 --synth zara_g4_mixv6:4.0 --synth $G:1.0 --synth-batch 680 --seed 4
wait; log "training done"
for n in $N1 $N2 $N3 $N4 $N5 $N6 $N7; do
  log "$n best $($PY -c "import json;d=json.load(open('$R/$n/best_mast.json'));print(d['epoch'],round(d['synthS'],4),'d3d',round(d['d3d_S'],4))" 2>&1 | tail -1)"; done
b() { for n in "$@"; do printf "%s " $R/$n/best_mast.pt; done; }
( cd $NN && GPUS="0 1 2" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mast${P}A-mm-Z15dataPlus${G}-x3 $(b $N1 $N2 $N6) > $S/build_${P}A.log 2>&1; log "${P}A: $(tail -1 $S/build_${P}A.log)" ) &
( cd $NN && GPUS="3 5" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mast${P}B-mm-Z15dataW1${G}-x2 $(b $N4 $N7) > $S/build_${P}B.log 2>&1; log "${P}B: $(tail -1 $S/build_${P}B.log)" ) &
( cd $NN && GPUS="6" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mast${P}C-mm-${G}only $(b $N3) > $S/build_${P}C.log 2>&1; log "${P}C: $(tail -1 $S/build_${P}C.log)" ) &
( cd $NN && GPUS="7" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mast${P}D-mm-Z9dataPlus${G}-x1 $(b $N5) > $S/build_${P}D.log 2>&1; log "${P}D: $(tail -1 $S/build_${P}D.log)" ) &
( cd $SS; export CUDA_VISIBLE_DEVICES=; SETS=g4_20000,g4_v4,g4_v6
  sc() { nice $PY -u ensemble_synth.py --sets $SETS --threads 4 --workers 6 "$@" 2>&1 | grep "RESULT.* all S"; }
  ( sc ${P}_A $(b $N1 $N2 $N6) ) & ( sc ${P}_B $(b $N4 $N7) ) & ( sc ${P}_C $(b $N3) ) & ( sc ${P}_D $(b $N5) ) &
  wait ) &
wait; log "all done"
