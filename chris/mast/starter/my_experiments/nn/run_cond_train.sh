#!/usr/bin/env bash
# 2026-10-16: train on the DIII-D-taught conditional gsolve set (gsolve_cond1000) once /fsx/chris.beckham/gsolve_data/
# tune2/COND_READY exists. Z9 recipe, one run per GPU:
#   C1, C2  Z9's data + cond set merged (seeds 0, 1)      C3  cond set only (the clean test, like T4)
#   C4, C5  Z9's data + v6 + cond set merged (seeds 0, 1)  C6  Z9's data (weight 4) + cond set as its own set (weight 1)
# then practice scores and the Challenge 1 candidates (q95 rule + V2 βN) for C3 alone and C1+C2+C6. Nothing is submitted.
set -u
WT=/fsx/chris.beckham/github/fec-multimachine; MM=$WT/my_experiments/nn; R=$MM/runs
NN=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/my_experiments/nn; SS=$NN/synth_S
S=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions
C=/fsx/chris.beckham/fusion_cache/synth; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python; G=gsolve_cond1000
export FEC_MAST_MASK=limiter
log() { echo "[$(date -u +%H:%M)] $*"; }
until [ -f /fsx/chris.beckham/gsolve_data/tune2/COND_READY ] && [ -f $C/${G}_limiter/shape_targets.npz ]; do sleep 120; done
log "cond set ready"
merge() { [ -f $C/$1_limiter/shape_targets.npz ] || ( cd $MM && CUDA_VISIBLE_DEVICES= $PY synth_merge.py $1_limiter "${@:2}" 2>&1 | tail -1 ); }
M1=zara_g4_20k5kv4cond; M2=zara_g4_20k5kv4v6cond
merge $M1 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter ${G}_limiter &
merge $M2 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter zara_g4_v6_limiter ${G}_limiter &
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
train scr-C1-z9cond-seed0 0 --synth $M1:4.0 --synth-batch $S1 --seed 0
train scr-C2-z9cond-seed1 1 --synth $M1:4.0 --synth-batch $S1 --seed 1
train scr-C3-condonly 2 --synth $G:4.0 --synth-batch $SG --seed 2
train scr-C4-z9v6cond-seed0 3 --synth $M2:4.0 --synth-batch $S2 --seed 0
train scr-C5-z9v6cond-seed1 4 --synth $M2:4.0 --synth-batch $S2 --seed 1
train scr-C6-z9-condw1 5 --synth zara_g4_20k5kv4:4.0 --synth $G:1.0 --synth-batch 640 --seed 3
wait; log "training done"
for n in scr-C1-z9cond-seed0 scr-C2-z9cond-seed1 scr-C3-condonly scr-C4-z9v6cond-seed0 scr-C5-z9v6cond-seed1 scr-C6-z9-condw1; do
  log "$n best $($PY -c "import json;d=json.load(open('$R/$n/best_mast.json'));print(d['epoch'],round(d['synthS'],4),'d3d',round(d['d3d_S'],4))" 2>&1 | tail -1)"; done
b() { for n in "$@"; do printf "%s " $R/$n/best_mast.pt; done; }
# candidates (GPU) and practice scores (CPU) in parallel
( cd $NN && GPUS="6" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mastZ17-mm-C3-condgsolveonly $(b scr-C3-condonly) > $S/build_Z17.log 2>&1; log "Z17: $(tail -1 $S/build_Z17.log)" ) &
( cd $NN && GPUS="7 0 1" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mastZ18-mm-Z9dataPlusCond-x3 $(b scr-C1-z9cond-seed0 scr-C2-z9cond-seed1 scr-C6-z9-condw1) > $S/build_Z18.log 2>&1; log "Z18: $(tail -1 $S/build_Z18.log)" ) &
( cd $SS; export CUDA_VISIBLE_DEVICES=; SETS=g4_20000,g4_v4,g4_v6; grep -q '"g4_gsc1000"' extra_sets.json && SETS=$SETS,g4_gsc1000
  sc() { nice $PY -u ensemble_synth.py --sets $SETS --threads 4 --workers 6 "$@" 2>&1 | grep "RESULT.* all S"; }
  ( sc CN_C126 $(b scr-C1-z9cond-seed0 scr-C2-z9cond-seed1 scr-C6-z9-condw1) ) &
  ( sc CN_C45 $(b scr-C4-z9v6cond-seed0 scr-C5-z9v6cond-seed1) ) &
  ( sc CN_C3 $(b scr-C3-condonly) ) &
  wait ) &
wait; log "all done"
