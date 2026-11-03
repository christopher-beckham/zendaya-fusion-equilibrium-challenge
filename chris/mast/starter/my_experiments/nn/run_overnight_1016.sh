#!/usr/bin/env bash
# Overnight 2026-10-15/16 (deadline 09:00 UTC). Waits for the critic-guided gsolve search to finish, then:
#   1. (from 03:30 UTC) resume the paused v6 variants on GPUs 4-7; build Z15qb (Z9's data + v6, 4 seeds) on GPUs 0-3
#   2. (when /fsx/chris.beckham/gsolve_data/tune/FINAL_READY exists) merge caches and train 8 runs of the Z9 recipe
#        T1, T2   Z9's data + tuned gsolve merged (seeds 0, 1)
#        T3       Z9's data (weight 4) + tuned gsolve as its own set (weight 1)
#        T4       tuned gsolve only
#        T5-T7    Z9's data + v6 + tuned gsolve merged (seeds 0-2)
#        T8       Z9's data + v6 merged (weight 4) + tuned gsolve as its own set (weight 1)
#   3. practice scores of the ensembles on 20k, v4, v6 and the tuned gsolve held-out set
# Nothing is submitted. usage: nohup bash run_overnight_1016.sh > run_overnight_1016.log 2>&1 &
set -u
WT=/fsx/chris.beckham/github/fec-multimachine; MM=$WT/my_experiments/nn; R=$MM/runs
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter; NN=$REPO/my_experiments/nn; SS=$NN/synth_S
C=/fsx/chris.beckham/fusion_cache/synth; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
TUNE=/fsx/chris.beckham/gsolve_data/tune; G=gsolve_tuned1000
export FEC_MAST_MASK=limiter
log() { echo "[$(date -u +%H:%M)] $*"; }
BASE="--mast-mask limiter --epochs 20 --score-every 3 --synthS-every 3 --synthS-sets g4_20000 --synthS-metric g4_20000 \
--mast-checks-every 0 --lindec-ip-floor 150 --lindec-exact-ip-mast --history-frames 1 --thomson-noise 0.6 \
--exclude-shot d3d_shot_955c35c962 --ema-halflife 3 --ema-start 5 --width 1024 --d3d-floor 0.85 --d3d-weight 0.25 \
--synth zara_v1_d3d:1.0 --synth-shape grad=1.0,jtor=0.2"
train() {  # NAME GPU ARGS... (appends to train.log; --resume if resume.pt exists and the run is unfinished)
  local n=$1 g=$2; shift 2; mkdir -p $R/$n
  ( cd $WT && CUDA_VISIBLE_DEVICES=$g $PY my_experiments/nn/train_mm.py --name $n $BASE "$@" >> $R/$n/train.log 2>&1 ) &
  log "$n on GPU $g: $*"; }
ntr() { $PY -c "import json;print(json.load(open('$C/$1_limiter/manifest.json'))['split_counts']['train'])"; }
sb() { echo $(( (640 * $1 + 79149) / 158299 )); }
merge() { [ -f $C/$1_limiter/shape_targets.npz ] || ( cd $MM && CUDA_VISIBLE_DEVICES= $PY synth_merge.py $1_limiter "${@:2}" 2>&1 | tail -1 ); }

# 1. after the search: resume the v6 variants, build Z15qb
until [ "$(date -u +%H%M)" -ge 0330 ] && [ "$(date -u +%H%M)" -lt 1200 ] || [ -f $TUNE/FINAL_READY ]; do sleep 120; done
log "phase 1"
SB6O=$(sb $(ntr zara_g4_v6)); SB6R=$(sb $(ntr zara_g4_20k5kv6))
train scr-V6W-z9data-v6w1-seed0 4 --synth zara_g4_20k5kv4:4.0 --synth zara_g4_v6:1.0 --synth-batch 640 --seed 0 --resume
train scr-V6W-z9data-v6w1-seed1 5 --synth zara_g4_20k5kv4:4.0 --synth zara_g4_v6:1.0 --synth-batch 640 --seed 1 --resume
train scr-V6R-20k5kv6 6 --synth zara_g4_20k5kv6:4.0 --synth-batch $SB6R --seed 2 --resume
train scr-V6O-v6only 7 --synth zara_g4_v6:4.0 --synth-batch $SB6O --seed 3 --resume
Z15="$R/scr-mastZ15-v6mix-seed0/best_mast.pt $R/scr-mastZ15-v6mix-seed1/best_mast.pt $R/scr-mastZ15-v6mix-seed2/best_mast.pt $R/scr-mastZ15-v6mix-seed3/best_mast.pt"
( cd $NN && GPUS="0 1 2 3" HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash build_qb.sh mastZ15-mm-v6mix-x4 $Z15 > $REPO/submissions/build_Z15.log 2>&1; log "Z15qb built: $(tail -1 $REPO/submissions/build_Z15.log)" ) &
wait; log "phase 1 done"

# 2. tuned gsolve set: merge and train
until [ -f $TUNE/FINAL_READY ] && [ -f $C/${G}_limiter/shape_targets.npz ]; do sleep 120; done
log "phase 2: $(cat $TUNE/FINAL_READY)"
M1=zara_g4_20k5kv4gst; M2=zara_g4_20k5kv4v6gst
merge $M1 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter ${G}_limiter &
merge $M2 zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter zara_g4_v6_limiter ${G}_limiter &
wait
S1=$(sb $(ntr $M1)); S2=$(sb $(ntr $M2)); SG=$(sb $(ntr $G))
log "batches: M1 $S1, M2 $S2, gsolve-only $SG"
train scr-T1-z9gst-seed0 0 --synth $M1:4.0 --synth-batch $S1 --seed 0
train scr-T2-z9gst-seed1 1 --synth $M1:4.0 --synth-batch $S1 --seed 1
train scr-T3-z9-gstw1 2 --synth zara_g4_20k5kv4:4.0 --synth $G:1.0 --synth-batch 640 --seed 2
train scr-T4-gstonly 3 --synth $G:4.0 --synth-batch $SG --seed 3
train scr-T5-z9v6gst-seed0 4 --synth $M2:4.0 --synth-batch $S2 --seed 0
train scr-T6-z9v6gst-seed1 5 --synth $M2:4.0 --synth-batch $S2 --seed 1
train scr-T7-z9v6gst-seed2 6 --synth $M2:4.0 --synth-batch $S2 --seed 2
train scr-T8-z9v6-gstw1 7 --synth zara_g4_mixv6:4.0 --synth $G:1.0 --synth-batch 680 --seed 3
wait; log "phase 2 done"
for n in scr-T1-z9gst-seed0 scr-T2-z9gst-seed1 scr-T3-z9-gstw1 scr-T4-gstonly scr-T5-z9v6gst-seed0 scr-T6-z9v6gst-seed1 \
         scr-T7-z9v6gst-seed2 scr-T8-z9v6-gstw1 scr-V6W-z9data-v6w1-seed0 scr-V6W-z9data-v6w1-seed1 scr-V6R-20k5kv6 scr-V6O-v6only \
         scr-mastZ15-v6mix-seed0 scr-mastZ15-v6mix-seed1 scr-mastZ15-v6mix-seed2 scr-mastZ15-v6mix-seed3; do
  log "$n best $($PY -c "import json;d=json.load(open('$R/$n/best_mast.json'));print(d['epoch'],round(d['synthS'],4),'d3d',round(d['d3d_S'],4))" 2>&1 | tail -1)"; done

# 3. practice scores of the ensembles (CPU)
SETS=g4_20000,g4_v4,g4_v6; grep -q '"g4_gst1000"' $SS/extra_sets.json && SETS=$SETS,g4_gst1000
b() { for n in "$@"; do printf "%s " $R/$n/best_mast.pt; done; }
cd $SS; export CUDA_VISIBLE_DEVICES=
sc() { nice $PY -u ensemble_synth.py --sets $SETS --threads 4 --workers 6 "$@" 2>&1 | grep "RESULT.* all S"; }
( sc ON_Z15 $(b scr-mastZ15-v6mix-seed{0,1,2,3}) ) &
( sc ON_T123 $(b scr-T1-z9gst-seed0 scr-T2-z9gst-seed1 scr-T3-z9-gstw1) ) &
( sc ON_T567 $(b scr-T5-z9v6gst-seed0 scr-T6-z9v6gst-seed1 scr-T7-z9v6gst-seed2) ) &
( sc ON_T5678 $(b scr-T5-z9v6gst-seed0 scr-T6-z9v6gst-seed1 scr-T7-z9v6gst-seed2 scr-T8-z9v6-gstw1) ) &
wait
( sc ON_V6W $(b scr-V6W-z9data-v6w1-seed0 scr-V6W-z9data-v6w1-seed1) ) &
( sc ON_T4 $(b scr-T4-gstonly) ) &
( sc ON_V6O $(b scr-V6O-v6only) ) &
( sc ON_Z9 $(b scr-V2-20k5kv4 scr-W1-20k5kv4-seed1 scr-W2-20k5kv4-seed2 scr-W3-20k5kv4-seed3 scr-W4-20k5kv4-seed4 scr-W5-20k5kv4-seed5 scr-W6-20k5kv4-seed6) ) &
wait
log "all done"
