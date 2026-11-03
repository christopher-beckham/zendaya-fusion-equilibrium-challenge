#!/usr/bin/env bash
# 2026-10-17: Zara's recipe v7 (gen4_set1500v7 = v6 + per-frame li from our DIII-D-taught model). Internal test only,
# nothing uploaded or submitted. Cache + practice set g4_v7, then (Z15 recipe, one run per GPU, GPU 4 is Zara's):
#   V7A s0, s1   Z15's data + v7 merged (20k + 5k + v4 + v6 + v7)
#   V7B s0, s1   Z15's data with v7 in place of v6 (20k + 5k + v4 + v7)
#   V7C          v7 only
# then practice scores (g4_20000, g4_v4, g4_v6, g4_v7) for each family and for Z15's members, and built files (no upload)
# for the label-free consistency screen.
set -u
ZDIR=/fsx/zara.ercan/more_data/m2/gen4_set1500v7; PRE=mast_g4s1500v7
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter; NN=$REPO/my_experiments/nn; SS=$NN/synth_S
WT=/fsx/chris.beckham/github/fec-multimachine; MM=$WT/my_experiments/nn; R=$MM/runs; S=$REPO/submissions
C=/fsx/chris.beckham/fusion_cache/synth; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
export FEC_MAST_MASK=limiter
log() { echo "[$(date -u +%H:%M)] $*"; }
NEWC=zara_g4_v7
if [ ! -f $C/${NEWC}_limiter/shape_targets.npz ]; then
  ( cd $MM && CUDA_VISIBLE_DEVICES= $PY -u synth_cache_seq.py --src $ZDIR/${PRE}_train --heldout-src $ZDIR/${PRE}_heldout \
      --tag ${NEWC}_limiter --workers 24 > $C/build_${NEWC}.log 2>&1 \
    && CUDA_VISIBLE_DEVICES= $PY -u synth_shape_targets.py $NEWC --workers 24 >> $C/build_${NEWC}.log 2>&1 ) \
    || { log "CACHE FAILED"; tail -5 $C/build_${NEWC}.log; exit 1; }
fi
grep -v "^  " $C/build_${NEWC}.log | tail -n 3
$PY - <<PYEOF
import json; from pathlib import Path
p = Path("$SS/extra_sets.json"); d = json.loads(p.read_text())
d["g4_v7"] = {"limiter_cache": "${NEWC}_limiter", "parquet": "$ZDIR/${PRE}_heldout"}
p.write_text(json.dumps(d, indent=1)); print("registered g4_v7")
PYEOF
( cd $SS && CUDA_VISIBLE_DEVICES= $PY build_sets.py subset g4_v7 2>&1 | tail -1 \
  && CUDA_VISIBLE_DEVICES= SYNTHS_WORKERS=24 $PY scorer.py truth g4_v7 2>&1 | tail -1 ) || { log "SET FAILED"; exit 1; }
merge() { [ -f $C/$1_limiter/shape_targets.npz ] || ( cd $MM && CUDA_VISIBLE_DEVICES= $PY synth_merge.py $1_limiter "${@:2}" 2>&1 | tail -1 ); }
MA=zara_g4_mixv6v7; MB=zara_g4_mixv7
merge $MA zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter zara_g4_v6_limiter ${NEWC}_limiter &
merge $MB zara_g4_20000_limiter zara_g4_5000_limiter zara_g4_5000v4_limiter ${NEWC}_limiter &
wait
ntr() { $PY -c "import json;print(json.load(open('$C/$1_limiter/manifest.json'))['split_counts']['train'])"; }
sb() { echo $(( (640 * $1 + 79149) / 158299 )); }
SA=$(sb $(ntr $MA)); SBB=$(sb $(ntr $MB)); SC=$(sb $(ntr $NEWC)); log "batches $SA $SBB $SC"
BASE="--mast-mask limiter --epochs 20 --score-every 3 --synthS-every 3 --synthS-sets g4_20000 --synthS-metric g4_20000 \
--mast-checks-every 0 --lindec-ip-floor 150 --lindec-exact-ip-mast --history-frames 1 --thomson-noise 0.6 \
--exclude-shot d3d_shot_955c35c962 --ema-halflife 3 --ema-start 5 --width 1024 --d3d-floor 0.85 --d3d-weight 0.25 \
--synth zara_v1_d3d:1.0 --synth-shape grad=1.0,jtor=0.2"
train() { local n=$1 g=$2; shift 2; mkdir -p $R/$n
  ( cd $WT && CUDA_VISIBLE_DEVICES=$g $PY my_experiments/nn/train_mm.py --name $n $BASE "$@" >> $R/$n/train.log 2>&1 ) &
  log "$n on GPU $g: $*"; }
A1=scr-V7A-z15v7-s0; A2=scr-V7A-z15v7-s1; B1=scr-V7B-v7forv6-s0; B2=scr-V7B-v7forv6-s1; C1=scr-V7C-v7only
train $A1 2 --synth $MA:4.0 --synth-batch $SA --seed 0
train $A2 3 --synth $MA:4.0 --synth-batch $SA --seed 1
train $B1 5 --synth $MB:4.0 --synth-batch $SBB --seed 0
train $B2 6 --synth $MB:4.0 --synth-batch $SBB --seed 1
train $C1 7 --synth $NEWC:4.0 --synth-batch $SC --seed 2
wait; log "training done"
for n in $A1 $A2 $B1 $B2 $C1; do
  log "$n best $($PY -c "import json;d=json.load(open('$R/$n/best_mast.json'));print(d['epoch'],round(d['synthS'],4),'d3d',round(d['d3d_S'],4))" 2>&1 | tail -1)"; done
b() { for n in "$@"; do printf "%s " $R/$n/best_mast.pt; done; }
Z15=$(for s in 0 1 2 3; do printf "%s " $R/scr-mastZ15-v6mix-seed$s/best_mast.pt; done)
( cd $NN && GPUS="2 3" UPLOAD=0 bash build_qb.sh mastV7A-mm-Z15dataPlusV7-x2 $(b $A1 $A2) > $S/build_V7A.log 2>&1; log "V7A: $(tail -1 $S/build_V7A.log)" ) &
( cd $NN && GPUS="5 6" UPLOAD=0 bash build_qb.sh mastV7B-mm-V7forV6-x2 $(b $B1 $B2) > $S/build_V7B.log 2>&1; log "V7B: $(tail -1 $S/build_V7B.log)" ) &
( cd $NN && GPUS="7" UPLOAD=0 bash build_qb.sh mastV7C-mm-V7only $(b $C1) > $S/build_V7C.log 2>&1; log "V7C: $(tail -1 $S/build_V7C.log)" ) &
( cd $SS; export CUDA_VISIBLE_DEVICES=; SETS=g4_20000,g4_v4,g4_v6,g4_v7
  sc() { nice $PY -u ensemble_synth.py --sets $SETS --threads 4 --workers 6 "$@" 2>&1 | grep "RESULT.* all S"; }
  ( sc V7_A $(b $A1 $A2) ) & ( sc V7_B $(b $B1 $B2) ) & ( sc V7_C $(b $C1) ) & ( sc V7_Z15 $Z15 ) &
  ( sc V7_Z15x2 $R/scr-mastZ15-v6mix-seed0/best_mast.pt $R/scr-mastZ15-v6mix-seed1/best_mast.pt ) &
  ( sc V7_FA $(b scr-F1-z15mix-s0 scr-F2-z15mix-s1 scr-F6-z15mix-s2) ) &
  wait ) &
wait; log "all done"
