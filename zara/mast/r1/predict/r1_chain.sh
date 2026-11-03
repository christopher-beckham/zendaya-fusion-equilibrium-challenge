#!/bin/bash
# usage: r1_chain.sh <split key> <input cache dir> <out dir> <first-pass cache dir> <split parquet dir> <member runs dir>
# r1's prediction chain on one MAST split, steps in sequence (the commands of the 13 October Development run, Q95=learned):
#   1. selector rt_ft at 11 inboard solenoid-shaped corrections b -> phase report of every frame (GPU, then CPU)
#   2. select_b: per frame the smallest |b| whose LCFS stays off the centre column (+ add_bv 0) (CPU)
#   3. 4 members (rt_ft rt_mix rt_ftw3 rt_nonoise) with their per-frame b (GPU) -> mean -> pp2 -> pp_guard -> final (CPU)
#   4. submission-format file <out dir>/r1_mast.npz
# Settings: R1_PY (fusion-eq-gpu environment), R1_PY_STARTER (fusion-equilibrium-starter environment), R1_WORKERS (16).
# Each step is skipped when its output exists, so an interrupted run resumes.
set -euo pipefail
K=$1; SRC=$2; C=$3; FP=$4; PQ=$5; R=$6
RD=$(cd "$(dirname "$0")" && pwd); ME=$(cd $RD/../starter/my_experiments && pwd); G4=$ME/ch2/g4; FZ=$G4/fz; L=$G4/lbfit
GPUPY=${R1_PY:-python}; STPY=${R1_PY_STARTER:-python}; NW=${R1_WORKERS:-16}
BS="0 -1 -2 -3 -4 -6 -8 -10 -13 -16 -20"; PA="--no-harm --betaN kinfill"
export MAST_SPLIT_CACHE=$SRC PYTHONUNBUFFERED=1
step() { echo "== $(date -u +%FT%TZ) $*"; }
pred() { (cd $FZ && OMP_NUM_THREADS=4 INTR_WALL=0.20 $GPUPY $L/split_run_any.py $FZ $K $SRC predict.py "$@"); }
mkdir -p $C
step "split $K: $(ls $FP | grep -c '\.npz$') shots in the first-pass cache $FP; members from $R"
for b in $BS; do
  OUT=$C/r1_g_b$(echo $b | tr '-' 'm')
  [ -f $OUT/rep1.npy ] && continue
  step "grid b=$b"
  pred --run $R/rt_ft --machine MAST --split $K --sol-factor 0.5 --tag _iv_full_wdiv --out $OUT $PA --batch 128 --add-field sol=$b | tail -1
  (cd $RD && OMP_NUM_THREADS=$NW CUDA_VISIBLE_DEVICES= $GPUPY $RD/phase_report.py $OUT --every 1 --workers $NW --save $OUT/rep1.npy | tail -1)
done
ls $FP | grep npz | sed 's/.npz//' | sort > $C/r1_stems.txt
step "select_b"
(cd $RD && $STPY $RD/select_b.py MAST $C r1_g $C/r1_stems.txt $C/sel_r1.npz $BS && $STPY $G4/diag/add_bv.py $C/sel_r1.npz 0)
P=""
for m in rt_ft rt_mix rt_ftw3 rt_nonoise; do
  OUT=$C/r1_m_$m; P="$P $OUT/preds"
  [ "$(ls $OUT/preds 2>/dev/null | grep -c '\.npz$')" -ge "$(wc -l < $C/r1_stems.txt)" ] && continue
  step "member $m"
  pred --run $R/$m --machine MAST --split $K --sol-factor 0.5 --tag _iv_full_wdiv --out $OUT $PA --batch 128 --add-field npz:$C/sel_r1.npz | tail -1
done
E=$C/r1_ens
step "ensemble + pp2"
(cd $FZ && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $GPUPY ensemble.py --out $E --preds $P && \
 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $GPUPY postproc_m2.py --src $E --dst ${E}_pp2 --machine MAST --window 3 --outlier 0.15 --kappa-max 2.6 \
   --vol-range 3 12 --fp-cache $FP --fp-res-max 0.05 --flux-jump 3 --workers 8)
step "pp_guard + final (learned q95 / betaN)"
(cd $RD && OMP_NUM_THREADS=$NW CUDA_VISIBLE_DEVICES= $GPUPY $RD/pp_guard.py $E ${E}_pp2 ${E}_pp2g --workers $NW)
mkdir -p ${E}_final && rm -rf ${E}_final/preds && cp -r ${E}_pp2g/preds ${E}_final/preds
step "submission-format file"
$GPUPY $RD/write_mast.py --parquet $PQ --preds ${E}_final/preds --out $C/r1_mast.npz
step "done: $C/r1_mast.npz"
