#!/bin/bash
# Run predict_model.py for the remaining models on mast_public_test (two GPUs), then compare every output with its
# original file byte for byte.
set -u
P=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python; O=/fsx/chris.beckham/fec_upload/step2/pred_public
S=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions; T=/fsx/chris.beckham/fusion_cache/team_blend_fetch
ZC=/fsx/zara.ercan/cache/ch2/g4/chris_subs
cd /fsx/chris.beckham/github/fec-final
worker() {   # GPU CORES MODEL...
  local g=$1 c=$2; shift 2
  for m in "$@"; do
    [ -s $O/$m/mast_public_test.npz ] && continue
    CUDA_VISIBLE_DEVICES=$g taskset -c $c $P -u chris/mast/predict_model.py $m --out $O --workers 6 > $O/$m.log 2>&1
    echo "EXIT $?" >> $O/$m.log
  done
}
worker 2 36-41 Z9qb Z15 Z2 Z8 V7C Z6b &
worker 3 42-47 Z18 FA Z3 Z7 Z17 &
wait
declare -A ORIG=(
 [Z]=$T/Z [Z2]=$T/Z2 [Z6b]=$T/Z6b [Z7]=$T/Z7 [Z3]=$ZC/Z3 [Z8]=$ZC/Z8
 [Z9qb]=$S/mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1 [Z15]=$S/mastZ15-mm-v6mix-x4qb-q95cc0932-bnV2-limiter-c1
 [Z17]=$S/mastZ17-mm-C3-condgsolveonlyqb-q95cc0932-bnV2-limiter-c1 [Z18]=$S/mastZ18-mm-Z9dataPlusCond-x3qb-q95cc0932-bnV2-limiter-c1
 [FA]=$S/mastFA-mm-Z15dataPlusgsolve_cond3_1000-x3qb-q95cc0932-bnV2-limiter-c1 [FC]=$S/mastFC-mm-gsolve_cond3_1000onlyqb-q95cc0932-bnV2-limiter-c1
 [V7C]=$S/mastV7C-mm-V7onlyqb-q95cc0932-bnV2-limiter-c1 )
declare -A BASE=(
 [Z15]=$S/mastZ15-mm-v6mix-x4-base [Z17]=$S/mastZ17-mm-C3-condgsolveonly-base [Z18]=$S/mastZ18-mm-Z9dataPlusCond-x3-base
 [FA]=$S/mastFA-mm-Z15dataPlusgsolve_cond3_1000-x3-base [FC]=$S/mastFC-mm-gsolve_cond3_1000only-base [V7C]=$S/mastV7C-mm-V7only-base )
for m in Z Z2 Z3 Z6b Z7 Z8 Z9qb Z15 Z17 Z18 FA FC V7C; do
  r="final $(cmp -s $O/$m/mast_public_test.npz ${ORIG[$m]}/mast_public_test.npz && echo IDENTICAL || echo DIFFERENT)"
  [ -n "${BASE[$m]:-}" ] && r="$r, base $(cmp -s $O/$m/steps/base/mast_public_test.npz ${BASE[$m]}/mast_public_test.npz && echo IDENTICAL || echo DIFFERENT)"
  printf "%-5s %s   (original: %s)\n" $m "$r" "${ORIG[$m]#/fsx/}"
done
