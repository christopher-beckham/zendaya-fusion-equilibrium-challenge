#!/bin/bash
# finish_caches.sh: the remaining r1 caches on the CPU, after the base, fine-tune and DIII-D training caches.
#  1. real-MAST public-test first pass (_iv_full_wdiv), the source of fix_time.py's time features
#  2. fix_time.py on our four fine-tune caches, reading our test cache
#  3. DIII-D holdout first pass (_p7), then the auxiliary targets (_s2p7, _p7) from our DIII-D caches
set -uo pipefail
W=/fsx/chris.beckham/zara_speedup; O=$W/out/cpucache/full; FM=$W/starter/my_experiments; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
Z=/fsx/zara.ercan/cache
until grep -q "MASTG4V3RT done" $O/ft_driver.log; do sleep 30; done
until grep -q "^end" $O/d3d_remote.log 2>/dev/null; do sleep 30; done      # DIII-D training cache runs on job 10479
echo "$(date -u +%T) step 1: MAST test cache"
for k in 0 1 2 3 4 5; do INTR_WALL=0.20 INTR_FP=divert unshare -rm bash $W/scripts/cpu_cache_job.sh $O/mtest $FM/ch2/g4/fz 8 --machine MAST --split test \
  --sol-factor 0.5 --p-share 0.7 --tag _iv_full_wdiv --shard $k --n-shards 6 > $O/mtest_$k.log 2>&1 & done; wait
echo "$(date -u +%T) step 2: fix_time ($(ls $O/mtest/* | wc -l) test files)"
$PY - <<'PY'
import glob, os, sys, numpy as np
sys.path.insert(0, "/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4")
import fix_time as FT
O = "/fsx/chris.beckham/zara_speedup/out/cpucache/full"
FT.RC = glob.glob(f"{O}/mtest/*")[0]                       # our CPU-built real-MAST test cache instead of Zara's
for M in ("MASTG4", "MASTG4RT", "MASTG4V3", "MASTG4V3RT"):
    src = f"/fsx/zara.ercan/cache/ch2/g4/multi/{M}/src"; nm = nt = 0
    for f in sorted(glob.glob(f"{O}/ft/{M.lower()}_synth_wd/*.npz")):
        a, b = FT.fix(f, str(np.load(os.path.join(src, os.path.basename(f)))["origin"])); nm += a; nt += b
    print(M, f"{nm}/{nt} frames got the real-shot time features", flush=True)
PY
echo "$(date -u +%T) step 3: DIII-D holdout first pass"
for k in 0 1 2 3; do unshare -rm bash $W/scripts/cpu_cache_job.sh $O/pegs $FM/pegs 12 --machine DIII-D --split train --subset holdout \
  --p-share 0.7 --tag _p7 --shard $k --n-shards 4 > $O/d3d_p7_$k.log 2>&1 & done; wait
echo "$(date -u +%T) step 3b: auxiliary targets (reading our DIII-D caches through the overlay)"
MAP="$O/pegs/diiid_train_s2p7=$Z/pegs/diiid_train_s2p7;$O/pegs/diiid_train_p7=$Z/pegs/diiid_train_p7;$O/upegs/aux_d3d_s2p7=$Z/upegs/aux_d3d_s2p7;$O/upegs/aux_d3d_p7=$Z/upegs/aux_d3d_p7"
for k in 0 1 2 3 4 5; do (cd $FM/ch2/g4/fz && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 unshare -rm bash $W/scripts/overlay_run.sh "$MAP" $PY build_aux.py --tag _s2p7 --shard $k --n-shards 6) > $O/aux_s2p7_$k.log 2>&1 & done; wait
(cd $FM/ch2/g4/fz && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 unshare -rm bash $W/scripts/overlay_run.sh "$MAP" $PY build_aux.py --tag _p7 --subset holdout) > $O/aux_p7.log 2>&1
echo "$(date -u +%T) all done"; for d in base/* ft/* pegs/* mtest/* upegs/*; do echo "  $d: $(ls $O/$d | wc -l) files"; done
