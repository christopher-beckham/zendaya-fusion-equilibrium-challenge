#!/bin/bash
# build_ft_caches.sh OUT: the four fine-tune synthetic MAST caches on the CPU (6 shards x 8 threads each), Zara's settings.
W=/fsx/chris.beckham/zara_speedup; FM=$W/starter/my_experiments; O=$1
for M in MASTG4 MASTG4RT MASTG4V3 MASTG4V3RT; do
  for k in 0 1 2 3 4 5; do
    INTR_WALL=0.20 INTR_FP=divert unshare -rm bash $W/scripts/cpu_cache_job.sh $O/ft $FM/ch2/g4/fz 8 --machine $M --split synth \
      --sol-factor 0.5 --p-share 0.7 --tag _wd --shard $k --n-shards 6 > $O/ft_${M}_$k.log 2>&1 &
  done
  wait; echo "$(date -u +%T) $M done: $(ls $O/ft/$(echo $M | tr A-Z a-z)_synth_wd | wc -l) files"
done
