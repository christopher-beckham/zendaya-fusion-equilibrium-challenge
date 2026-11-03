#!/bin/bash
# d3d_s2p7_remote.sh: the DIII-D training first-pass cache (_s2p7) on the CPU, 10 shards x 4 threads, skipping shots already built.
W=/fsx/chris.beckham/zara_speedup; O=$W/out/cpucache/full
echo "host $(hostname) start $(date -u +%T)"
for k in $(seq 0 9); do
  unshare -rm bash $W/scripts/cpu_cache_job.sh $O/pegs $W/starter/my_experiments/pegs 4 --machine DIII-D --split train --subset train \
    --stride 2 --p-share 0.7 --tag _s2p7 --shard $k --n-shards 10 > $O/d3d_s2p7b_$k.log 2>&1 &
done
wait; echo "end $(date -u +%T): $(ls $O/pegs/diiid_train_s2p7 | wc -l) files"
