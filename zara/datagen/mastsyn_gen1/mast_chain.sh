#!/bin/bash
# usage: slurm/mast_chain.sh <gen_train_job> <gen_val_job>
#   convert raw -> MASTSYN / MASTVAL records, first-pass caches (6 + 1 GPU shards), then three training runs (fork upegs_fz).
set -euo pipefail
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2
GT=$1; GV=$2
M2=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2
FZ=$M2/upegs_fz
jc1=$(sbatch --parsable --dependency=afterok:$GT --cpus-per-task=32 --mem=120G -J m2-conv-syn slurm/cpu_pin.sbatch fusion-eq-gpu $M2/gen python convert_m2.py --pool train --machine MASTSYN --workers 32)
jc2=$(sbatch --parsable --dependency=afterok:$GV --cpus-per-task=16 --mem=60G -J m2-conv-val slurm/cpu_pin.sbatch fusion-eq-gpu $M2/gen python convert_m2.py --pool val --machine MASTVAL --workers 16 --truth)
deps=""
for k in 0 1 2 3 4 5; do
  j=$(sbatch --parsable --dependency=afterok:$jc1 -J m2-cache-syn$k $FZ/slurm/gpu_pin.sbatch python build_cache.py --machine MASTSYN --split synth --p-share 0.7 --tag _p7 --shard $k --n-shards 6)
  deps="$deps:$j"
done
jv=$(sbatch --parsable --dependency=afterok:$jc2 -J m2-cache-val $FZ/slurm/gpu_pin.sbatch python build_cache.py --machine MASTVAL --split synth --p-share 0.7 --tag _p7 --check)
echo "convert $jc1 $jc2; caches $deps val $jv"
U4=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/upegs/runs/u4_all/model.pt
COMMON="--warm 2 --train-tag _s2p7 --steps 4 --batch 256 --frames-per-epoch 400000 --phys gs,ampere,ts,tsedge,kin --w-phys 0.3 --aux harm --w-aux 0.5 --w-func 3 --w-band 2 --width 512 --depth 2 --harm-scale 0.05 --multi-tag _p7 --log-every 200"
ALL=JET,EAST,CMOD,KSTAR,NSTXU_FWD,NSTX_FWD,MASTSYN
cd $FZ
DEP="afterok${deps}:$jv" bash slurm/run.sh m2_a --epochs 28 --lr 7e-4 --multi $ALL --mix 0.5 $COMMON
DEP="afterok${deps}:$jv" bash slurm/run.sh m2_b --epochs 8 --lr 2e-4 --init $U4 --multi $ALL --mix 0.7 $COMMON
DEP="afterok${deps}:$jv" bash slurm/run.sh m2_c --epochs 28 --lr 7e-4 --multi MASTSYN --mix 1.0 $COMMON
