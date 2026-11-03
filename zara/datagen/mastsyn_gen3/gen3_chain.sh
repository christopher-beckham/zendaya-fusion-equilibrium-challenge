#!/bin/bash
# usage: slurm/gen3_chain.sh <gen3 job>  -> gen-3 records appended to MASTSYN (prefix mastsyn3), caches for the new stems,
#   m2_g = m2_b fine-tuned 8 more epochs on gen1+gen3, m2_h = from scratch (mixed, 28 ep) on gen1+gen3; real-MAST nt inference behind each.
set -euo pipefail
M2=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2; FZ=$M2/upegs_fz
G3=$1
jc=$(sbatch --parsable --dependency=afterany:$G3 --cpus-per-task=32 --mem=120G -J m2-conv-syn3 $M2/slurm/cpu_pin.sbatch fusion-eq-gpu $M2/gen python convert_m2.py --pool train --machine MASTSYN --raw-name mast_train3 --prefix mastsyn3 --copies 2 --workers 32)
deps=""
for k in 0 1 2 3 4 5; do j=$(sbatch --parsable --dependency=afterok:$jc -J m2-cache-syn3-$k $FZ/slurm/gpu_pin.sbatch python build_cache.py --machine MASTSYN --split synth --p-share 0.7 --tag _p7 --shard $k --n-shards 6); deps="$deps:$j"; done
echo "convert $jc; caches $deps"
COMMON="--warm 1 --train-tag _s2p7 --steps 4 --batch 256 --frames-per-epoch 400000 --phys gs,ampere,ts,tsedge,kin --w-phys 0.3 --aux harm --w-aux 0.5 --w-func 3 --w-band 2 --width 512 --depth 2 --harm-scale 0.05 --multi-tag _p7 --log-every 200"
ALL=JET,EAST,CMOD,KSTAR,NSTXU_FWD,NSTX_FWD,MASTSYN
cd $FZ
DEP="afterok${deps}" bash slurm/run.sh m2_g --epochs 8 --lr 2e-4 --init $FZ/runs/m2_b/model.pt --multi $ALL --mix 0.7 $COMMON
DEP="afterok${deps}" bash slurm/run.sh m2_h --epochs 28 --lr 7e-4 --warm 2 --multi $ALL --mix 0.5 $COMMON
cd $M2
TG=$(squeue -u zara.ercan -h -o "%i %j" | grep "upegs-tr-m2_g" | cut -d' ' -f1); TH=$(squeue -u zara.ercan -h -o "%i %j" | grep "upegs-tr-m2_h" | cut -d' ' -f1)
DEP=afterok:$TG bash slurm/mast_infer.sh runs/m2_g nt
DEP=afterok:$TH bash slurm/mast_infer.sh runs/m2_h nt
