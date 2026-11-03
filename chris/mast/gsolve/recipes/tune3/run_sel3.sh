#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
for v in lim lip; do
for s in 0 1 2 3; do
  OMP_NUM_THREADS=2 ./venv/bin/python gen_d3d.py --frames d3d/val_th.parquet --knobs knobs_d3d_h1_s075_$v.json --cond $C --variants 4 --keep 4 \
    --out d3dchk/sel75$v\_$s.npz --shard $s/4 > logs/d3dchk_sel75$v\_$s.log 2>&1 &
done; wait; echo DONE sel75$v
done
