#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
for s in 0 1 2 3; do
  OMP_NUM_THREADS=3 ./venv/bin/python gen_d3d.py --frames d3d/val_th.parquet --knobs knobs_d3d_h1_s075.json --cond $C --variants 8 --keep 8 \
    --out d3dchk/selv75_$s.npz --shard $s/4 > logs/d3dchk_selv75_$s.log 2>&1 &
done; wait; echo DONE selv75
