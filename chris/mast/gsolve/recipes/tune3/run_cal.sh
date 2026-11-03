#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
for s in 0 1 2 3; do
  OMP_NUM_THREADS=4 ./venv/bin/python gen_d3d.py --frames d3d/train_cal.parquet --knobs knobs_x8.json --cond cond_model.pkl \
     --out d3dchk/cal_$s.npz --shard $s/4 > logs/d3dchk_cal_$s.log 2>&1 &
done
wait
OMP_NUM_THREADS=1 ./venv/bin/python ecoil_cal.py fit cal
echo CAL DONE
