#!/bin/bash
# critic-guided selection: 8-variant replays (keep all accepted) on a TRAIN subset and on VAL, incumbent+h1 recipe
cd /fsx/chris.beckham/gsolve_data/tune3
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
until ! pgrep -f "knobs_d3d_h1_s075" > /dev/null; do sleep 10; done
for s in 0 1 2 3; do
  OMP_NUM_THREADS=3 ./venv/bin/python gen_d3d.py --frames d3d/train_sel.parquet --knobs knobs_d3d_h1.json --cond $C --variants 8 --keep 8 \
    --out d3dchk/seltr_$s.npz --shard $s/4 --seed 3 > logs/d3dchk_seltr_$s.log 2>&1 &
done; wait; echo DONE seltr
for s in 0 1 2 3; do
  OMP_NUM_THREADS=3 ./venv/bin/python gen_d3d.py --frames d3d/val_th.parquet --knobs knobs_d3d_h1.json --cond $C --variants 8 --keep 8 \
    --out d3dchk/selva_$s.npz --shard $s/4 > logs/d3dchk_selva_$s.log 2>&1 &
done; wait; echo DONE selva
export FRAMES=d3d/val_th.parquet
for v in s05 lip lim esd; do ./run_d3dchk.sh thc_h1_$v knobs_d3d_h1_$v.json $C 4 3; done
