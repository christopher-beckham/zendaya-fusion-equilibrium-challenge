#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/fsx/chris.beckham/gsolve_data/tune/venv/bin/python
$PY d3d_table3.py --split train --n 700 --per-shot 16 --workers 5 --psi > logs/table3_train.log 2>&1 &
$PY d3d_table3.py --split val --n 250 --per-shot 8 --workers 3 > logs/table3_val.log 2>&1 &
wait
echo TABLES3 DONE
