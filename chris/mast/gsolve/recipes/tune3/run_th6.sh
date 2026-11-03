#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_th.parquet
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
until grep -q "DONE thc_h1_lim" logs/th5.log; do sleep 10; done
for v in lkp lkm; do ./run_d3dchk.sh thc_h1_$v knobs_d3d_h1_$v.json $C 4 4; done
