#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_thh.parquet
until grep -q "DONE tht_h1" logs/th.log; do sleep 10; done
CH=/fsx/chris.beckham/gsolve_data/tune3/cond_model_thh.pkl
./run_d3dchk.sh thh_h1 knobs_d3d_h1.json $CH 4 4
