#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_thh.parquet
until grep -q "DONE thh_h1" logs/th2.log; do sleep 10; done
CH=/fsx/chris.beckham/gsolve_data/tune3/cond_model_thh.pkl
./run_d3dchk.sh thh_h1_s075 knobs_d3d_h1_s075.json $CH 4 4
./run_d3dchk.sh thh_h1_s05 knobs_d3d_h1_s05.json $CH 4 4
