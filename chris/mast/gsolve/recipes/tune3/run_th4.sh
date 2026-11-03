#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_thh.parquet
until grep -q "DONE thh_h1_s05" logs/th3.log; do sleep 10; done
CH=/fsx/chris.beckham/gsolve_data/tune3/cond_model_thh.pkl
./run_d3dchk.sh thh_h1_esd knobs_d3d_h1_esd.json $CH 4 4
./run_d3dchk.sh thh_h1_so3 knobs_d3d_h1_so3.json $CH 4 4
