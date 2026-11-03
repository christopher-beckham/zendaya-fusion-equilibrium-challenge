#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
./run_d3dchk.sh cond_th0 knobs_d3d_th0.json $C
./run_d3dchk.sh cond_r1 knobs_x8.json "$C --seed 1"
./run_d3dchk.sh cond_s05 knobs_d3d_s05.json $C
./run_d3dchk.sh cond_ridge knobs_d3d_ridge.json $C
