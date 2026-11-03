#!/bin/bash
cd /fsx/chris.beckham/gsolve_data/tune3
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
until grep -q "DONE cond_ridge" logs/d3d_extra.log; do sleep 30; done
./run_d3dchk.sh cond_lkm knobs_d3d_lkm.json $C
./run_d3dchk.sh cond_lip knobs_d3d_lip.json $C
