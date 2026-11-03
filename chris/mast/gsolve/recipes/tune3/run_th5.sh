#!/bin/bash
# spread / edge / li-offset variants on the incumbent teacher + height rule (honest judge)
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_th.parquet
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl
for v in s075 s05 esd so3 lip lim; do ./run_d3dchk.sh thc_h1_$v knobs_d3d_h1_$v.json $C 4 4; done
