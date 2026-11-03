#!/bin/bash
# honest (Thomson-edge) DIII-D replays, one at a time (4 shards x 4 threads)
cd /fsx/chris.beckham/gsolve_data/tune3
export FRAMES=d3d/val_th.parquet
until grep -q "CAL DONE" logs/cal.log; do sleep 10; done
C=/fsx/chris.beckham/gsolve_data/tune3/cond_model.pkl; CT=/fsx/chris.beckham/gsolve_data/tune3/cond_model_th.pkl
./run_d3dchk.sh thc knobs_x8.json $C 4 4
./run_d3dchk.sh thc_h1 knobs_d3d_h1.json $C 4 4
./run_d3dchk.sh tht knobs_x8.json $CT 4 4
./run_d3dchk.sh tht_h1 knobs_d3d_h1.json $CT 4 4
