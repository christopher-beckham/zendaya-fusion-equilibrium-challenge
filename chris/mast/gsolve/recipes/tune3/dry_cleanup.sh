#!/bin/bash
# remove a dry-run build: dry_cleanup.sh NAME TAG SET
NAME=$1; TAG=$2; SET=$3
[[ "$NAME" == cdry* ]] || { echo "refusing: $NAME"; exit 1; }
rm -rf /fsx/chris.beckham/fusion_cache/synth_S_scratch/truth/${SET}.pkl /fsx/chris.beckham/gsolve_data/$NAME /fsx/chris.beckham/fusion_cache/synth/${TAG}_limiter /fsx/chris.beckham/fusion_cache/synth_S_scratch/sets/${SET}_ho_limiter
/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python - <<P
import json
p='/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/my_experiments/nn/synth_S/extra_sets.json'
d=json.load(open(p)); d.pop("$SET", None); open(p,'w').write(json.dumps(d,indent=1))
P
