#!/bin/bash
# train_overlay.sh "MYDIR=ZARAPATH;..." CODE_FZ OUT ARGS_JSON [key=value overrides...]
# One training run (Zara's recorded args.json, our code copy) with our CPU-built caches mounted over Zara's cache paths and
# the rest of her data read-only. Call through unshare -rm. INTR_WALL passed through.
set -uo pipefail
MAP=$1; FZ=$2; OUT=$3; AJ=$4; shift 4
W=/fsx/chris.beckham/zara_speedup; PY=${PY:-/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python}
ARGS=$($PY $W/scripts/args_to_cli.py $FZ/train.py $AJ "$@") || exit 5
mkdir -p $OUT; echo "args: $ARGS"; echo "overlay: $MAP"
export OMP_NUM_THREADS=4 CUDA_MEM_CAP_GB=80
T0=$(date +%s)
bash $W/scripts/overlay_run.sh "$MAP" bash -c "cd $FZ && $PY $W/starter/my_experiments/ml/final/capmem.py train.py --out $OUT $ARGS"
echo "exit $? wall $(( $(date +%s) - T0 )) s"
