#!/bin/bash
# cpu_cache_job.sh OUT_ROOT FZ_DIR THREADS <build_cache args...>: build_cache.py on the CPU only (no GPU visible), Zara's
# data read-only, output redirected under OUT_ROOT. Call through unshare -rm.
set -uo pipefail
OUT=$1; FZ=$2; T=$3; shift 3
PY=${PY:-/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python}
for d in /fsx/zara.ercan /fsx/hackathon_offsite; do mount --bind $d $d && mount -o remount,ro,bind $d; done
touch /fsx/zara.ercan/.rotest 2>/dev/null && { echo "WRITE NOT BLOCKED"; exit 4; }
export CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
mkdir -p $OUT
/usr/bin/time -v $PY $(dirname $0)/build_cache_redirect.py $OUT $FZ "$@" 2>&1 | grep -v "^\s*\(Page\|Swaps\|File system\|Socket\|Signals\|Average\|Exit\|Voluntary\|Involuntary\|Minor\|Major\)"
