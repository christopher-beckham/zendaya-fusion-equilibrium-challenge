#!/bin/bash
#SBATCH --job-name=m2-gen
#SBATCH --partition=gpu4,gpu2
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/logs/%x-%j.out
# usage: sbatch --cpus-per-task=N --mem=M gen_workers.sbatch <pool> <n_workers> <n_points> [stall_s]
#   n_workers single-threaded gen_m2.py workers on the already-running node, with a watchdog: a worker whose output
#   directory has not changed for STALL seconds is killed and relaunched (it resumes; the point it hung on is marked "hang").
set -uo pipefail
source /fsx/zara.ercan/env.sh
conda activate /fsx/zara.ercan/conda/envs/freegsnke
cd /fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/gen
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
POOL=$1; NW=$2; NP=$3; STALL=${4:-100}; ISTART=${5:-0}
LOGS=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/m2/logs
RAW=/fsx/zara.ercan/more_data/m2/mast_${POOL}/raw
declare -a PID DONE
launch() { python gen_m2.py --pool $POOL --worker $1 --n-workers $NW --n-points $NP --i-start $ISTART >> $LOGS/gen_${POOL}_x${ISTART}_w$1.log 2>&1 & PID[$1]=$!; }
for k in $(seq 0 $((NW-1))); do mkdir -p $RAW/x${ISTART}_w$(printf %02d $k); launch $k; DONE[$k]=0; done
sleep 60
restarts=0
while true; do
  alive=0; now=$(date +%s)
  for k in $(seq 0 $((NW-1))); do
    [ "${DONE[$k]}" = 1 ] && continue
    if ! kill -0 ${PID[$k]} 2>/dev/null; then
      if grep -q "finished" $LOGS/gen_${POOL}_x${ISTART}_w$k.log; then DONE[$k]=1; continue; fi
      echo "$(date -u +%T) worker $k died, relaunching"; launch $k; alive=$((alive+1)); continue
    fi
    alive=$((alive+1))
    d=$RAW/x${ISTART}_w$(printf %02d $k); newest=$(ls -t $d 2>/dev/null | head -1)
    if [ -n "$newest" ]; then m=$(stat -c %Y "$d/$newest" 2>/dev/null || echo $now); else m=$now; fi
    if [ $((now - m)) -gt $STALL ]; then
      echo "$(date -u +%T) worker $k stalled $((now - m)) s, killing pid ${PID[$k]}"; kill -9 ${PID[$k]} 2>/dev/null; wait ${PID[$k]} 2>/dev/null
      restarts=$((restarts+1)); launch $k
    fi
  done
  [ $alive -eq 0 ] && break
  sleep 20
done
echo "all workers finished; restarts $restarts"
python gen_m2.py --summary --pool $POOL
