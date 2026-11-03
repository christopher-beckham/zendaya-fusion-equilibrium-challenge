#!/bin/bash
# r1's training caches, rebuilt on the CPU. r1 (Zara Ercan's MAST model) trains on first-pass caches: for every synthetic MAST
# and DIII-D training frame, the plasma boundary found by the model's own equilibrium solve, plus auxiliary DIII-D targets. The
# shipped r1 networks were trained on caches built this way, with no GPU visible, so they count 0 GPU-hours (rule 6 asks for
# this work to be declared with its hardware and wall-clock: README.md in this folder). The same caches can be downloaded
# instead; this script is how to make them yourself.
#
#   R1_ROOT=<Zara's data folder> R1_PY=<python> R1_PY_STARTER=<python> bash zara/mast/r1/cpu_caches/build_r1_caches.sh
#
# R1_ROOT holds Zara's data bundle (cache/...) WITHOUT the caches this script builds: delete any downloaded copy first (the
# script stops if one is there). Official data must be linked: zara/mast/r1/starter/challenge_data/data/{diii_d_train,
# mast_public_test} and zara/diii-d/challenge_data/data/diii_d_train (zara/diii-d/link_data.sh). R1_PY: the environment the
# caches were built with (chris/environment, fusion-eq); R1_PY_STARTER: zara/environment fusion-equilibrium-starter.
# Steps, in order (each runs with CUDA_VISIBLE_DEVICES empty):
#   0. inputs derived from the official data that the bundle lacks (DIII-D input, Thomson, reference and label caches; the
#      MAST public-test input cache); files already present are kept
#   1. base model's cache       cache/ch2/syn/mastsyn_synth_p7            m2/upegs_fz/build_cache.py   6 shards x 8 threads
#   2. fine-tune caches         cache/ch2/g4/caches/mastg4*_synth_wd      ch2/g4/fz/build_cache.py     6 x 8, per machine
#   3. real-MAST test cache     cache/ch2/g4/caches/mast_test_iv_full_wdiv ch2/g4/fz/build_cache.py    6 x 8
#   4. time features of the fine-tune caches from the real shots (ch2/g4/fix_time.py)
#   5. DIII-D first passes      cache/pegs/diiid_train_s2p7, _p7          pegs/build_cache.py          10 x 4, 4 x 12
#   6. DIII-D auxiliary targets cache/upegs/aux_d3d_s2p7, _p7             ch2/g4/fz/build_aux.py       6 x 8, 1 x 16
# The last bits of each solve depend on the CPU type, the threads per process and the grouping of frames into processes: the
# shard counts are fixed, and the default threads (R1_THREADS overrides them) are those that built the shipped caches. Sized for a 48-core machine: about 7 h there. Logs and step times (times.tsv): $R1_ROOT/cache_build_logs/. If the run
# is interrupted, run it again: folders this script created are continued (finished shots are kept), others stop it.
# R1_SMOKE_SHOTS=N: only the first N shots of each cache (a quick test of the whole chain; not a usable cache).
set -euo pipefail
: "${R1_ROOT:?set R1_ROOT, the folder of the r1 data bundle}" "${R1_PY:?set R1_PY}" "${R1_PY_STARTER:?set R1_PY_STARTER}"
export R1_ROOT
R1=$(cd "$(dirname "$0")/.." && pwd); ME=$R1/starter/my_experiments; DIIID=$(cd $R1/../../diii-d && pwd)
C=$R1_ROOT/cache; L=$R1_ROOT/cache_build_logs; MARK=.r1_cpu_build
export CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $L; [ -f $L/times.tsv ] || printf "step\tseconds\tfinished_utc\tkind\n" > $L/times.tsv

# ---------------------------------------------------------------- checks before anything runs
OUTS="$C/m2/mastsyn_synth_p7 $C/ch2/syn/mastsyn_synth_p7 $C/ch2/g4/caches/mast_test_iv_full_wdiv $C/pegs/diiid_train_s2p7
      $C/pegs/diiid_train_p7 $C/upegs/aux_d3d_s2p7 $C/upegs/aux_d3d_p7"
for m in mastg4 mastg4rt mastg4v3 mastg4v3rt; do OUTS="$OUTS $C/ch2/g4/caches/${m}_synth_wd"; done
for d in $OUTS; do
  if [ -n "$(ls -A $d 2>/dev/null)" ] && [ ! -e $d/$MARK ]; then
    echo "$d already holds a cache this script did not build (a downloaded copy, or Zara's GPU-built one)."
    echo "Delete it first (rm -rf $d), then run this script again."; exit 3
  fi
done
for d in $R1/starter/challenge_data/data/diii_d_train $R1/starter/challenge_data/data/mast_public_test \
         $DIIID/challenge_data/data/diii_d_train $C/ch2/g4/multi/MASTG4/src $C/m2/multi/MASTSYN/src; do
  [ -d $d ] || { echo "missing $d (official data links or Zara's data bundle: zara/mast/r1/README.md)"; exit 3; }
done
NTR=$(ls $R1/starter/challenge_data/data/diii_d_train | grep -c '\.parquet$')
NMT=$(ls $R1/starter/challenge_data/data/mast_public_test | grep -c '\.parquet$')
echo "R1_ROOT $R1_ROOT; $NTR DIII-D training shots, $NMT MAST public-test shots; $(nproc) CPU cores"

# ---------------------------------------------------------------- helpers
step() {   # step <name> <command ...>: run once, time it, record it in times.tsv
  local name=$1; shift
  echo "== $(date -u +%FT%TZ) $name"; local t0=$(date +%s)
  "$@"
  local dt=$(( $(date +%s) - t0 ))
  printf "%s\t%s\t%s\tCPU\n" "$name" "$dt" "$(date -u +%FT%TZ)" >> $L/times.tsv
  echo "   $name: $(( dt / 3600 )) h $(( dt % 3600 / 60 )) min"
}
shards() {   # shards <log name> <n shards> <threads> <dir> <script and args ...>: one process per shard, all at once
  local name=$1 n=$2 th=${R1_THREADS:-$3} dir=$4; shift 4
  local pids="" k
  for k in $(seq 0 $(( n - 1 ))); do
    (cd $dir && OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th nice $R1_PY "$@" --shard $k --n-shards $n ${R1_SMOKE_SHOTS:+--max-shots $R1_SMOKE_SHOTS}) \
      > $L/${name}_$k.log 2>&1 &
    pids="$pids $!"
  done
  local p rc=0; for p in $pids; do wait $p || rc=1; done
  [ $rc = 0 ] || { echo "a shard of $name failed: $L/${name}_*.log"; exit 1; }
}
claim() { mkdir -p $1 && touch $1/$MARK; }   # mark an output folder as built here
count() { ls $1 2>/dev/null | grep -c '\.npz$' || true; }

# ---------------------------------------------------------------- 0. inputs derived from the official data
# DIII-D per-shot input and Thomson caches (Zara's DIII-D package code), reference LCFS and EFIT profile labels of every training
# shot; the bundle holds only the shots r1's training opens. Each builder skips the shots already present.
step inputs_nn_diii_d_train bash -c "cd $DIIID/my_experiments/c1/nn && $R1_PY_STARTER preprocess.py \
    --data-dir $R1/starter/challenge_data/data/diii_d_train --out $C/nn/diii_d_train --workers ${R1_WORKERS:-16} > $L/inputs_nn.log 2>&1"
step inputs_px_diii_d_train bash -c "cd $DIIID/my_experiments/c1/nn && FEQ_CACHE=$C $R1_PY_STARTER preprocess_px.py \
    --split diii_d_train --workers ${R1_WORKERS:-16} > $L/inputs_px.log 2>&1"
step inputs_refs_d3d bash -c "cd $ME/ch2/g4/fz && $R1_PY build_refs.py --workers ${R1_WORKERS:-16} > $L/inputs_refs.log 2>&1"
step inputs_labels_d3d bash -c "cd $ME/xg && $R1_PY labels.py --workers ${R1_WORKERS:-16} > $L/inputs_labels.log 2>&1"
# MAST public-test input cache (inputs only), read by step 3 and by fix_time.py
if [ "$(ls $C/mast/mast_public_test 2>/dev/null | grep -c '^mast_shot.*\.npz$')" -lt $NMT ]; then
  step inputs_mast_public_test bash -c "cd $ME/c2 && $R1_PY_STARTER preprocess_mast.py --data-dir $R1/starter/challenge_data/data/mast_public_test \
      --out $C/mast/mast_public_test --workers ${R1_WORKERS:-16} > $L/inputs_mast.log 2>&1"
fi

# ---------------------------------------------------------------- 1. base model's cache (MASTSYN, 93,148 synthetic records)
# m2/upegs_fz writes it to cache/m2/mastsyn_synth_p7; the base model reads it as cache/ch2/syn/mastsyn_synth_p7
if [ ! -e $C/ch2/syn/mastsyn_synth_p7/$MARK ]; then
  claim $C/m2/mastsyn_synth_p7
  step base shards base 6 8 $ME/m2/upegs_fz build_cache.py --machine MASTSYN --split synth --p-share 0.7 --tag _p7
  mkdir -p $C/ch2/syn && mv $C/m2/mastsyn_synth_p7 $C/ch2/syn/mastsyn_synth_p7
fi
echo "   base cache: $(count $C/ch2/syn/mastsyn_synth_p7) files"

# ---------------------------------------------------------------- 2-4. fine-tune caches, real-MAST test cache, time features
export INTR_WALL=0.20 INTR_FP=divert
for M in MASTG4 MASTG4RT MASTG4V3 MASTG4V3RT; do
  claim $C/ch2/g4/caches/$(echo $M | tr A-Z a-z)_synth_wd
  step ft_$M shards ft_$M 6 8 $ME/ch2/g4/fz build_cache.py --machine $M --split synth --sol-factor 0.5 --p-share 0.7 --tag _wd
done
claim $C/ch2/g4/caches/mast_test_iv_full_wdiv
step mast_test shards mast_test 6 8 $ME/ch2/g4/fz build_cache.py --machine MAST --split test --sol-factor 0.5 --p-share 0.7 \
    --tag _iv_full_wdiv
unset INTR_WALL INTR_FP
for M in MASTG4 MASTG4RT MASTG4V3 MASTG4V3RT; do
  D=$C/ch2/g4/caches/$(echo $M | tr A-Z a-z)_synth_wd
  [ -e $D/.fix_time_done ] && continue
  [ -n "${R1_SMOKE_SHOTS:-}" ] && { echo "   fix_time_$M: skipped (R1_SMOKE_SHOTS: it needs the whole real-MAST test cache)"; continue; }
  step fix_time_$M bash -c "cd $ME/ch2/g4 && $R1_PY fix_time.py train $M > $L/fix_time_$M.log 2>&1"
  touch $D/.fix_time_done
done

# ---------------------------------------------------------------- 5-6. DIII-D first passes and auxiliary targets
claim $C/pegs/diiid_train_s2p7; claim $C/pegs/diiid_train_p7
step diiid_s2p7 shards diiid_s2p7 10 4 $ME/pegs build_cache.py --machine DIII-D --split train --subset train --stride 2 \
    --p-share 0.7 --tag _s2p7
step diiid_p7 shards diiid_p7 4 12 $ME/pegs build_cache.py --machine DIII-D --split train --subset holdout --p-share 0.7 --tag _p7
claim $C/upegs/aux_d3d_s2p7; claim $C/upegs/aux_d3d_p7
step aux_s2p7 shards aux_s2p7 6 8 $ME/ch2/g4/fz build_aux.py --tag _s2p7
step aux_p7 shards aux_p7 1 16 $ME/ch2/g4/fz build_aux.py --tag _p7 --subset holdout

echo "== $(date -u +%FT%TZ) done. Files per cache:"
for d in ch2/syn/mastsyn_synth_p7 ch2/g4/caches/mastg4_synth_wd ch2/g4/caches/mastg4rt_synth_wd ch2/g4/caches/mastg4v3_synth_wd \
         ch2/g4/caches/mastg4v3rt_synth_wd ch2/g4/caches/mast_test_iv_full_wdiv pegs/diiid_train_s2p7 pegs/diiid_train_p7 \
         upegs/aux_d3d_s2p7 upegs/aux_d3d_p7; do
  printf "   %-42s %6s\n" $d "$(count $C/$d)"
done
echo "   expected: 93148, 24874, 24874, 5000, 5000, 1206, 6337, 704, 6337, 704; step times in $L/times.tsv"
