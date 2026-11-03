#!/bin/bash
# r1_cpu_chain.sh GPU_SEQ (0 or 1): retrain r1 with the graphs code, reading only our CPU-built caches (mounted over Zara's
# cache paths). Waits for finish_caches.sh. Seq 0: h_muon_mo -> w3_ft -> rt_ftw3 -> rt_ft; seq 1: rt_mix, rt_nonoise.
set -uo pipefail
W=/fsx/chris.beckham/zara_speedup; O=$W/out/cpucache/full; Z=/fsx/zara.ercan/cache; R=$W/out/r1cpu; mkdir -p $R
G4=$W/graphs_wt/starter/my_experiments/ch2/g4/fz; SYN=$W/graphs_wt/starter/my_experiments/ch2/syn/fz
ZR=$Z/ch2/g4/runs; HMA=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/syn/runs/h_muon_mo/args.json
MAP="$O/base/mastsyn_synth_p7=$Z/ch2/syn/mastsyn_synth_p7"
for s in mastg4 mastg4rt mastg4v3 mastg4v3rt; do MAP="$MAP;$O/ft/${s}_synth_wd=$Z/ch2/g4/caches/${s}_synth_wd"; done
MAP="$MAP;$O/pegs/diiid_train_s2p7=$Z/pegs/diiid_train_s2p7;$O/pegs/diiid_train_p7=$Z/pegs/diiid_train_p7"
MAP="$MAP;$O/upegs/aux_d3d_s2p7=$Z/upegs/aux_d3d_s2p7;$O/upegs/aux_d3d_p7=$Z/upegs/aux_d3d_p7"
run() {  # name code args_json intr_wall [overrides]
  local n=$1 fz=$2 aj=$3 iw=$4; shift 4
  grep -q "^exit 0" $R/$n.log 2>/dev/null && { echo "skip $n"; return 0; }
  echo "$(date -u +%T) start $n"
  INTR_WALL=$iw unshare -rm bash $W/scripts/train_overlay.sh "$MAP" $fz $R/$n $aj "$@" > $R/$n.log 2>&1
  echo "$(date -u +%T) end $n: $(grep '^exit' $R/$n.log)"; grep -q "^exit 0" $R/$n.log
}
until grep -q "all done" $O/finish.log 2>/dev/null; do sleep 30; done
if [ "$1" = 0 ]; then
  export CUDA_VISIBLE_DEVICES=4
  run h_muon_mo $SYN $HMA "" || exit 1
  run w3_ft $G4 $ZR/w3_ft/args.json 0.20 init=$R/h_muon_mo/model.pt || exit 1
  run rt_ftw3 $G4 $ZR/rt_ftw3/args.json 0.20 init=$R/w3_ft/model.pt || exit 1
  run rt_ft $G4 $ZR/rt_ft/args.json 0.20 init=$R/h_muon_mo/model.pt || exit 1
else
  export CUDA_VISIBLE_DEVICES=5
  until grep -q "^exit 0" $R/h_muon_mo.log 2>/dev/null; do sleep 30; grep -q "^exit [1-9]" $R/h_muon_mo.log 2>/dev/null && exit 1; done
  run rt_mix $G4 $ZR/rt_mix/args.json 0.20 init=$R/h_muon_mo/model.pt || exit 1
  run rt_nonoise $G4 $ZR/rt_nonoise/args.json 0.20 init=$R/h_muon_mo/model.pt || exit 1
fi
echo "$(date -u +%T) sequence $1 done"
