#!/bin/bash
# Queue runner (copy of ../tune/struct_runner2.sh): runner.sh QUEUE_FILE [THREADS]
# lines: TAG SEED KNOB-OVERRIDES-JSON (added to knobs_x8.json unless "_base": "none"). Claims in claims/.
cd /fsx/chris.beckham/gsolve_data/tune2
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/fsx/chris.beckham/gsolve_copy
Q=$1; TH=${2:-5}
export OMP_NUM_THREADS=$((4*TH))
while true; do
  line=""
  while read -r tag seed js; do
    [ -z "$tag" ] && continue; [[ "$tag" == \#* ]] && continue
    [ -f runs/$tag/result.json ] && continue
    mkdir -p claims; if mkdir claims/$tag 2>/dev/null; then line="$tag $seed $js"; break; fi
  done < $Q
  [ -z "$line" ] && { [ -f ${Q}.stop ] && exit 0; sleep 60; continue; }
  read -r tag seed js <<< "$line"
  ./venv/bin/python -c "import json,sys; o=json.loads(sys.argv[1]); b={} if o.pop('_base',None)=='none' else json.load(open('knobs_x8.json')); print(json.dumps(b|o))" "$js" > claims/$tag/knobs.json
  echo "[$(date -u +%H:%M)] $tag seed $seed $js"
  ./venv/bin/python tune.py eval --tag $tag --seed $seed --stage cond --perslice 1 --shards 4 --threads $TH --knobs claims/$tag/knobs.json 2>&1 | grep -v Warning | tail -2
  [ -f runs/$tag/result.json ] && OMP_NUM_THREADS=8 ./venv/bin/python fixed_frames.py $tag 2>&1 | grep "^{" | sed "s/^/FIXED /"
done
