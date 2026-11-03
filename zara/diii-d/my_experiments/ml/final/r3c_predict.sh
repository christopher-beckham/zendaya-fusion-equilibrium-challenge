#!/bin/bash
# q95 / betaN of recipe R3c for an inputs-only DIII-D split, sequential (package version of ml/ch1q/slurm/newfold_R3c.sh):
#   bash r3c_predict.sh <split> <Group H mean map dir of the split>
# -> my_experiments/ml/ch1q/runs/nf_<split>_R3c/preds_test/<stem>.npz (q95, betaN, t). Needs the c1/nn and Group A caches of
# the split, the trained R3c networks (ch1q/runs/prod_*, ch1s/runs/prod_*_D) and the training-side caches built by
# train_ch1.sh (inp_rfull_train, hist_ch1hq_train, mapf_poof, xb_all): the networks recompute their input statistics from them.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/env.sh"
SPLIT=$1; GH=$2
NSHOT=$(ls $FEQ_ROOT/challenge_data/data/$SPLIT/*.parquet | wc -l)
done_dir() { [ -d "$1" ] && [ "$(ls "$1" | grep -c '\.npz$')" -ge "$NSHOT" ]; }
Q=$FEQ_ROOT/my_experiments/ml/ch1q; C=$FEQ_CACHE/ml/ch1q; NNC=$FEQ_CACHE/nn/$SPLIT; W=$FEQ_WORKERS
cd $Q
mkdir -p runs $C/hist_ch1h
[ -f $C/inp_rfull_$SPLIT.npz ] || $FEQ_PY export_inputs_q.py --split $SPLIT
[ -f $C/hist_$SPLIT.npz ] || $FEQ_PY hist_feats.py --split $SPLIT --workers $W
[ -f $C/hist_ch1h/$SPLIT/meta.json ] || $FEQ_PY hist_features_ch1h.py --split $SPLIT --workers $W
[ -f $C/hist_ch1hq_$SPLIT.npz ] || $FEQ_PY hist_from_ch1h.py --split $SPLIT --with-q --src $C/hist_ch1h
[ -f $C/qc_${SPLIT}_gh.npz ] || $FEQ_PY qcurve_any.py --preds $GH --cache $NNC --out $C/qc_${SPLIT}_gh.npz --workers $W
T="--test-inp $C/inp_rfull_$SPLIT.npz --test-hist $C/hist_ch1hq_$SPLIT.npz --test-qc $C/qc_${SPLIT}_gh.npz --test-ens $GH"
BASE="--targets q95,betaN,xb --pos --opt muon --muon-wd 0.1 --epochs 40"
QQ=""; SS=""; DD=""
for f in plans/prod_hm6.txt plans/prod_hm12_{1,2,3,4,5,6}.txt plans/prod_seq_{1,2,3,4,5,6}.txt; do
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$FEQ_CACHE}
    echo "== $(date -u +%T) predict $name"
    done_dir runs/nf_${SPLIT}_$name/preds_test || $FEQ_PY train_sq.py $BASE $args --load runs/$name/model.pt $T --out runs/nf_${SPLIT}_$name
    case $name in prod_seq*) SS="$SS,q:nf_${SPLIT}_$name";; *) QQ="$QQ,q:nf_${SPLIT}_$name";; esac
  done < $f
done
for f in ../ch1s/plans/prodD_{a,b,c}.txt; do
  while read -r name args; do
    [ -z "$name" ] && continue
    args=${args//@FEQ_CACHE@/$FEQ_CACHE}
    args=$(echo "$args" | sed "s#--xbmix ,[^ ]*#--xbmix ,$C/qc_${SPLIT}_gh.npz#")
    echo "== $(date -u +%T) predict $name"
    done_dir runs/nf_${SPLIT}_$name/preds_test || $FEQ_PY train_sq.py $BASE $args --load ../ch1s/runs/$name/model.pt --test-inp $C/inp_rfull_$SPLIT.npz --out runs/nf_${SPLIT}_$name
    DD="$DD,q:nf_${SPLIT}_$name"
  done < $f
done
QQ=${QQ#,}; SS=${SS#,}; DD=${DD#,}
$FEQ_PY fin_test.py --out runs/nf_${SPLIT}_R3c --test-cache $NNC --ens $GH --qc $C/qc_${SPLIT}_gh.npz --q $QQ --b $QQ --x $QQ \
    --q2 $DD --b2 $DD --w2q 0.2 --w2b 0.2 --q3 $SS --b3 $SS --w3q 0.3 --w3b 0.5 --wq 0.1,0.5,0.4 --wb 0.3,0.7 --e-bn-range 0,6
echo "== $(date -u +%T) R3c $SPLIT -> $Q/runs/nf_${SPLIT}_R3c/preds_test"
