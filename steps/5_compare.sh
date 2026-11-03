#!/bin/bash
# Step 5: compare regenerated files with the submitted ones.
#   bash steps/5_compare.sh A|B OUT_DIR
# Downloads the submitted files from the predictions repository at the commit the organisers recorded
# (A: zenithon-zendaya/fec-final-sub1 @ 807b5104, B: zenithon-zendaya/fec-final-sub5 @ 3a2a2a26) into $WORK/submitted/<A|B>,
# then, for each file in OUT_DIR: SHA256 of both (equal when OUT_DIR came from the shipped weights), and per quantity the
# number of byte-identical arrays, the largest difference and the pooled ψ R² of OUT_DIR's file against the submitted one
# (for retrained models). No label is used: the organisers score with the Final labels.
set -euo pipefail
source "$(dirname "$0")/../settings.sh"
case "${1:-}" in
  A) REPO_ID=zenithon-zendaya/fec-final-sub1; REV=807b5104fa53619102cb728c980c5f481db9b499 ;;
  B) REPO_ID=zenithon-zendaya/fec-final-sub5; REV=3a2a2a269053df11ac3c22ab5fdb184c5dfda6b7 ;;
  *) echo "usage: $0 A|B OUT_DIR"; exit 2 ;;
esac
OUT=$(cd "${2:?OUT_DIR}" && pwd); SUB=$WORK/submitted/$1
$FEQ_PY_STARTER - $REPO_ID $REV $SUB <<'PY'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["*.npz"],
                  token=os.environ.get("FEC_HF_TOKEN") or None)
PY
for f in $OUT/*_private_test.npz; do
  n=$(basename $f)
  echo "== $n"
  sha256sum $SUB/$n $f | sed "s#$WORK/##"
  $ENVS/fusion-eq-gpu/bin/python $REPO/zara/diii-d/checks/compare_sub.py $SUB/$n $f
done
