#!/bin/bash
# Step 2: download the official challenge data into $DATA and point the packages at it.
#   bash steps/2_official_data.sh A    DIII-D splits only (about 93 GB)
#   bash steps/2_official_data.sh B    DIII-D and MAST splits (about 100 GB)
# Development dataset Sophelio/fusion-equilibrium-challenge (diii_d_train, diii_d_public_test, mast_public_test) and Final
# dataset ganatma/fusion-equilibrium-challenge-final (diii_d_private_test, mast_private_test), at the revisions pinned in
# data/fetch_official.py. Every file is checked against the SHA256 Hugging Face publishes. Both datasets are public.
# Test splits hold inputs only. A finished download is checked again and not downloaded twice.
set -euo pipefail
source "$(dirname "$0")/../settings.sh"
SPLITS="diii_d_train diii_d_public_test diii_d_private_test"
case "${1:-}" in A) ;; B) SPLITS="$SPLITS mast_public_test mast_private_test" ;; *) echo "usage: $0 A|B"; exit 2 ;; esac
$FEQ_PY_STARTER $REPO/data/fetch_official.py --root $DATA --splits $SPLITS
bash $REPO/zara/diii-d/link_data.sh $DATA                       # zara/diii-d/challenge_data/data/<split> -> $DATA
if [ "$1" = B ]; then
  mkdir -p $R1_ROOT && bash $REPO/zara/mast/r1/link_data.sh $DATA   # zara/mast/r1/starter/challenge_data/data/<split> -> $DATA
fi
