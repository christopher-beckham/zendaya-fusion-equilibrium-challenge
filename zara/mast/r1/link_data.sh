#!/bin/bash
# usage: bash zara/mast/r1/link_data.sh <ROOT of data/fetch_official.py>
# Points r1's two views of the official data at the download of the repository's data/fetch_official.py:
#   zara/mast/r1/starter/challenge_data/data/<split>                                  (training: diii_d_train, mast_public_test)
#   $R1_ROOT/fusion-equilibrium-challenge-starter/challenge_data/data/mast_public_test (one file read by the post-processing)
# R1_ROOT must be set (zara/mast/r1/README.md). Only inputs are linked; r1 reads no MAST label.
set -euo pipefail
ROOT=$(cd "$1" && pwd); HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R1=${R1_ROOT:?set R1_ROOT, the r1 data folder}
mkdir -p $HERE/starter/challenge_data/data $R1/fusion-equilibrium-challenge-starter/challenge_data/data
for s in diii_d_train diii_d_public_test mast_public_test; do
  [ -d $ROOT/development/data/$s ] && ln -sfn $ROOT/development/data/$s $HERE/starter/challenge_data/data/$s
done
[ -d $ROOT/development/data/mast_public_test ] && \
  ln -sfn $ROOT/development/data/mast_public_test $R1/fusion-equilibrium-challenge-starter/challenge_data/data/mast_public_test
for s in diii_d_train diii_d_public_test mast_public_test; do
  [ -e $HERE/starter/challenge_data/data/$s ] && echo "$s: $(ls $HERE/starter/challenge_data/data/$s/ | grep -c '\.parquet$') shots -> $(readlink $HERE/starter/challenge_data/data/$s)" || echo "$s: not downloaded"
done
