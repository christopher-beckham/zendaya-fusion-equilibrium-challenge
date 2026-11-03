#!/bin/bash
# usage: bash link_data.sh <ROOT of data/fetch_official.py> [<Final dataset folder>]
# Points challenge_data/data/<split>/ of this package at the official data downloaded by the repository's data/fetch_official.py
# (ROOT/development/data/<split> for diii_d_train and diii_d_public_test, ROOT/final/data/<split> for diii_d_private_test).
# The optional second argument is a download of the Final dataset (ganatma/fusion-equilibrium-challenge-final, the folder that
# holds data/diii_d_private_test/); it replaces ROOT/final for diii_d_private_test.
set -euo pipefail
ROOT=$(cd "$1" && pwd); HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FINAL=$ROOT/final; [ -n "${2:-}" ] && FINAL=$(cd "$2" && pwd)
mkdir -p $HERE/challenge_data/data
for s in diii_d_train diii_d_public_test; do [ -d $ROOT/development/data/$s ] && ln -sfn $ROOT/development/data/$s $HERE/challenge_data/data/$s; done
[ -d $FINAL/data/diii_d_private_test ] && ln -sfn $FINAL/data/diii_d_private_test $HERE/challenge_data/data/diii_d_private_test
for s in diii_d_train diii_d_public_test diii_d_private_test; do
  [ -e $HERE/challenge_data/data/$s ] && echo "$s: $(ls $HERE/challenge_data/data/$s/ | grep -c '\.parquet$') shots -> $(readlink $HERE/challenge_data/data/$s)" || echo "$s: not downloaded"
done
