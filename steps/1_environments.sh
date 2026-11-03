#!/bin/bash
# Step 1: build the three pinned conda environments under $ENVS, then check each against its pinned package list.
#   fusion-eq-gpu               Zara's GPU environment: DIII-D networks, r1             zara/environment/fusion-eq-gpu.yml
#   fusion-equilibrium-starter  the starter kit's packages: DIII-D input caches,       zara/environment/fusion-equilibrium-starter.yml
#                               the organisers' validator, the download scripts
#   fusion-eq                   Chris's environment: his MAST models                    chris/environment/environment.yml
# Every conda package comes from conda-forge; the two variables below keep conda off Anaconda's `defaults` channel, which
# would stop at a terms-of-service prompt. An environment that already exists is only checked.
set -euo pipefail
source "$(dirname "$0")/../settings.sh"
export CONDA_CHANNELS=conda-forge CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge
build() {  # name spec pinned-pip-list
  [ -x $ENVS/$1/bin/python ] || conda env create -p $ENVS/$1 -f $REPO/$2
  if diff <($ENVS/$1/bin/python -m pip list --format=freeze | sort) <(sort $REPO/$3); then echo "$1: identical to $3"
  else echo "$1: differs from $3 (above)"; exit 1; fi
}
build fusion-eq-gpu zara/environment/fusion-eq-gpu.yml zara/environment/fusion-eq-gpu.pip-freeze.txt
build fusion-equilibrium-starter zara/environment/fusion-equilibrium-starter.yml zara/environment/fusion-equilibrium-starter.pip-freeze.txt
build fusion-eq chris/environment/environment.yml chris/environment/pip-freeze.txt
for e in fusion-eq-gpu fusion-eq; do
  $ENVS/$e/bin/python -c "import torch; print('$e: torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
done
