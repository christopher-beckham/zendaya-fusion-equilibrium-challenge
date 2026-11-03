# settings.sh: every folder and interpreter the package uses. Edit WORK (or export it before), then `source settings.sh`
# in each new shell. Every script under steps/ sources this file itself.
#
# Use one copy of this repository and one WORK folder for each check of REPRODUCE.md (shipped weights; retraining): the shipped
# DIII-D networks are downloaded to the same paths that training writes. ENVS and DATA can be set before sourcing this file to
# share the environments and the official data between the two copies.
#
# WORK is a folder on a disk with about 700 GB free. Everything downloaded or built goes under it. The only other writes are
# links inside this repository (made by the link_data.sh scripts) and the retrained MAST checkpoints, which
# chris/mast/train.sh writes under chris/mast/multimachine/my_experiments/nn/runs/.

export WORK=${WORK:-/path/to/big/disk}
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ENVS=${ENVS:-$WORK/envs}           # the three conda environments (step 1); can be shared between copies
export DATA=${DATA:-$WORK/data}           # official data: $DATA/development/data/<split>, $DATA/final/data/<split>; can be shared
export NCPU=${NCPU:-$(nproc --all)}       # worker processes of the CPU steps; lower it on a machine with little memory per core

# DIII-D, Zara Ercan's package (zara/diii-d/README.md)
export FEQ_CACHE=$WORK/diii-d_cache
export FEQ_PY_STARTER=$ENVS/fusion-equilibrium-starter/bin/python
export FEQ_PY="$ENVS/fusion-eq-gpu/bin/python $REPO/zara/diii-d/my_experiments/ml/final/capmem.py"   # caps CUDA memory at 80 GB
export FEQ_WORKERS=$NCPU OMP_NUM_THREADS=1 CUDA_MEM_CAP_GB=80

# MAST, Chris Beckham's 8 models (chris/mast/README.md)
export PY=$ENVS/fusion-eq/bin/python
export FEC_DATA_ROOT=$DATA/development    # training; prediction switches to $DATA/final (predict.sh B does it)
export FEC_CACHE=$WORK/mast_cache         # derived caches and the synthetic training data
export MAST_FINAL_CACHE=$WORK/mast_final_cache   # inputs of the Final MAST shots
export FEC_WEIGHTS=$WORK/mast_weights     # shipped checkpoints

# MAST, Zara Ercan's r1 (zara/mast/r1/README.md)
export R1_ROOT=$WORK/r1_data              # Zara's data and the CPU-built training caches
export R1_WEIGHTS=$WORK/r1_weights        # shipped networks: $R1_WEIGHTS/runs/<member>/model.pt
export R1_OUT=$WORK/r1_runs               # retrained networks (train.sh B)
export R1_PY=$ENVS/fusion-eq-gpu/bin/python R1_PY_STARTER=$ENVS/fusion-equilibrium-starter/bin/python

# Hugging Face: no token is needed; every repository the package downloads from is public.
# Downloads use Hugging Face's plain HTTP transfer: its newer Xet transfer failed repeatedly on one 8 GB file of the r1
# bundle in our test. Every file is still checked by SHA256.
export HF_HUB_DISABLE_XET=1
