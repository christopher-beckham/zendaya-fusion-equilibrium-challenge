#!/bin/bash
#SBATCH -p g7e8-spot
#SBATCH -c 96
#SBATCH --mem=600G
#SBATCH -J synval_chris_inputs
#SBATCH -o /fsx/chris.beckham/synval/prep_chris.log
set -euo pipefail
S=/fsx/chris.beckham/synval; cd $S/repo-shipped
export WORK=$S/work-shipped ENVS=/fsx/chris.beckham/fec_pkgtest/work/envs DATA=$S/DATA; source settings.sh
FEC_DATA_ROOT=$DATA/final FEC_CACHE=$MAST_FINAL_CACHE bash chris/mast/build_mast_test_inputs.sh mast_private_test 48
