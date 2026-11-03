#!/bin/bash
#SBATCH -J d3d-labnoise
#SBATCH -p gpu2,g7e8-spot,gpu4
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 5:00:00
#SBATCH -o /fsx/zara.ercan/tmp/claude-2009/-fsx-zara-ercan/f5b9ea66-d9e3-4a9d-b150-64acdaf7e4ba/scratchpad/repro/%x-%j.out
# How much GPU rounding noise moves S, measured on real labels. The package copy with the shipped weights (the 22 Oct test
# copy, fresh environments) predicts ~100 labelled training shots twice: alone (split diii_d_lab1) and mixed into ~200 shots
# (diii_d_lab2, which changes every batch the lab1 shots sit in). Both runs are then scored on the lab1 shots against their EFIT
# labels with the official scorer code (score_lab.py); S(lab1 run) - S(lab2 run) is the noise effect. All three recipes.
# The shots are training shots (in-sample for the shipped weights): S is higher than on test shots, the difference is what counts.
set -euo pipefail
R=/fsx/zara.ercan/tmp/claude-2009/-fsx-zara-ercan/f5b9ea66-d9e3-4a9d-b150-64acdaf7e4ba/scratchpad/repro
T=/fsx/zara.ercan/tmp/pkgtest; PK=$T/diii-d; CA=$T/cache_lab; P=/fsx/zara.ercan/cache; E=/fsx/zara.ercan/tmp/envtest
TR=/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/diii_d_train
export FEQ_ROOT=$PK FEQ_CACHE=$CA FEQ_PY=$E/fusion-eq-gpu/bin/python FEQ_PY_STARTER=$E/fusion-equilibrium-starter/bin/python
export FEQ_WORKERS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 FEQ_CLEAN=1
# splits: lab1 = every 70th training shot from the 4th; lab2 = lab1 + every 70th from the 39th
for s in diii_d_lab1 diii_d_lab2; do mkdir -p $PK/challenge_data/data/$s; done
ls $TR/*.parquet | sort | awk 'NR % 70 == 4' | while read f; do ln -sfn $f $PK/challenge_data/data/diii_d_lab1/; ln -sfn $f $PK/challenge_data/data/diii_d_lab2/; done
ls $TR/*.parquet | sort | awk 'NR % 70 == 39' | while read f; do ln -sfn $f $PK/challenge_data/data/diii_d_lab2/; done
echo "lab1 $(ls $PK/challenge_data/data/diii_d_lab1 | wc -l) shots, lab2 $(ls $PK/challenge_data/data/diii_d_lab2 | wc -l) shots"
# cache: only the shipped inference artifacts (the production copies the weights were trained against) + the CPU training cache
mkdir -p $CA/nn; [ -L $CA/nn/diii_d_train ] || ln -s $P/nn/diii_d_train $CA/nn/diii_d_train
for f in nn/ridge_full.npz nn/bt_sign_diii_d_train.json nn/px/diii_d_train/meta.json \
         ml/ch1h/hist/diii_d_train/meta.json ml/ch1h/hist/diii_d_train/check.json ml/ch1s/inp_rfull_train.npz ml/ch1s/xb_all.npz \
         ml/ch1q/hist_ch1hq_train.npz ml/ch1q/hist_train.npz ml/ch1q/mapf_poof.npz; do
  mkdir -p $CA/$(dirname $f); ln -sfn $P/$f $CA/$f
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; df -h /fsx | tail -1
for rec in ch1min ch1 ch1safe; do
  for s in diii_d_lab1 diii_d_lab2; do
    t0=$(date +%s); rc=0; bash $PK/predict_$rec.sh $s > $R/lab_${rec}_$s.log 2>&1 || rc=$?
    echo "== predict_$rec.sh $s: exit $rc $(( $(date +%s) - t0 )) s"; [ $rc = 0 ] || tail -5 $R/lab_${rec}_$s.log
    df -h /fsx | tail -1
  done
done
L=$PK/challenge_data/data/diii_d_lab1; F=$CA/pred
$FEQ_PY_STARTER $R/score_lab.py $R/lab_noise.json $L \
  ch1min_a=$F/ch1min/diii_d_lab1/final ch1min_b=$F/ch1min/diii_d_lab2/final \
  v13_a=$F/diii_d_lab1/final v13_b=$F/diii_d_lab2/final \
  ch1safe_a=$F/ch1safe/diii_d_lab1/final ch1safe_b=$F/ch1safe/diii_d_lab2/final \
  --pair ch1min_a,ch1min_b --pair v13_a,v13_b --pair ch1safe_a,ch1safe_b
echo "== done"
