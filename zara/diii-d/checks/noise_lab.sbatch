#!/bin/bash
#SBATCH -J d3d-noise-lab
#SBATCH -p gpu1,gpu2,g6e1,gpu4
#SBATCH -c 4
#SBATCH --mem=56G
#SBATCH -t 3:00:00
#SBATCH -o /fsx/zara.ercan/tmp/claude-2009/-fsx-zara-ercan/f5b9ea66-d9e3-4a9d-b150-64acdaf7e4ba/scratchpad/repro/%x-%j.out
# CPU only: score the package's shipped-weights outputs (98-shot subset, 22 Oct test) against the per-shot predictions the
# submitted files were built from, with the official scorer code; bound the S change (noise_bound.py docstring).
export OMP_NUM_THREADS=1
/fsx/zara.ercan/conda/envs/fusion-equilibrium-starter/bin/python /fsx/zara.ercan/tmp/claude-2009/-fsx-zara-ercan/f5b9ea66-d9e3-4a9d-b150-64acdaf7e4ba/scratchpad/repro/noise_bound.py /fsx/zara.ercan/tmp/claude-2009/-fsx-zara-ercan/f5b9ea66-d9e3-4a9d-b150-64acdaf7e4ba/scratchpad/repro/noise_lab.json \
  v13=/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/diii_d_lab1/final:/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/diii_d_lab2/final \
  ch1min=/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/ch1min/diii_d_lab1/final:/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/ch1min/diii_d_lab2/final \
  ch1safe=/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/ch1safe/diii_d_lab1/final:/fsx/zara.ercan/tmp/pkgtest/cache_lab/pred/ch1safe/diii_d_lab2/final
