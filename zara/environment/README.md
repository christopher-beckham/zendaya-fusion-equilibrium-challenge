# Zara's environments

Three conda environments, exported on 22 October 2026 from the environments that trained and predicted every model in
`zara/` (`/fsx/zara.ercan/conda/envs/<name>` on the AWS cluster). Every conda package comes from conda-forge; everything else
is installed by pip from PyPI (and the PyTorch index for the CUDA 12.8 build of torch).

| Environment | Python | Used for |
|---|---|---|
| `fusion-eq-gpu` | 3.11.16, torch 2.11.0+cu128, numpy 2.4.6 | every GPU step: DIII-D networks (training and inference), q95 / βN networks, calibration, submission files |
| `fusion-equilibrium-starter` | 3.11.16, numpy 2.4.6 (the starter kit's packages, no torch) | per-shot input caches of the raw parquet files, C1 physics + Ridge base model, the organisers' validator |
| `freegsnke` | 3.10.21, FreeGSNKE 3.1.1, numpy 2.2.6 | generating synthetic equilibria (CPU only); not needed to train or predict the DIII-D models |

Files per environment:

| File | What it is |
|---|---|
| `<name>.yml` | `conda env export` (conda and pip packages, exact versions and builds). Two changes by hand: `nodefaults` next to `conda-forge` (no package needs Anaconda's `defaults` channel, which asks for its terms of service), and in `fusion-eq-gpu.yml` the PyTorch index URL, which pip needs to find `torch==2.11.0+cu128` |
| `<name>.conda-explicit-linux-64.txt` | `conda list --explicit --md5`: exact package URLs and checksums for the conda part |
| `<name>.pip-freeze.txt` | `pip list --format=freeze`: every package visible to pip, including those installed by conda |
| `<name>.versions.txt` | Python and key package versions as the interpreter reports them (torch with its CUDA version) |

Rebuild:

```bash
CONDA_CHANNELS=conda-forge CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge \
  conda env create -p ./envs/fusion-eq-gpu -f zara/environment/fusion-eq-gpu.yml
CONDA_CHANNELS=conda-forge CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge \
  conda env create -p ./envs/fusion-equilibrium-starter -f zara/environment/fusion-equilibrium-starter.yml
```

The two variables keep conda off Anaconda's `defaults` channel, which a stock Miniconda configures and which stops
`conda env create` at a terms-of-service prompt. The DIII-D package takes the interpreters as `FEQ_PY`
(`fusion-eq-gpu`) and `FEQ_PY_STARTER` (`fusion-equilibrium-starter`); see `zara/diii-d/README.md`.

## Checks (22 October 2026, AWS cluster)

All three environments were built from these files into an empty folder, with an empty conda package cache and no pip cache
(`final/slurm/env_buildtest.sbatch`):

| Environment | Build time | Size | `pip list` against `<name>.pip-freeze.txt` |
|---|---|---|---|
| `fusion-equilibrium-starter` | 94 s | 307 MB | identical |
| `fusion-eq-gpu` | 201 s | 4.5 GB | identical |
| `freegsnke` | 101 s | 350 MB | identical |

- `fusion-eq-gpu`: torch 2.11.0+cu128 sees the GPU (NVIDIA RTX PRO 6000 Blackwell) and a 4096 × 4096 matmul runs.
- `freegsnke`: FreeGSNKE 3.1.1 imports.
- Smoke test, `fusion-equilibrium-starter`: the DIII-D input caches of the public test (874 shots, `c1/nn/preprocess.py`)
  built in the new environment are byte-identical to the caches the models were run on (875 of 875 files).
- Smoke test, `fusion-eq-gpu`: three v13 networks of different kinds (ψ network `prod_auxghc_s1`, context head
  `prod_scog_ctx_s1`, sequence head `prod_seqh_s1`) predicted the public test from the shipped weights in the new environment
  (`infer_c1h.py`): byte-identical to the predictions made in the original environment (874 of 874 shots each).

Not yet built on a machine outside this cluster.
