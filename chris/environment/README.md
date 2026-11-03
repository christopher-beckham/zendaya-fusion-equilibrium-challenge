# Chris's environment

Exported from the conda environment `fusion-eq` (`/fsx/chris.beckham/miniconda3/envs/fusion-eq`) that trained and
predicted every one of Chris's MAST models. Python 3.11.16, torch 2.11.0+cu128.

| File | What it is |
|---|---|
| `environment.yml` | `conda env export` (conda and pip packages, exact versions and builds). Two changes by hand: the PyTorch index URL, which pip needs to find `torch==2.11.0+cu128`, and `nodefaults` in place of the `defaults` channel (every conda package comes from conda-forge; `defaults` would ask for Anaconda's terms of service) |
| `conda-explicit-linux-64.txt` | `conda list --explicit --md5`: exact package URLs and checksums for the conda part |
| `pip-freeze.txt` | `pip list --format=freeze`: every package visible to pip, including those installed by conda |

Rebuild:

```bash
CONDA_CHANNELS=conda-forge CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge \
  conda env create -n fusion-eq -f environment.yml
```

The two variables keep conda off Anaconda's `defaults` channel, which a stock Miniconda configures and which stops
`conda env create` at a terms-of-service prompt; no package here needs it.

Checked by building a new environment from this file on the AWS cluster (4 min): its `pip freeze` is identical to `pip-freeze.txt`, and torch sees CUDA 12.8. Not yet built on a machine outside this cluster.
