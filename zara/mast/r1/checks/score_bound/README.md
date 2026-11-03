# Label-free comparison of two MAST prediction files

`score_bound.py` compares two prediction files of the same split without labels and bounds how far the composite S of the
second can be from the first's (method in its header). `score_estimate.py` reads its result and gives the expected change
in S when the differences between the files are unrelated to the errors against the labels, as for the last-digit rounding
of another GPU type. Both run in `zara/environment/fusion-equilibrium-starter.yml` (any CPU machine); `score_bound.py` reads
the scorer's files in `chris/mast/starter/` and uses the MAST mask only as the scorer does.

    python score_bound.py A.npz B.npz --machine mast --json RESULT.json --workers 16
    python score_estimate.py RESULT.json

A is always the submitted `mast_private_test.npz` of B (SHA256 `3300302e…`). The three results quoted in the package:

| Result | File B | Strict bound (stand-in / no assumption) | Quoted in |
|---|---|---|---|
| `firstpass_4threads_vs_submitted.json` | r1's first pass of the 1,205 Final MAST shots rebuilt with 8 shards of 4 threads, then r1's chain with the shipped networks and the blend with Chris's shipped member files (30 October) | 2.5e-4 / 1.1e-3 | `SUBMISSION_B.md`, "Checks" |
| `r1_retrained_vs_submitted.json` | the blend with r1 retrained from rebuilt training caches and Chris's 8 shipped member files (31 October) | 0.050 / 0.090 | `SUBMISSION_B.md`, "Checks" and "What to expect when B is retrained" |
| `checks/a100-final/logs/timing/score_bound_B_mast_shipped_a100.json` | `predict.sh B` with the shipped weights on an A100 (SHA256 `424ee809…`) | 1.5e-3 / 2.6e-3 | `checks/a100-final/README.md`, `SUBMISSION_B.md`, `REPRODUCE.md` step 4 |

`score_estimate.py` is quoted for the A100 file only: on average 2e-5, one standard deviation above 6e-5, two above 1.0e-4.
It does not apply to a retrained model, whose differences from the shipped one are not independent of its errors.
