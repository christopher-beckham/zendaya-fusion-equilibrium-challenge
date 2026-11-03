# Third-party code in this repository

Everything in this repository is released under the MIT licence in `LICENSE` (Christopher Beckham and Zara Ercan), except
the files below, which keep their own licence.

| Paths | Origin | Licence |
|---|---|---|
| `fusion_scoring/`, `data_fixes.py`, `local_score.py`, `validate_submission.py`, `submission_skeleton.py`, `push_predictions.py` in `zara/diii-d/`, `chris/mast/starter/` and `chris/mast/multimachine/`; `fusion_scoring/`, `data_fixes.py` and `local_score.py` in `zara/mast/r1/starter/`; `zara/datagen/shared/starter/data_fixes.py` | copied from the organisers' starter kit of the Fusion Equilibrium Challenge | MIT, Copyright (c) 2026 Sophelio and the Fusion Equilibrium Challenge authors; full text in `zara/diii-d/LICENSE` |
| `chris/mast/gsolve/solver/` | gsolve, a Grad-Shafranov solver by Artur Toshev, who is not a team member and released it for this package (`chris/mast/gsolve/SOURCES.md`) | MIT, Copyright (c) 2026 Artur Toshev; full text in `chris/mast/gsolve/solver/LICENSE` |

The challenge datasets are not in this repository; they are downloaded from Hugging Face under their own terms.

The team's Hugging Face repositories named in `REPRODUCE.md` (weights, predictions, generated data) are released under the
MIT licence of this repository. Files derived from the challenge data (the DIII-D caches and artefacts, and the synthetic
MAST sets built from the `mast_public_test` inputs) are also subject to the challenge data's CC BY 4.0 licence (Sophelio
and the Fusion Equilibrium Challenge organisers).
