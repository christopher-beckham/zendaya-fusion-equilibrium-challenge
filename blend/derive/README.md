# How the blend weights were derived

`../weights_chris8r1.json` (the MAST blend of submission B) is a fixed setting: it is computed once, before any
training, and is never used as weights, an initialisation or predictions of a network. CPU only.

Inputs:

- the Development leaderboard scores of the member models' `mast_public_test` files and of earlier team blends
  (`inputs/board_team3_exact.json`, `inputs/board_s3.json`; ψ, q95 and βN R², D_LCFS and the seven Consistency terms, as
  reported by the organisers' scorer). FC and V7C were not submitted alone: their entries, marked `pinned`, are solved
  from the scored blends that contain them;
- label-free statistics between the members' `mast_public_test` prediction files (`code/pairs2.py`, `code/add_pairs.py`:
  pooled ψ differences, per-frame q95 and βN, the scorer's seven derived shape scalars, LCFS distances). No MAST label is read;
- the earlier team blends used as calibration (`inputs/wS1_FC.json`, `inputs/wS2_V7C.json`, `inputs/wT2b.json`,
  `inputs/wT3c.json`) and the training cost of each member (`inputs/costs4.json`).

Method (`code/fpb.py`, `code/search.py`): the error of a frame-by-frame linear blend equals the weighted errors of its
members minus half their weighted disagreement with each other. The score of one scored team blend and those of its members
therefore give the total sum of squares of the public-fold labels for ψ, q95 and βN (`fpb.py`, lines 182-190), and with it
the leaderboard R² of these three quantities for any blend of scored members follows exactly from the members' R² and the
pairwise differences. The shape terms and D_LCFS are estimated from the blended maps with the scorer's own code, calibrated
on the scored team blends. Full description: `../../SUBMISSION_B.md`, section "Development feedback and checkpoint
selection". `search.py` searched the weights over the pool
FA, FC, V7C, Z, Z15, Z2, Z3, Z8 and r1 (FC and V7C barred from βN), on 27 October:

```bash
python search.py --out <dir> --rounds 3 --pool FA,FC,V7C,Z,Z15,Z2,Z3,Z8,r1 --qpool FA,FC,V7C,Z,Z15,Z2,Z3,Z8,r1 --no-t3c \
  --extra-pairs <add_pairs output> --board-extra board_s3.json --blends s1fc=wS1_FC.json,s2v7c=wS2_V7C.json \
  --bar betaN:FC,betaN:V7C
```

Its `best_weights.json` is `../weights_chris8r1.json`. `code/budget2.py` ranked member subsets by training cost (`costs4.json`)
to choose the pool. The pair-statistics files are large and not in Git; `add_pairs.py` regenerates them from the members'
public-test prediction files.

The `pinned` entries of FC and V7C in `inputs/board_s3.json` were solved by `code/pin.py` from the board results of the
scored blends `s1fc` and `s2v7c` (their entries in the same file):

```bash
python pin.py --member FC --weights wS1_FC.json --result <s1fc board numbers> --board board_s3.json --out board_pin_FC.json
python pin.py --member V7C --weights wS2_V7C.json --result <s2v7c board numbers> --board board_s3.json --out board_pin_V7C.json
```

Rerun on 3 November, both commands reproduce the `pinned` values in `board_s3.json` exactly (ψ, q95, D_LCFS and the seven
Consistency terms).
