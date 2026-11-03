"""Shared paths and eval-set definitions for the synthetic-MAST S evaluator (synth_S/)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PY = os.environ.get("FEC_PYTHON", sys.executable)
STARTER = Path(__file__).resolve().parents[3]                                      # starter tree: scorer + postproc_qb
MAIN_NN = STARTER / "my_experiments" / "nn"
WT_NN = STARTER.parent / "multimachine" / "my_experiments" / "nn"                  # multimachine tree: models, SynthSet
RUNS = WT_NN / "runs"
HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("FEC_CACHE", "/fsx/chris.beckham/fusion_cache"))
GEN4 = Path(os.environ.get("FEC_GEN4_ROOT", "/fsx/zara.ercan/more_data/m2"))      # Zara's raw gen4 sets (held-out parts)
# Re-buildable data (held-out-only input sets, reference targets, per-checkpoint predictions). On /fsx so it survives
# allocation changes; predictions are deleted after each checkpoint is scored.
SCR = Path(os.environ.get("SYNTHS_SCRATCH", "/fsx/chris.beckham/fusion_cache/synth_S_scratch"))

# eval sets: name -> (source cache tags by region, held-out parquet folder)
SETS = {
    "g4_5000": {"cache": {"limiter": "zara_g4_5000_limiter", "envelope": None},
                "parquet": GEN4 / "gen4_set5000/mast_g4s5000_heldout"},
    # up-down mirror image of the g4_5000 held-out cases (synth_cache_seq.py --mirror), for test-time flip averaging
    "g4_5000m": {"cache": {"limiter": "zara_g4_5000m_limiter", "envelope": None},
                 "parquet": GEN4 / "gen4_set5000/mast_g4s5000_heldout"},
    # Zara's 2026-10-12 sets: 20,000 cases of the gen4_set5000 recipe, and 5,000 cases of recipe v3 (centre-column
    # limited plasmas, independent upper/lower coil factors, structured Thomson variation)
    "g4_20000": {"cache": {"limiter": "zara_g4_20000_limiter", "envelope": None},
                 "parquet": GEN4 / "gen4_set20000/mast_g4s20000_heldout"},
    "g4_v3": {"cache": {"limiter": "zara_g4_5000v3_limiter", "envelope": None},
              "parquet": GEN4 / "gen4_set5000v3/mast_g4s5000v3_heldout"},
    # Zara's 2026-10-13 set, recipe v4: blended coil currents, a hidden outward field instead of extra pressure,
    # wider current profiles (labels shift: betaN, li lower, kappa higher than every earlier set)
    "g4_v4": {"cache": {"limiter": "zara_g4_5000v4_limiter", "envelope": None},
              "parquet": GEN4 / "gen4_set5000v4/mast_g4s5000v4_heldout"},
    # the same held-out cases with a ±80 ms input window (FEC_SYNTH_HALF=16), for models with --history-frames > 1
    "g4_20000h16": {"cache": {"limiter": "zara_g4_20000h16_limiter", "envelope": None},
                    "parquet": GEN4 / "gen4_set20000/mast_g4s20000_heldout"},
    "g4_v4h16": {"cache": {"limiter": "zara_g4_5000v4h16_limiter", "envelope": None},
                 "parquet": GEN4 / "gen4_set5000v4/mast_g4s5000v4_heldout"},
    "g4_v2": {"cache": {"limiter": "zara_g4_v2_limiter", "envelope": "zara_g4_v2"},
              "parquet": GEN4 / "gen4_set500v2/mast_g4s500v2_heldout"},
}
# Further held-out sets registered without editing this file (run_new_set.sh): extra_sets.json maps
# name -> {"limiter_cache": tag, "parquet": folder}.
_EXTRA = HERE / "extra_sets.json"
if _EXTRA.exists():
    import json as _json
    for _n, _v in _json.loads(_EXTRA.read_text()).items():
        SETS.setdefault(_n, {"cache": {"limiter": _v["limiter_cache"], "envelope": None}, "parquet": Path(_v["parquet"])})
# Training origins to exclude for the "_noV1" variants (zara_v1_mast's training split reuses some held-out origins).
V1_TAG = "zara_v1_mast"
