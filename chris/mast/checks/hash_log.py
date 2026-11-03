"""Write HASHES.md: SHA256 of both sides of every byte-level comparison recorded in checks/, so each IDENTICAL can be
checked against a fingerprint after the scratch copies are removed.

    python chris/mast/checks/hash_log.py > chris/mast/checks/HASHES.md

Paths are those of the 21 October 2026 checks on the AWS cluster. A row whose file no longer exists is reported as
missing, so regenerate the log before removing anything it covers.
"""
from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ST = Path("/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions")
TB = Path("/fsx/chris.beckham/fusion_cache/team_blend_fetch")
ZC = Path("/fsx/zara.ercan/cache/ch2/g4/chris_subs")
ZB = Path("/fsx/zara.ercan/cache/ch2/g4/blend")
FC = Path("/fsx/chris.beckham/fusion_cache")
S2 = Path("/fsx/chris.beckham/fec_upload/step2")
RB = Path("/fsx/chris.beckham/fec_cache/rebuild_d3d")
PC = Path("/fsx/chris.beckham/fec_cache/public_check")
DR = Path("/fsx/chris.beckham/fec_dryrun")
MP = "mast_public_test.npz"

ORIG = {"Z": TB / "Z", "Z2": TB / "Z2", "Z6b": TB / "Z6b", "Z7": TB / "Z7", "Z3": ZC / "Z3", "Z8": ZC / "Z8",
        "Z9qb": ST / "mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1",
        "Z15": ST / "mastZ15-mm-v6mix-x4qb-q95cc0932-bnV2-limiter-c1",
        "Z17": ST / "mastZ17-mm-C3-condgsolveonlyqb-q95cc0932-bnV2-limiter-c1",
        "Z18": ST / "mastZ18-mm-Z9dataPlusCond-x3qb-q95cc0932-bnV2-limiter-c1",
        "FA": ST / "mastFA-mm-Z15dataPlusgsolve_cond3_1000-x3qb-q95cc0932-bnV2-limiter-c1",
        "FC": ST / "mastFC-mm-gsolve_cond3_1000onlyqb-q95cc0932-bnV2-limiter-c1",
        "V7C": ST / "mastV7C-mm-V7onlyqb-q95cc0932-bnV2-limiter-c1"}
BASE = {"Z15": ST / "mastZ15-mm-v6mix-x4-base", "Z17": ST / "mastZ17-mm-C3-condgsolveonly-base",
        "Z18": ST / "mastZ18-mm-Z9dataPlusCond-x3-base", "FA": ST / "mastFA-mm-Z15dataPlusgsolve_cond3_1000-x3-base",
        "FC": ST / "mastFC-mm-gsolve_cond3_1000only-base", "V7C": ST / "mastV7C-mm-V7only-base"}
CACHE_FILES = ["READY", "meta.json", "shots.parquet", "frames.parquet", "score_subset.parquet", "coils.npy", "scal.npy",
               "phase.npy", "psi.npy", "features_v2/pictures_d3d.npz", "features_v2/pictures_mast.npz",
               "features_v5.npy", "features_v5.json", "features_v5_diii_d_public_test.npy",
               "features_v5_diii_d_public_test.parquet", "features_v5_mast_public_test.npy",
               "features_v5_mast_public_test.parquet", "thomson_raw_d3d.npz", "thomson_raw_ne_d3d.npz",
               "thomson_raw_mast.npz", "thomson_raw_ne_mast.npz", "lindec_G_d3d.npz", "lindec_G_mast_limiter.npz",
               "features_v2/pictures_mast_sol0.5.npz", "features_v5_mast_public_test_sol0.5.npy",
               "features_v5_mast_public_test_sol0.5.parquet", "features_v5_mast_public_test_sol0.5_limiter.npy",
               "features_v5_mast_public_test_sol0.5_limiter.parquet", "mast_inputs.npz", "norm_train.npz",
               "features_v5_norm.npz"]
MAST_INPUT_FILES = ["features_v5_mast_public_test.npy", "features_v5_mast_public_test.parquet",
                    "features_v2/pictures_mast_sol0.5.npz", "features_v5_mast_public_test_sol0.5.npy",
                    "features_v5_mast_public_test_sol0.5.parquet", "features_v5_mast_public_test_sol0.5_limiter.npy",
                    "features_v5_mast_public_test_sol0.5_limiter.parquet", "mast_inputs.npz", "thomson_raw_mast.npz",
                    "thomson_raw_ne_mast.npz"]


def sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def shards_hash(d: Path) -> str | None:
    """SHA256 of the sorted '<sha256>  <name>' lines of every file in d."""
    if not d.is_dir():
        return None
    files = sorted(d.iterdir())
    with ThreadPoolExecutor(24) as ex:
        hs = list(ex.map(sha, files))
    return hashlib.sha256("".join(f"{h}  {f.name}\n" for h, f in zip(hs, files)).encode()).hexdigest() + \
        f" ({len(files)} files)"


def table(title: str, note: str, rows: list[tuple[str, Path | str, Path | str]], ex) -> None:
    print(f"## {title}\n\n{note}\n")
    print("| Item | Reference SHA256 | Reproduced SHA256 | Match |")
    print("|---|---|---|---|")
    jobs = [(name, ex.submit(sha, a) if isinstance(a, Path) else None, ex.submit(sha, b) if isinstance(b, Path) else None,
             a, b) for name, a, b in rows]
    for name, fa, fb, a, b in jobs:
        ha = fa.result() if fa else a
        hb = fb.result() if fb else b
        m = "missing" if ha is None or hb is None else ("yes" if ha == hb else "**NO**")
        print(f"| {name} | `{ha or 'missing'}` | `{hb or 'missing'}` | {m} |")
    print()


def main() -> int:
    print("# Verification log: file fingerprints\n")
    print("SHA256 of both sides of every byte-level comparison in `checks/`, taken on 21 October 2026 on the AWS cluster "
          "(generated by `checks/hash_log.py`). The reference is the frozen original on `/fsx`; the reproduced file is "
          "the one the check produced. Paths are listed at the end.\n")
    with ThreadPoolExecutor(16) as ex:
        table("Models predicted from the original checkpoints (step 2.3)",
              "`predict_model.py MODEL` on `mast_public_test`, against the file the team blends read.",
              [(f"{m} final", ORIG[m] / MP, S2 / "pred_public" / m / MP) for m in ORIG], ex)
        table("Intermediate -base files (step 2.3)",
              "The file after `postproc_qb.py` and βN × 1.2, before the q95 rule and βN V2.",
              [(f"{m} base", BASE[m] / MP, S2 / "pred_public" / m / "steps" / "base" / MP) for m in BASE], ex)
        table("Post-processing checks (steps 2.1 to 2.3)",
              "FC (and Z15) rebuilt from the kept -base folder with `postproc_check.sh`.",
              [(f"{d}", ORIG[m] / MP, S2 / d / "final" / MP) for d, m in
               (("baseline_FC", "FC"), ("s21_unset", "FC"), ("s21_alt", "FC"), ("s22_FC", "FC"), ("s22_Z15", "Z15"),
                ("s23_FC_default", "FC"))], ex)
        table("Shipped weights and the from-scratch dry run (steps 2.6, 2.8)",
              "Predictions from the downloaded weights, and V7C's original checkpoint through the fresh clone and "
              "environment.",
              [("V7C from downloaded weights", ORIG["V7C"] / MP, S2 / "pred_weights" / "V7C" / MP),
               ("Z15 from downloaded weights", ORIG["Z15"] / MP, S2 / "pred_weights" / "Z15" / MP),
               ("V7C dry run, original checkpoint", ORIG["V7C"] / MP, DR / "pred_orig" / "V7C" / MP)], ex)
        table("Derived caches rebuilt from the official data (steps 2.4, 2.8)",
              "Reference: `/fsx/chris.beckham/fusion_cache`. Reproduced: `/fsx/chris.beckham/fec_cache/rebuild_d3d` "
              "(the normalisation statistics were written there by the dry run's training). `shards/` is one hash over "
              "the sorted per-file SHA256 lines.",
              [(f, FC / f, RB / f) for f in CACHE_FILES] + [("shards/", shards_hash(FC / "shards"),
                                                              shards_hash(RB / "shards"))], ex)
        table("MAST public-test inputs rebuilt into a fresh cache (step 2.3)",
              "Reference: `/fsx/chris.beckham/fusion_cache`. Reproduced: `/fsx/chris.beckham/fec_cache/public_check`.",
              [(f, FC / f, PC / f) for f in MAST_INPUT_FILES], ex)
        print("## Files kept as the record\n")
        print("| File | SHA256 |\n|---|---|")
        for name, p in [(f"Zara's blend {b}", ZB / f"{b}.npz") for b in ("s1fc", "s2v7c")] + \
                       [(f"blend {b} rebuilt from {s}", S2 / "blends" / f"{b}_{s}.npz")
                        for b in ("s1fc", "s2v7c") for s in ("pred_public", "pred_rebuilt")]:
            print(f"| {name} | `{sha(p) or 'missing'}` |")
        print("\nThe blends rebuilt from the original model files equal Zara's array for array (compressed bytes differ "
              "with the Python environment); see `step1_models_and_blends.result.txt`. The 13 Final prediction files "
              "are fingerprinted in `step2/step2_3_predict_private.result.txt`, the shipped weights in "
              "`data/weights_manifest.json`, the synthetic data in `data/manifest.json`.\n")
    print("## Paths\n")
    for name, p in [("Original model files", "submissions/ folders in the starter, team_blend_fetch/ (Z, Z2, Z6b, Z7), "
                     "Zara's chris_subs/ (Z3, Z8)"), ("Reproduced model files", str(S2 / "pred_public")),
                    ("Check scratch", str(S2)), ("Rebuilt caches", str(RB)), ("Dry run", str(DR))]:
        print(f"- {name}: `{p}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
