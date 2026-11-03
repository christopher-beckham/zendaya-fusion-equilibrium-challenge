"""Build + validate Challenge 1 submission files from nn predictions. Never uploads.

    python make_submission_nn.py --preds runs/final_free/preds_test --out ../../../submission/nn_free_v1 \
        [--psi-dtype float32|float16] [--source local|hf] [--check-against ../../../submission/c1_full_phys_v1]

--source local  : DIII-D shots from the local parquet files in sorted order, MAST zeros with the
                  local MAST frame counts. The Hugging Face stream order equals the sorted local
                  file order; --check-against verifies that against an existing streamed build
                  (per-shot frame counts must match for every shot) before trusting it.
--source hf     : stream the test inputs from Hugging Face through submission_skeleton.build_submission
                  (slow, ~1 h) with hook_from_preds matching shots by their efit_times.
The skeleton casts psi to float16; --psi-dtype float32 keeps full precision (validator accepts any float).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
DATA = ROOT / "challenge_data" / "data"
CONFIGS = [("diii_d_public_test", "DIII-D"), ("mast_public_test", "MAST")]


def local_T(config: str) -> list[tuple[str, int]]:
    out = []
    for f in sorted((DATA / config).glob("*.parquet")):
        t = pq.read_table(f, columns=["efit_times"]).column(0).to_pylist()[0]
        out.append((f.stem, len(t)))
    return out


def check_order(ref_dir: Path) -> bool:
    ok = True
    for config, _ in CONFIGS:
        ref = np.load(ref_dir / f"{config}.npz")
        loc = local_T(config)
        n_ref = sum(1 for k in ref.files if k.endswith("_psirz"))
        if n_ref != len(loc):
            print(f"  {config}: {n_ref} shots in {ref_dir.name} vs {len(loc)} local files"); ok = False; continue
        bad = [(i, s, T, ref[f"shot_{i:04d}_psirz"].shape[0]) for i, (s, T) in enumerate(loc)
               if ref[f"shot_{i:04d}_psirz"].shape[0] != T]
        print(f"  {config}: {len(loc)} shots, frame counts mismatching the streamed build: {len(bad)}")
        ok &= not bad
    return ok


def build_local(config: str, machine: str, preds: Path | None, out_dir: Path, dtype) -> Path:
    arrays = {}
    for i, (stem, T) in enumerate(local_T(config)):
        if preds is not None:
            p = np.load(preds / f"{stem}.npz")
            assert p["psi"].shape == (T, 65, 65), (stem, p["psi"].shape, T)
            assert np.allclose(len(p["t"]), T)
            psi, q95, betaN = p["psi"], p["q95"], p["betaN"]
            assert np.isfinite(psi).all(), f"{stem}: non-finite psi"
        else:
            psi, q95, betaN = np.zeros((T, 65, 65), np.float32), np.zeros(T, np.float32), np.zeros(T, np.float32)
        arrays[f"shot_{i:04d}_psirz"] = np.asarray(psi, dtype=dtype)
        arrays[f"shot_{i:04d}_q95"] = np.asarray(q95, np.float32)
        arrays[f"shot_{i:04d}_betaN"] = np.asarray(betaN, np.float32)
    path = out_dir / f"{config}.npz"
    np.savez_compressed(path, **arrays)
    print(f"  {config}: {len(arrays) // 3} shots -> {path}  ({path.stat().st_size / 1e6:.0f} MB, psi {dtype.__name__})")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True, help="predict_test.py output dir (DIII-D test shots)")
    ap.add_argument("--mast-preds", type=Path, default=None, help="MAST test predictions dir (same npz format); zeros if absent")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--psi-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--source", choices=["local", "hf"], default="local")
    ap.add_argument("--check-against", type=Path, default=None, help="streamed build to verify the local order against")
    args = ap.parse_args()
    if any(a.startswith("--repo") for a in sys.argv[1:]):
        raise SystemExit("never uploads here; push with push_predictions.py by hand")
    dtype = np.float32 if args.psi_dtype == "float32" else np.float16
    args.out.mkdir(parents=True, exist_ok=True)

    if args.check_against is not None:
        print(f"checking local sorted order against {args.check_against}:")
        if not check_order(args.check_against):
            raise SystemExit("local order does NOT match the streamed build; use --source hf")
        print("  order verified")

    if args.source == "local":
        for config, machine in CONFIGS:
            build_local(config, machine, args.preds if machine == "DIII-D" else args.mast_preds, args.out, dtype)
    else:
        import submission_skeleton as sk
        import hook_from_preds
        os.environ["NN_PREDS"] = str(args.preds.resolve())
        sk.your_model_predict = hook_from_preds.your_model_predict
        for config, split in sk.TEST_CONFIGS:
            sk.build_submission(config, split, args.out, 0)
        if dtype is np.float32:
            print("note: the skeleton path writes float16 psi; rebuild with --source local for float32")

    from validate_submission import validate
    for config, _ in CONFIGS:
        if validate(args.out / f"{config}.npz", config, 0):
            print(f"validation FAILED for {config}", file=sys.stderr)
            return 1
    print("\nboth files valid. Push by hand: python push_predictions.py --repo <you>/fusion-eq-predictions "
          f"--read-token hf_... --dir {args.out} (add --dry-run first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
