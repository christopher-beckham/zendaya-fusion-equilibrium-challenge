"""Predict one of Chris's MAST models on a MAST test set and post-process it, as the original build scripts did.

    python chris/mast/predict_model.py MODEL [--split mast_public_test] [--out DIR] [--weights DIR] [--workers 12]

MODEL is a key of models.json. Steps, each written to a temporary folder under OUT/MODEL (kept as OUT/MODEL/steps/
with --keep-steps; each step's file is about 1.2 GB):
  mean      every member checkpoint through predict.predict_mast (history step 4, the checkpoint's MAST variant),
            then the float32 mean of the members (build_ensemble.sh)
  physqb    postproc_qb.py: q95 from the predicted map, βN from Thomson (build_candidate.sh / build_ensemble.sh)
  base      βN × 1.2
  q, final  for post C only: apply_cc.py --factor 0.932, then apply_bn_variant.py --variant V2 (build_qb.sh)
  q95_from  q95 replaced by that model's output for the same split (splice_q95.py); build that model first
The result is OUT/MODEL/<split>.npz.

Settings: FEC_DATA_ROOT (folder with data/<split>/), FEC_CACHE (feature caches for the split), FEC_WEIGHTS (folder of
run folders holding the checkpoints; default --weights). --rebuilt uses retrain.py's RUN-rebuild/last.pt instead. FEC_MAST_MASK is set to limiter, as every model was trained.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MM = HERE / "multimachine" / "my_experiments" / "nn"
ST = HERE / "starter" / "my_experiments" / "nn"
MODELS = json.loads((HERE / "models.json").read_text())


def predict_members(members: list, weights: Path, split: str, device: str, rebuilt: bool = False) -> dict:
    """Mean of the members' raw MAST predictions, keyed as in a submission file."""
    sys.path.insert(0, str(MM))
    import torch
    import mast_region
    from common import CACHE, DATA_ROOT
    from model import load_checkpoint
    from predict import predict_mast
    paths = sorted((DATA_ROOT / "data" / split).glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files in {DATA_ROOT / 'data' / split}")
    preds = []
    for run, ck_name in members:
        if rebuilt:                                   # retrain.py output: RUN-rebuild/last.pt
            run, ck_name = f"{run}-rebuild", "last.pt"
        net, ckpt = load_checkpoint(weights / run / ck_name, torch.device(device))
        trained = str(ckpt.get("args", {}).get("mast_mask", "envelope"))
        if mast_region.name() != trained:
            raise SystemExit(f"{run}: trained with MAST region {trained!r}, FEC_MAST_MASK selects {mast_region.name()!r}")
        variant = ckpt.get("args", {}).get("mast_variant", "") or ""
        print(f"{run}/{ck_name}: MAST variant {variant or 'as shipped'}", flush=True)
        preds.append(predict_mast(net, ckpt["norm"], paths, CACHE, torch.device(device), 4, variant=variant,
                                  split=split))
        del net
        torch.cuda.empty_cache()
    keys = list(preds[0])
    assert all(list(p) == keys and all(p[k].shape == preds[0][k].shape for k in keys) for p in preds)
    return {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in keys}


def run(cmd: list[str], log: Path, env: dict) -> None:
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}; see {log}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=[k for k in MODELS if not k.startswith("_")])
    ap.add_argument("--split", default="mast_public_test")
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--weights", type=Path, default=os.environ.get("FEC_WEIGHTS"), help="default: $FEC_WEIGHTS")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-steps", action="store_true", help="keep the intermediate files in OUT/MODEL/steps/")
    ap.add_argument("--rebuilt", action="store_true",
                    help="use each run's retrained checkpoint (WEIGHTS/RUN-rebuild/last.pt, from retrain.py)")
    a = ap.parse_args()
    if a.weights is None:
        ap.error("set FEC_WEIGHTS (source settings.sh first) or pass --weights")
    a.weights = Path(a.weights)
    os.environ["FEC_MAST_SPLIT"] = a.split          # read by every script below (and by predict's imports)
    os.environ["FEC_MAST_MASK"] = "limiter"
    spec = MODELS[a.model]
    out = a.out.resolve() / a.model
    out.mkdir(parents=True, exist_ok=True)
    steps = out / "steps" if a.keep_steps else Path(tempfile.mkdtemp(prefix=".steps-", dir=out))
    for d in ("mean", "physqb", "base", "q", "final"):
        (steps / d).mkdir(parents=True, exist_ok=True)
    name = f"{a.split}.npz"
    py = sys.executable
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    mean = predict_members(spec["members"], a.weights, a.split, a.device, a.rebuilt)
    np.savez(steps / "mean" / name, **mean)
    print(f"averaged {len(spec['members'])} members over {len(mean)} arrays", flush=True)

    run([py, "-u", str(ST / "postproc_qb.py"), str(steps / "mean"), str(steps / "physqb"), "--workers", str(a.workers)],
        steps / "physqb.log", env)
    z = np.load(steps / "physqb" / name)
    arr = {k: np.asarray(z[k] * (1.2 if k.endswith("_betaN") else 1.0), np.float32) for k in z.files}
    np.savez(steps / "base" / name, **arr)
    final = steps / "base"
    if spec["post"] == "C":
        run([py, "-u", str(ST / "q95_study" / "H_v4" / "apply_cc.py"), str(steps / "base"), str(steps / "q"),
             "--factor", "0.932", "--workers", str(a.workers)], steps / "apply_cc.log", env)
        run([py, "-u", str(ST / "q95_study" / "F_betaN" / "apply_bn_variant.py"), str(steps / "q"), str(steps / "final"),
             "--variant", "V2", "--workers", str(a.workers)], steps / "apply_bn.log", env)
        final = steps / "final"
    za = np.load(final / name)
    if "q95_from" in spec:
        zb = np.load(a.out.resolve() / spec["q95_from"] / name)
        assert za.files == zb.files
        arrays = {k: np.asarray(zb[k] if k.endswith("_q95") else za[k], np.float32) for k in za.files}
    else:
        arrays = {k: za[k] for k in za.files}
    np.savez(out / name, **arrays)
    print(f"wrote {out / name}: {sum(k.endswith('_psirz') for k in arrays)} shots", flush=True)
    if not a.keep_steps:                     # this run's own temporary files
        shutil.rmtree(steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
