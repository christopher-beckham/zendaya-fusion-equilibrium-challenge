"""Download the shipped DIII-D checkpoints (v13, Ch1-safe, Ch1-min) and inference artifacts, and check them against
weights_manifest.json.

    python fetch_weights.py [--root DIR] [--cache CACHE]

Checkpoints land in DIR/my_experiments/<path in the repository> (for example my_experiments/ml/ch1h/runs/prod_auxghc_s1/model.pt),
where the predict_*.sh scripts read them; DIR defaults to this package. Inference artifacts (repository folder artifacts/) land
in CACHE (default $FEQ_CACHE) at the same relative path: the files of the training pipeline that the networks were trained
against and that inference reads again (Ridge base, q95 / βN training inputs and history features, out-of-fold map features,
feature layouts). A from-scratch rebuild makes slightly different copies of these, so shipped-weights inference uses these.
The repository and revision are pinned in the manifest. A token is needed only while the repository is private:
$FEC_HF_TOKEN, else the usual Hugging Face login.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--cache", type=Path, default=Path(os.environ["FEQ_CACHE"]) if os.environ.get("FEQ_CACHE") else None)
    a = ap.parse_args()
    m = json.loads((HERE / "weights_manifest.json").read_text())
    if a.cache is None and any(r.startswith("artifacts/") for r in m["files"]):
        ap.error("set --cache or FEQ_CACHE (the inference artifacts go there)")

    def dest(rel: str) -> Path:
        return a.cache / rel[len("artifacts/"):] if rel.startswith("artifacts/") else a.root / "my_experiments" / rel

    from huggingface_hub import snapshot_download
    tmp = a.root / ".download"
    print(f"downloading {m['repo']} @ {m['revision']} ({len(m['files'])} files)", flush=True)
    snapshot_download(m["repo"], repo_type=m["repo_type"], revision=m["revision"], local_dir=tmp,
                      allow_patterns=list(m["files"]), token=os.environ.get("FEC_HF_TOKEN"), max_workers=8)
    for rel in m["files"]:
        dst = dest(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / rel, dst)
    shutil.rmtree(tmp)                        # the download's own scratch folder (now empty apart from metadata)
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(m["files"], ex.map(lambda r: sha256(dest(r)), m["files"])))
    bad = [r for r, v in m["files"].items() if got[r] != v["sha256"]]
    for r in bad:
        print(f"  MISMATCH {r}", flush=True)
    print(f"{len(m['files']) - len(bad)}/{len(m['files'])} files match", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
