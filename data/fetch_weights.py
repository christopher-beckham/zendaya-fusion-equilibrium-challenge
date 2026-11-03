"""Download the shipped checkpoints of Chris's MAST models and check them against data/weights_manifest.json.

    python data/fetch_weights.py --root DIR

Files land in DIR/<run>/<checkpoint> (the repository's runs/ prefix is dropped), so DIR is what predict_model.py and
predict.sh take as --weights or $FEC_WEIGHTS. The repository and revision are pinned in the manifest. A token is needed
only while the repository is private: $FEC_HF_TOKEN, else the usual Hugging Face login.
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
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    m = json.loads((HERE / "weights_manifest.json").read_text())
    from huggingface_hub import snapshot_download
    tmp = a.root / ".download"
    print(f"downloading {m['repo']} @ {m['revision']} ({len(m['files'])} files)", flush=True)
    snapshot_download(m["repo"], repo_type=m["repo_type"], revision=m["revision"], local_dir=tmp,
                      allow_patterns=list(m["files"]), token=os.environ.get("FEC_HF_TOKEN"), max_workers=8)
    for rel in m["files"]:
        dst = a.root / rel[len("runs/"):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / rel, dst)
    shutil.rmtree(tmp)                        # the download's own scratch folder (now empty apart from metadata)
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(m["files"], ex.map(lambda r: sha256(a.root / r[len("runs/"):]), m["files"])))
    bad = [r for r, v in m["files"].items() if got[r] != v["sha256"]]
    for r in bad:
        print(f"  MISMATCH {r}", flush=True)
    print(f"{len(m['files']) - len(bad)}/{len(m['files'])} checkpoints match", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
