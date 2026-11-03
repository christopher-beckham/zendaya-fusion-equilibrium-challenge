"""Download r1's shipped weights and inference data, pinned in r1_files.json (repository, revision, SHA256 per file).

    python zara/mast/r1/predict/fetch_r1.py --root DIR [--data-root $R1_ROOT/cache]
    python zara/mast/r1/predict/fetch_r1.py --firstpass-only [--data-root $R1_ROOT/cache]

runs/<member>/ (model.pt, args.json, hist.json) land in DIR/runs/: give DIR/runs to predict_r1.sh. data/ lands in the data root
the r1 code reads from (pegs/machines/ and mast/mast_public_test/basis.npz under $R1_ROOT/cache): a file that is already
there is checked and kept, never overwritten. Then r1's first-pass cache of the Final MAST inputs (r1_final_firstpass.json):
the cache the submitted file was predicted with, unpacked under the data root, so predict_r1.sh does not rebuild it (about
2 hours on 96 CPU cores); a complete copy already there is kept. --firstpass-only downloads that cache alone, without the
networks: for the copy that predicts with the retrained networks (REPRODUCE.md step 6). A token is needed only while the
repositories are private: $FEC_HF_TOKEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
R1_ROOT = os.environ.get("R1_ROOT")   # r1's data folder (settings.sh; zara/mast/r1/README.md)

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path)
    ap.add_argument("--data-root", type=Path, help="default: $R1_ROOT/cache")
    ap.add_argument("--firstpass-only", action="store_true", help="only the first-pass cache of the Final MAST inputs")
    a = ap.parse_args()
    if a.root is None and not a.firstpass_only:
        ap.error("--root is required unless --firstpass-only")
    if a.data_root is None:
        if not R1_ROOT:
            ap.error("set R1_ROOT (source settings.sh first) or pass --data-root")
        a.data_root = Path(R1_ROOT) / "cache"
    m = json.loads((HERE / "r1_files.json").read_text())
    token = os.environ.get("FEC_HF_TOKEN") or None
    for rel, v in ({} if a.firstpass_only else m["files"]).items():
        dest = (a.root / rel) if rel.startswith("runs/") else (a.data_root / rel.removeprefix("data/"))
        if dest.exists():
            if sha256(dest) != v["sha256"]:
                raise SystemExit(f"{dest} exists with a different SHA256; not overwritten")
            print(f"OK (present) {dest}")
            continue
        src = hf_hub_download(m["repo"], rel, repo_type=m["repo_type"], revision=m["revision"], token=token)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        if sha256(dest) != v["sha256"]:
            raise SystemExit(f"SHA256 mismatch: {dest}")
        print(f"OK {dest}")
    fp = json.loads((HERE / "r1_final_firstpass.json").read_text())
    folder = a.data_root / fp["folder"]
    have = sum(1 for f in folder.glob("*.npz")) if folder.is_dir() else 0
    if have == fp["n_files"]:
        print(f"OK (present) {folder}: {have} files")
        return 0
    dl = (a.root or a.data_root) / ".download"
    tar = Path(hf_hub_download(fp["repo"], fp["file"], repo_type=fp["repo_type"], revision=fp["revision"], token=token,
                               local_dir=dl))
    if sha256(tar) != fp["sha256"]:
        raise SystemExit(f"SHA256 mismatch: {tar}")
    a.data_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "--no-same-owner", "-C", str(a.data_root), "-f", str(tar)], check=True)
    shutil.rmtree(dl)
    print(f"OK {folder}: {sum(1 for f in folder.glob('*.npz'))} files (first-pass cache of the Final MAST inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
