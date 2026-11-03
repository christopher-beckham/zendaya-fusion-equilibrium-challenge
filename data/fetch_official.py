"""Download the official challenge data at pinned revisions, or check an existing copy, against the SHA256 that Hugging
Face publishes for every file.

    python data/fetch_official.py --root DIR [--splits diii_d_train diii_d_public_test mast_public_test mast_private_test]
    python data/fetch_official.py --root DIR --verify-local [--splits ...]     # check DIR without downloading

Each split lands in ROOT/<dataset>/data/<split>/ (the layout FEC_DATA_ROOT expects: point it at ROOT/<dataset>).
Datasets and revisions:
  development  Sophelio/fusion-equilibrium-challenge        diii_d_train, diii_d_public_test, mast_public_test
  final        ganatma/fusion-equilibrium-challenge-final   diii_d_private_test, mast_private_test
Fixes in the organisers' data_fixes.py are applied when the files are read, not to the files.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = {
    "development": ("Sophelio/fusion-equilibrium-challenge", "1e280905b8"),
    "final": ("ganatma/fusion-equilibrium-challenge-final", "965040c12b814f33c8e7209f6b1799932247e793"),
}
SPLIT_DATASET = {"diii_d_train": "development", "diii_d_public_test": "development", "mast_public_test": "development",
                 "diii_d_private_test": "final", "mast_private_test": "final"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_DATASET))
    ap.add_argument("--verify-local", action="store_true", help="check the files already in ROOT; download nothing")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    bad_total = 0
    for name, (repo, rev) in DATASETS.items():
        splits = [s for s in a.splits if SPLIT_DATASET[s] == name]
        if not splits:
            continue
        rev = api.dataset_info(repo, revision=rev).sha                   # full commit hash
        files = [f for f in api.dataset_info(repo, revision=rev, files_metadata=True).siblings
                 if any(f.rfilename.startswith(f"data/{s}/") for s in splits)]
        dest = a.root / name
        if not a.verify_local:
            print(f"{repo} @ {rev}: downloading {len(files)} files into {dest}", flush=True)
            snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=dest,
                              allow_patterns=[f"data/{s}/*" for s in splits], max_workers=8)
        want = {f.rfilename: (f.lfs.sha256 if f.lfs else None) for f in files}
        with ThreadPoolExecutor(a.workers) as ex:
            got = dict(zip(want, ex.map(lambda r: sha256(dest / r) if (dest / r).is_file() else None, want)))
        for s in splits:
            keys = [r for r in want if r.startswith(f"data/{s}/")]
            missing = [r for r in keys if got[r] is None]
            bad = [r for r in keys if got[r] is not None and want[r] is not None and got[r] != want[r]]
            extra = sorted({p.name for p in (dest / "data" / s).glob("*")} - {Path(r).name for r in keys}) \
                if (dest / "data" / s).is_dir() else []
            bad_total += len(missing) + len(bad) + len(extra)
            print(f"{repo} @ {rev[:10]} {s}: {len(keys) - len(missing) - len(bad)}/{len(keys)} files match"
                  + (f", {len(missing)} missing" if missing else "") + (f", {len(bad)} different" if bad else "")
                  + (f", {len(extra)} extra local files" if extra else ""), flush=True)
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
