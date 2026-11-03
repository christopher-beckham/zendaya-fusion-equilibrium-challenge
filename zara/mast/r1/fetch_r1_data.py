"""Download r1's training data into R1_ROOT, pinned in r1_data.json (repository, revision, SHA256 per file).

    python zara/mast/r1/fetch_r1_data.py --root $R1_ROOT [--no-cpu-caches]

1. Zara's data bundle (beckhamc/zara-r1-a100-bundle): synthetic MAST records, the DIII-D training inputs and reference
   targets r1 reads, machine matrices. Its six tar parts are downloaded one at a time, checked by SHA256 and streamed into
   tar, so they are never on disk together. The bundle's GPU-built first-pass caches and Zara's own base model are not
   unpacked (r1_data.json, "not_unpacked"): r1 is trained from random weights on caches built on the CPU.
2. Those CPU-built caches (beckhamc/zara-r1-cpu-caches), one tar, unpacked into place. With --no-cpu-caches this step is
   skipped, for rebuilding them instead with zara/mast/r1/cpu_caches/build_r1_caches.sh.
Disk: about 37 GB unpacked plus 13.2 GB of CPU caches; the largest single download is 13.2 GB. The repositories are public:
no token is needed ($FEC_HF_TOKEN is used if set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(src: dict, name: str, tmp: Path) -> Path:
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(src["repo"], name, repo_type=src["repo_type"], revision=src["revision"], local_dir=tmp,
                             token=os.environ.get("FEC_HF_TOKEN") or None))
    got = sha256(p)
    if got != src["files"][name]:
        raise SystemExit(f"SHA256 mismatch: {name} ({got})")
    print(f"  OK {name}", flush=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--no-cpu-caches", action="store_true", help="skip step 2 (the caches are rebuilt instead)")
    a = ap.parse_args()
    m = json.loads((HERE / "r1_data.json").read_text())
    root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True); tmp = root / ".download"

    b = m["bundle"]
    print(f"1. {b['repo']} @ {b['revision'][:8]}: {len(b['files'])} parts -> {root}", flush=True)
    excl = [f"--exclude={x}" for x in m["not_unpacked"]]
    tar = subprocess.Popen(["tar", "-x", "--no-same-owner", "-C", str(root), "--strip-components=1", *excl], stdin=subprocess.PIPE)
    for name in sorted(b["files"]):
        p = download(b, name, tmp)
        with open(p, "rb") as f:
            shutil.copyfileobj(f, tar.stdin, 1 << 24)
        p.unlink()
    tar.stdin.close()
    if tar.wait():
        raise SystemExit("tar failed on the bundle")

    if not a.no_cpu_caches:
        c = m["cpu_caches"]
        print(f"2. {c['repo']} @ {c['revision'][:8]}: CPU-built training caches -> {root}", flush=True)
        for name in c["files"]:
            p = download(c, name, tmp)
            subprocess.run(["tar", "-x", "--no-same-owner", "-C", str(root), "-f", str(p)], check=True)
            p.unlink()
    shutil.rmtree(tmp, ignore_errors=True)

    bad = 0
    for d, n in m["expected_files"].items():
        if a.no_cpu_caches and d in m["cpu_caches"]["folders"]:
            continue
        got = sum(1 for _ in (root / d).iterdir()) if (root / d).is_dir() else 0
        bad += got != n
        print(f"  {d}: {got} files" + ("" if got == n else f"  (expected {n})"), flush=True)
    print("r1 data ready" if not bad else f"{bad} folders differ from r1_data.json", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
