"""Download the generated training data from Hugging Face, check every file against data/manifest.json, and rebuild the
merged training caches.

    python data/fetch.py [--root DIR] [--only TAG ...] [--gsolve-raw] [--no-merge]

- The data is the dataset repo named in the manifest, pinned to one commit. Files land in ROOT with the repo's layout:
  ROOT/caches/<tag>/{arrays.npz,shape_targets.npz,manifest.json}, ROOT/gsolve_raw/<set>/...
- ROOT defaults to $FEC_DATA_ROOT, else data/cache next to this script.
- ROOT/synth is linked to ROOT/caches, so with --root $FEC_CACHE training finds the caches at FEC_CACHE/synth/<tag>.
- Merged caches (manifest "merged") are rebuilt from their base caches with Chris's synth_merge.py, unchanged, and
  written to ROOT/caches/<tag>; synth_merge.py runs with its CACHE pointed at ROOT.
- --only limits the work to these caches (a merged tag pulls in its base caches). --gsolve-raw also downloads the raw
  gsolve output, checked against data/checksums/gsolve_raw.sha256.
- Run it in Chris's environment (chris/environment): the merge needs numpy and the modules synth_merge.py imports.
- A token is needed only while the repo is private: $FEC_HF_TOKEN, else the usual Hugging Face login.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGE_DIR = HERE.parent / "chris" / "mast" / "multimachine" / "my_experiments" / "nn"   # synth_merge.py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check(root: Path, expected: dict[str, str], what: str) -> None:
    """expected: path relative to root -> sha256. Exits on any missing or different file."""
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(expected, ex.map(lambda r: sha256(root / r) if (root / r).is_file() else None, expected)))
    bad = [r for r in expected if got[r] != expected[r]]
    for r in bad:
        print(f"  {'MISSING' if got[r] is None else 'MISMATCH'} {r}", flush=True)
    print(f"{what}: {len(expected) - len(bad)}/{len(expected)} files match", flush=True)
    if bad:
        raise SystemExit(f"{what}: {len(bad)} files missing or different")


def merge(root: Path, out_tag: str, tags: list[str]) -> None:
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import synth_merge; "
            "synth_merge.CACHE = Path(sys.argv[2]); sys.argv = ['synth_merge.py', *sys.argv[3:]]; "
            "raise SystemExit(synth_merge.main())")
    env = {**os.environ, "FEC_MAST_MASK": "limiter"}
    subprocess.run([sys.executable, "-c", code, str(MERGE_DIR), str(root), out_tag, *tags], env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("FEC_DATA_ROOT", HERE / "cache")))
    ap.add_argument("--only", nargs="+", default=None, help="cache tags (base or merged); default: all")
    ap.add_argument("--gsolve-raw", action="store_true", help="also download the raw gsolve output")
    ap.add_argument("--no-merge", action="store_true", help="download and check the base caches only")
    a = ap.parse_args()

    m = json.loads((HERE / "manifest.json").read_text())
    merged = m["merged"]
    wanted = set(a.only) if a.only else set(m["caches"]) | set(merged)
    unknown = wanted - set(m["caches"]) - set(merged)
    if unknown:
        raise SystemExit(f"unknown cache tags: {sorted(unknown)}")
    to_merge = [] if a.no_merge else [t for t in merged if t in wanted]
    base = sorted({t for t in wanted if t in m["caches"]} | {s for t in to_merge for s in merged[t]["from"]})

    expected = {f"caches/{t}/{f}": v["sha256"] for t in base for f, v in m["caches"][t].items()}
    patterns = list(expected)
    if a.gsolve_raw:
        raw = {}
        for line in (HERE / m["gsolve_raw"]["checksums"]).read_text().splitlines():
            h, p = line.split(maxsplit=1)
            raw[p] = h
        patterns += [f"{p}/*" for p in m["gsolve_raw"]["folders"]]

    from huggingface_hub import snapshot_download
    a.root.mkdir(parents=True, exist_ok=True)
    print(f"downloading {m['repo']} @ {m['revision']} into {a.root}", flush=True)
    snapshot_download(m["repo"], repo_type=m["repo_type"], revision=m["revision"], local_dir=a.root,
                      allow_patterns=patterns, token=os.environ.get("FEC_HF_TOKEN"), max_workers=8)
    check(a.root, expected, "base caches")
    link = a.root / "synth"                   # training reads <cache>/synth/<tag>
    if not link.exists():
        link.symlink_to("caches")
    if a.gsolve_raw:
        check(a.root, raw, "raw gsolve output")

    for t in to_merge:
        out = {f"caches/{t}/{f}": v["sha256"] for f, v in merged[t]["files"].items()}
        if all((a.root / r).is_file() for r in out):
            print(f"{t}: already built, checking", flush=True)
        else:
            print(f"{t}: merging {merged[t]['from']}", flush=True)
            merge(a.root, t, merged[t]["from"])
        check(a.root, out, t)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
