"""Check files built on any machine against the reference fingerprints in HASHES.md.

    python chris/mast/checks/verify_hashes.py --cache DIR          # derived caches (build_caches.sh output)
    python chris/mast/checks/verify_hashes.py --predictions DIR    # DIR/<model>/mast_public_test.npz (predict.sh output)

Uses the "Reference SHA256" column of HASHES.md: the derived-cache table for --cache (including the combined hash of
shards/) and the model table for --predictions. Files not present are reported as missing (the normalisation
statistics, for example, only exist after the first training run). Exit status 1 if any present file differs.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def shards_hash(d: Path) -> str | None:
    if not d.is_dir():
        return None
    files = sorted(d.iterdir())
    with ThreadPoolExecutor(16) as ex:
        hs = list(ex.map(sha, files))
    return hashlib.sha256("".join(f"{h}  {f.name}\n" for h, f in zip(hs, files)).encode()).hexdigest() + \
        f" ({len(files)} files)"


def table(section: str) -> dict[str, str]:
    """Item -> reference SHA256 from the HASHES.md table under the heading that starts with `section`."""
    text = (HERE / "HASHES.md").read_text()
    block = text.split(f"## {section}", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| (.+?) \| `([^`]+)` \| `[^`]+` \| \w+ \|$", block, re.M)
    return dict(rows)


def report(title: str, expected: dict[str, str], got: dict[str, str | None]) -> int:
    bad = 0
    print(f"## {title}")
    for k, ref in expected.items():
        g = got[k]
        state = "missing" if g is None else ("MATCH" if g == ref else "DIFFERENT")
        bad += state == "DIFFERENT"
        print(f"  {state:9s} {k}")
    n = sum(got[k] == expected[k] for k in expected)
    print(f"  {n}/{len(expected)} match, {sum(got[k] is None for k in expected)} missing, {bad} different\n")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--predictions", type=Path)
    a = ap.parse_args()
    bad = 0
    if a.cache:
        ref = table("Derived caches rebuilt")
        files = [k for k in ref if k != "shards/"]
        with ThreadPoolExecutor(16) as ex:
            got = dict(zip(files, ex.map(lambda f: sha(a.cache / f), files)))
        got["shards/"] = shards_hash(a.cache / "shards")
        bad += report(f"derived caches in {a.cache}", ref, got)
    if a.predictions:
        ref = {k[:-len(" final")]: v for k, v in table("Models predicted from the original checkpoints").items()}
        with ThreadPoolExecutor(16) as ex:
            got = dict(zip(ref, ex.map(lambda m: sha(a.predictions / m / "mast_public_test.npz"), ref)))
        bad += report(f"model predictions in {a.predictions}", ref, got)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
