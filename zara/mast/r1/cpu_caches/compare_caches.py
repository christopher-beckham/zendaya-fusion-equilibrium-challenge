"""Compare a rebuilt cache folder with a reference copy, file by file.

    python compare_caches.py NEW_DIR REF_DIR [--only-new]

Every .npz/.npy in NEW_DIR is compared with the file of the same name in REF_DIR: first byte for byte; if the bytes differ,
array by array (same keys, same shapes, equal values with NaN == NaN), and otherwise the largest absolute difference per key.
Files present on one side only are counted (with --only-new, files missing from NEW_DIR are not reported: use it to check
one shard of a cache against the full reference). Exit status 0 only when every compared file is byte-identical.
"""
import argparse
import filecmp
import os
import sys

import numpy as np


def load(p):
    a = np.load(p, allow_pickle=False)
    return {k: np.asarray(a[k]) for k in a.files} if hasattr(a, "files") else {"": np.asarray(a)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new")
    ap.add_argument("ref")
    ap.add_argument("--only-new", action="store_true")
    a = ap.parse_args()
    ext = (".npz", ".npy")
    new = sorted(f for f in os.listdir(a.new) if f.endswith(ext))
    ref = set(f for f in os.listdir(a.ref) if f.endswith(ext))
    missing_ref = [f for f in new if f not in ref]
    missing_new = [] if a.only_new else sorted(ref - set(new))
    same_bytes = same_arrays = 0
    worst = {}                      # key -> largest |difference| over files whose arrays differ
    for f in new:
        if f not in ref:
            continue
        p, q = os.path.join(a.new, f), os.path.join(a.ref, f)
        if filecmp.cmp(p, q, shallow=False):
            same_bytes += 1
            continue
        x, y = load(p), load(q)
        equal = sorted(x) == sorted(y)
        for k in x:
            if k not in y:
                worst[k] = float("inf"); equal = False; continue
            u, v = x[k], y[k]
            if u.shape != v.shape:
                worst[k] = float("inf"); equal = False
            elif not np.array_equal(u, v, equal_nan=u.dtype.kind in "fc"):
                equal = False
                if u.dtype.kind in "fciub":
                    d = np.abs(u.astype(np.float64) - v.astype(np.float64)); d = d[np.isfinite(d)]
                    worst[k] = max(worst.get(k, 0.0), float(d.max()) if d.size else 0.0)
                else:
                    worst[k] = float("inf")
        same_arrays += equal
    n = len(new) - len(missing_ref)
    print(f"{a.new}\n  against {a.ref}")
    print(f"  {n} files compared: {same_bytes} byte-identical, {same_arrays} with equal arrays but different bytes, "
          f"{n - same_bytes - same_arrays} different")
    for k, d in sorted(worst.items()):
        print(f"    key {k}: largest |difference| {d:.3g}")
    if missing_ref:
        print(f"  {len(missing_ref)} files only in the rebuilt folder, e.g. {missing_ref[0]}")
    if missing_new:
        print(f"  {len(missing_new)} files only in the reference, e.g. {missing_new[0]}")
    return 0 if same_bytes == n and not missing_ref and not missing_new else 1


if __name__ == "__main__":
    sys.exit(main())
