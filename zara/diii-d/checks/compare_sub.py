"""Compare two submission .npz files key by key: byte-identical arrays, largest |difference| per quantity, pooled psi R2 of b
against a.    python checks/compare_sub.py <a.npz> <b.npz>"""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
print(f"keys: {len(ka)} vs {len(kb)}, common {len(ka & kb)}")
same = 0; mx = {}; sse = sst = s1 = n = 0.0
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape == y.shape and np.array_equal(x, y):
        same += 1
    q = k.rsplit("_", 1)[1]
    if x.shape == y.shape:
        mx[q] = max(mx.get(q, 0.0), float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0)
    else:
        mx[q] = float("inf")
    if q == "psirz" and x.shape == y.shape:
        x64, y64 = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
        sse += ((x64 - y64) ** 2).sum(); s1 += x64.sum(); sst += (x64 ** 2).sum(); n += x64.size
r2 = 1 - sse / (sst - s1 * s1 / n) if n else float("nan")
print(f"{same} of {len(ka & kb)} arrays byte-identical; largest |difference| {mx}; psi R2 {r2:.10f}")
