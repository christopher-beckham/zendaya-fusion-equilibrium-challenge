"""Add a constant uniform vertical field (mT) per frame to a select_b.py npz: <stem>__bv arrays next to <stem>__sol.
    python diag/add_bv.py <sel.npz> <bv mT>"""
import sys

import numpy as np

f, bv = sys.argv[1], float(sys.argv[2])
d = dict(np.load(f))
for k in [k for k in d if k.endswith("__sol")]:
    d[k[:-5] + "__bv"] = np.full_like(d[k], bv)
np.savez(f, **d)
print(f"{f}: __bv = {bv} mT on {sum(k.endswith('__sol') for k in d)} stems")
