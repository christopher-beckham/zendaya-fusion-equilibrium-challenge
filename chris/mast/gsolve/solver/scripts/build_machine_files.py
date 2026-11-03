#!/usr/bin/env python3
"""Write gsolve/machines/<name>_coils.csv from the challenge's coil geometry (one-off, needs pyarrow).

Source: the coil columns of the Fusion Equilibrium Challenge parquet files (coil_input_column, coil_R,
coil_Z, coil_width, coil_height, coil_angle1, coil_angle2). The geometry is identical in every shot file
of a machine, so the first file of each folder is read:
    MAST:   <data>/mast_public_test/*.parquet   812 elements (656 solenoid, 156 in the P2-P6 coils)
    DIII-D: <data>/diii_d_train/*.parquet        19 elements (F1A-F9B shaping coils, E coil)

`turns` is the current in the element per unit current of its input column:
    MAST P coils: the column is the coil's total current (kA x turns), shared evenly by its n elements -> 1/n
    MAST solenoid: every element carries the column current -> 1 (the x0.5 factor is in mast.json)
    DIII-D: each column feeds its one rectangle -> 1

    python scripts/build_machine_files.py /path/to/fusion_equilibrium_challenge/data
"""
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / "gsolve" / "machines"
COLS = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2"]

for name, folder in (("mast", "mast_public_test"), ("diii-d", "diii_d_train")):
    f = sorted((Path(sys.argv[1]) / folder).glob("*.parquet"))[0]
    r = pq.read_table(f, columns=COLS).to_pylist()[0]
    cols = [c.replace("magnetics_", "").replace("_current", "") for c in r["coil_input_column"]]
    n = Counter(cols)
    lines = ["column,R,Z,width,height,angle1,angle2,turns"]
    for i, c in enumerate(cols):
        turns = 1.0 if (name != "mast" or c == "sol") else 1.0 / n[c]
        vals = [r[k][i] for k in COLS[1:]] + [turns]
        lines.append(",".join([c] + [f"{v:.7g}" for v in vals]))
    (OUT / f"{name}_coils.csv").write_text("\n".join(lines) + "\n")
    print(name, f.name, len(cols), "elements")
