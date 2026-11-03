"""Run a training / prediction script with the CUDA memory of this process capped (default 80 GB = one A100 80GB).

    CUDA_MEM_CAP_GB=80 python capmem.py <script.py> <script args ...>

The cap uses torch.cuda.set_per_process_memory_fraction, so an allocation beyond it raises an out-of-memory error exactly as
on a smaller card. At exit it prints the GPU name, the cap, the peak allocated / reserved CUDA memory and the wall-clock time.
"""
import atexit
import os
import runpy
import sys
import time
from pathlib import Path

import torch

T0 = time.time()
CAP = float(os.environ.get("CUDA_MEM_CAP_GB", "80"))
REPO = str(Path(__file__).resolve().parents[3]) + "/"   # here, not at exit: Python deletes __main__.__file__ when the script ends
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, CAP / total))


@atexit.register
def _report():
    mods = sorted({getattr(m, "__file__", "") or "" for m in list(sys.modules.values())} - {""})
    print("[capmem] local modules: " + " ".join(f[len(REPO):] for f in mods if f.startswith(REPO)), flush=True)
    if torch.cuda.is_available():
        print(f"[capmem] {torch.cuda.get_device_name(0)}: cap {CAP:.0f} GB, peak allocated "
              f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB, peak reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB, "
              f"wall {time.time() - T0:.0f} s", flush=True)


script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
