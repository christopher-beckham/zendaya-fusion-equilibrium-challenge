"""Run a script of any pegs-family code folder (upegs, m2/upegs_fz, ch2/g4/fz) with an extra MAST split registered in that
folder's machine.MAST_CACHE. The registration is in memory, in this process only; no shared file is edited.

    python split_run_any.py <code dir> <split> <source cache dir> [--alias-test] <script.py> [script args...]
"""
import runpy
import sys
from pathlib import Path

code, split, src, rest = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
sys.path.insert(0, str(Path(code).resolve()))
import machine  # noqa: E402

machine.MAST_CACHE[split] = src
if rest and rest[0] == "--alias-test":
    machine.MAST_CACHE["test"] = src
    rest = rest[1:]
script = str(Path(rest[0]).resolve())
sys.argv = [script] + rest[1:]
sys.path.insert(0, str(Path(script).parent))
runpy.run_path(script, run_name="__main__")
