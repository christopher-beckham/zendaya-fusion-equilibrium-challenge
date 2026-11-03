"""build_cache_redirect.py OUT_ROOT FZ_DIR <build_cache.py args...>
Runs FZ_DIR/build_cache.py unchanged, except that every cache it writes goes to OUT_ROOT/<the folder name it would use>
instead of Zara's cache tree (cache_dir() is wrapped). Prints the wall time at the end."""
import os, sys, time
out, fz = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.argv = [os.path.join(fz, "build_cache.py")] + sys.argv[3:]
sys.path.insert(0, fz); os.chdir(fz)
import build_cache as bc
from pathlib import Path
_cd = bc.cache_dir
bc.cache_dir = lambda *a, **k: Path(out) / _cd(*a, **k).name
t0 = time.time(); rc = bc.main()
print(f"[redirect] wrote under {out}; wall {time.time() - t0:.0f} s; device {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}", flush=True)
sys.exit(rc)
