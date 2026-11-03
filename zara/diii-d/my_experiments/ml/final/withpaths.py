"""Run a script's main() with module-level path constants replaced, for the cache builders that read raw parquet files from
the repo's challenge_data/data/<split> (a link to the shared read-only data): new splits (the Final-phase private test, or a
dry-run copy of the public test) live under a private data root instead.

    python withpaths.py <script.py> NAME=PATH [NAME=PATH ...] -- <script arguments>
e.g. python withpaths.py ../../c1/nn/preprocess_px.py DATA=/fsx/zara.ercan/feq_data/challenge_data/data -- --split diii_d_pubnew

The script is imported as a normal module (its directory on sys.path), NAME is set to Path(PATH), sys.argv is replaced and
main() is called. Worker processes of the builders receive their file / output paths as explicit arguments, so they need no
patch (spawn workers re-import the unpatched module; fork workers inherit the patched one).
"""
import importlib
import sys
from pathlib import Path


def run():
    script = Path(sys.argv[1]).resolve()
    i = sys.argv.index("--")
    sets = [kv.split("=", 1) for kv in sys.argv[2:i]]
    sys.path.insert(0, str(script.parent))
    mod = importlib.import_module(script.stem)
    for k, v in sets:
        assert hasattr(mod, k), f"{script.name} has no module-level {k}"
        setattr(mod, k, Path(v))
        print(f"[withpaths] {script.name}: {k} = {v}", flush=True)
    sys.argv = [str(script)] + sys.argv[i + 1:]
    try:
        return mod.main()
    finally:
        repo = str(Path(__file__).resolve().parents[3]) + "/"
        mods = sorted({getattr(m, "__file__", "") or "" for m in list(sys.modules.values())} - {""})
        print("[withpaths] local modules: " + " ".join(f[len(repo):] for f in mods if f.startswith(repo)), flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
