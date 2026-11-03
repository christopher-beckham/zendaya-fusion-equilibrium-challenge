"""Retrain one recorded run with the frozen train_mm.py, stopping at the epoch its checkpoint was taken from.

    python chris/mast/retrain.py RUN [--stop-epoch N] [--name NEW] [--dry-run] [-- EXTRA_TRAIN_MM_ARGS ...]

RUN is a folder name under chris/mast/run_records/. Every argument in its config.json is passed back to train_mm.py
unchanged, except:
  --name          NEW (default: RUN + "-rebuild"), so the original run folder is never touched
  --stop-epoch    N (default: the epoch in best_mast.json; required for runs without one). The learning-rate schedule
                  still spans --epochs, so epochs 1..N are trained as in the original run.
  --cache, --data-root
                  $FEC_CACHE and $FEC_DATA_ROOT when set (the recorded values are the defaults of both)
  --pictures      the repository's copy of the recorded file (same name, byte-identical), not the old worktree's
  --score-every 0 --synthS-every 0 --mast-checks-every 0
                  no scoring jobs during training. They ran in separate processes and never fed back into training;
                  without them the checkpoint at epoch N is last.pt instead of best_mast.pt / score_epNNN.pt.
Runs from chris/mast/multimachine/, as the original runs did from fec-multimachine/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "multimachine" / "my_experiments" / "nn" / "train_mm.py"
# Set by train_mm.py itself, not options: n_in from the data; the rest fixed after parsing (train_mm.py line 157), and
# every recorded run holds those same fixed values.
DERIVED = {"n_in", "decoder_norm", "lindec_erode", "lindec_act", "lindec_exact_ip"}
OFF = {"score_every": 0, "synthS_every": 0, "mast_checks_every": 0}


def command(cfg: dict, name: str, stop: int) -> list[str]:
    cfg = {**cfg, **OFF, "name": name, "stop_epoch": stop, "resume": False}
    for key, env in (("cache", "FEC_CACHE"), ("data_root", "FEC_DATA_ROOT")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    if cfg.get("pictures"):
        cfg["pictures"] = str(TRAIN.parent / Path(cfg["pictures"]).name)
    out = [sys.executable, "-u", str(TRAIN)]
    for k, v in cfg.items():
        if k in DERIVED or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else []
        elif isinstance(v, list):
            for x in v:
                out += [flag, str(x)]
        else:
            out += [flag, str(v)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--stop-epoch", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    rec = HERE / "run_records" / a.run
    cfg = json.loads((rec / "config.json").read_text())
    stop = a.stop_epoch
    if stop is None:
        best = rec / "best_mast.json"
        if not best.exists():
            raise SystemExit(f"{a.run} has no best_mast.json: pass --stop-epoch")
        stop = json.loads(best.read_text())["epoch"]
    cmd = command(cfg, a.name or f"{a.run}-rebuild", stop) + extra
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return 0
    return subprocess.run(cmd, cwd=HERE / "multimachine").returncode


if __name__ == "__main__":
    raise SystemExit(main())
