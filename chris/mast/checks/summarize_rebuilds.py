"""Summarise the rebuilds of the 30 training runs: synthetic-MAST S of original and rebuild, wall clock, peak GPU memory,
and training time per model and per team blend.

    python chris/mast/checks/summarize_rebuilds.py JOBS_DIR > chris/mast/checks/rebuilds/SUMMARY.md

JOBS_DIR holds rebuild_<run>.log (with /usr/bin/time -v output), gpumem_<run>.csv and synthS_<run>-rebuild.json, plus
synthS_<run>-orig.json for the runs whose original checkpoint was never scored during training.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MODELS = {k: v for k, v in json.loads((HERE / "models.json").read_text()).items() if not k.startswith("_")}
BLENDS = {
    "One 24 h budget (B16+FC)": ["Z15", "Z18", "Z", "Z2", "FC"],
    "24 h per challenge (B16+V7C)": ["Z15", "Z18", "Z17", "Z", "Z2", "V7C"],
    "Maximum": ["V7C", "FC", "Z15", "Z18", "Z3", "Z9qb", "Z", "Z6b", "Z7", "Z8", "FA", "Z2"],
}


def hms(s: str) -> float:
    p = [float(x) for x in s.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def main() -> int:
    jobs = Path(sys.argv[1])
    runs = sorted({r for m in MODELS.values() for r, _ in m["members"]})
    rows = {}
    for r in runs:
        log = (jobs / f"rebuild_{r}.log").read_text()
        wall = hms(re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", log).group(1))
        mem = max(int(re.sub(r"\D", "", line.split(",")[1]) or 0)
                  for line in (jobs / f"gpumem_{r}.csv").read_text().splitlines() if "," in line)
        best = HERE / "run_records" / r / "best_mast.json"
        if best.exists() and json.loads(best.read_text()).get("metric") == "g4_20000":
            b = json.loads(best.read_text())
            orig, stop, ck = b["synthS"], b["epoch"], "best_mast.pt"
        elif best.exists():           # selected on another set: the original checkpoint scored on g4_20000 here
            stop, ck = json.loads(best.read_text())["epoch"], "best_mast.pt"
            orig = json.loads((jobs / f"synthS_{r}-orig.json").read_text())["S"]["g4_20000"]
        else:
            orig = json.loads((jobs / f"synthS_{r}-orig.json").read_text())["S"]["g4_20000"]
            stop, ck = 30, "score_ep030.pt"
        new = json.loads((jobs / f"synthS_{r}-rebuild.json").read_text())["S"]["g4_20000"]
        rows[r] = dict(stop=stop, ck=ck, wall=wall, mem=mem / 1024, orig=orig, new=new)
    print("# Rebuilds of Chris's 30 training runs\n")
    print("Each run retrained from scratch with `retrain.py` (frozen `train_mm.py`, stopped at the checkpoint's epoch, no "
          "scoring jobs), one RTX PRO 6000 per run. Synthetic-MAST S: `ensemble_synth.py --sets g4_20000`, the scorer the "
          "original runs used. Originals: the g4_20000 score recorded in `best_mast.json`; for the four runs without one "
          "(`score_ep030.pt`) and the one run selected on another set (W4-v4only, `g4_v4`), the original checkpoint scored "
          "the same way.\n")
    print("| Run | Stop epoch | Wall clock | Peak GPU memory | Synthetic-MAST S original | Rebuild | Difference |")
    print("|---|---|---|---|---|---|---|")
    for r, d in rows.items():
        print(f"| `{r}` | {d['stop']} | {d['wall'] / 60:.1f} min | {d['mem']:.1f} GB | {d['orig']:.6f} | {d['new']:.6f} | "
              f"{d['new'] - d['orig']:+.6f} |")
    diffs = [d["new"] - d["orig"] for d in rows.values()]
    print(f"\nAcross the 30 runs: mean difference {sum(diffs) / len(diffs):+.6f}, largest absolute difference "
          f"{max(abs(x) for x in diffs):.6f}. The organisers allow 0.01 S between a retrained entry and the submitted "
          f"score.\n")
    print("## Training time per model and per blend (wall clock of the training commands)\n")
    print("| Model | Runs | Training time |")
    print("|---|---|---|")
    for m, spec in MODELS.items():
        rs = sorted({r for r, _ in spec["members"]} | ({r for r, _ in MODELS[spec["q95_from"]]["members"]}
                                                        if "q95_from" in spec else set()))
        print(f"| {m} | {len(rs)} | {sum(rows[r]['wall'] for r in rs) / 3600:.2f} h |")
    print("\nShared runs count once per blend (Z3 shares two runs with Z and Z2, Z9qb holds Z7's four, Z6b needs Z's run "
          "for its q95). Chris's part only; Zara's members are not included.\n")
    print("| Blend | Chris's runs | Chris's training time |")
    print("|---|---|---|")
    for b, ms in BLENDS.items():
        rs = set()
        for m in ms:
            rs |= {r for r, _ in MODELS[m]["members"]}
            if "q95_from" in MODELS[m]:
                rs |= {r for r, _ in MODELS[MODELS[m]["q95_from"]]["members"]}
        print(f"| {b} | {len(rs)} | {sum(rows[r]['wall'] for r in rs) / 3600:.2f} h |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
