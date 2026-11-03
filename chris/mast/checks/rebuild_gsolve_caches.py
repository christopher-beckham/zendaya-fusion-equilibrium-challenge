"""Rebuild gsolve_<set>_limiter from the downloaded gen4_layout with Chris's synth_cache_seq.py and synth_shape_targets.py
(unchanged; CACHE pointed at a scratch root), then compare with the /fsx original."""
import hashlib, json, os, subprocess, sys
from pathlib import Path

NN = "/fsx/chris.beckham/github/fec-final/chris/mast/multimachine/my_experiments/nn"
DATA = Path("/fsx/chris.beckham/fec_data/gsolve_raw")
ORIG = Path("/fsx/chris.beckham/fusion_cache/synth")
ROOT = Path("/fsx/chris.beckham/fec_upload/rebuild_check")
W = sys.argv[1] if len(sys.argv) > 1 else "16"

(ROOT / "features_v2").mkdir(parents=True, exist_ok=True)
pic = ROOT / "features_v2" / "pictures_mast_sol0.5.npz"
if not pic.exists():
    pic.symlink_to("/fsx/chris.beckham/fusion_cache/features_v2/pictures_mast_sol0.5.npz")
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "FEC_SYNTH_HALF": "4", "FEC_MAST_MASK": "limiter"}

def run(module, argv):
    code = (f"import sys; from pathlib import Path; sys.path.insert(0, {NN!r}); import {module} as m; "
            f"m.CACHE = Path({str(ROOT)!r}); sys.argv = {[module + '.py', *argv]!r}; raise SystemExit(m.main())")
    subprocess.run([sys.executable, "-u", "-c", code], env=env, check=True, cwd=NN)

def sha(p):
    h = hashlib.sha256(); f = open(p, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

ok = True
for s in ("cond1000", "cond3_1000"):
    tag = f"gsolve_{s}"
    src = DATA / s / "gen4_layout"
    run("synth_cache_seq", ["--src", str(src / "train"), "--heldout-src", str(src / "heldout"),
                            "--tag", f"{tag}_limiter", "--workers", W])
    run("synth_shape_targets", [tag, "--workers", W])
    new, old = ROOT / "synth" / f"{tag}_limiter", ORIG / f"{tag}_limiter"
    for f in ("arrays.npz", "shape_targets.npz"):
        same = sha(new / f) == sha(old / f); ok &= same
        print(f"{tag}/{f}: {'MATCH' if same else 'DIFFERENT'}", flush=True)
    mn, mo = json.loads((new / "manifest.json").read_text()), json.loads((old / "manifest.json").read_text())
    paths = ("source", "heldout_source")
    diff = sorted(k for k in set(mn) | set(mo) if k not in paths and mn.get(k) != mo.get(k))
    ok &= not diff
    print(f"{tag}/manifest.json: fields other than {paths}: {'MATCH' if not diff else 'DIFFERENT ' + str(diff)}", flush=True)
    for k in paths:
        print(f"   {k}: {mo[k]} -> {mn[k]}", flush=True)
print("ALL MATCH" if ok else "SOME DIFFERENCES", flush=True)
