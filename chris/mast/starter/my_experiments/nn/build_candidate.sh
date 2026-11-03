#!/usr/bin/env bash
# Build a candidate: MAST maps from CK, physics q95 (no multiplier), physics betaN x1.2, Challenge 1 DIII-D file.
# Uploaded and hash-checked; nothing is submitted.  usage: build_candidate.sh TAG CKPT [GPU]
# NO_UPLOAD=1 stops after the local build (upload later with upload_candidate.sh TAG).
set -u
TAG=$1
CK=$2
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter
MM=/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn
S=$REPO/submissions; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
log() { echo "[$(date +%T)] $*"; }
T=$S/tmp-$TAG; mkdir -p $T
(cd $MM && CUDA_VISIBLE_DEVICES=${3:-3} $PY -u predict.py $CK --out $T --device cuda --check-order 0 > $T/predict.log 2>&1) &
PID=$!
until grep -q "Validating diii_d" $T/predict.log 2>/dev/null || grep -q Traceback $T/predict.log 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 20; done
pkill -f "predict.py $(printf %s "$CK --out $T" | sed 's/[][\\.*^$+?(){}|]/\\&/g')" 2>/dev/null; kill $PID 2>/dev/null   # escape regex characters (run names contain +)
grep -q Traceback $T/predict.log && { log "PREDICT FAILED"; tail -5 $T/predict.log; exit 1; }
[ -f $T/mast_public_test.npz ] || { log "NO MAST FILE"; exit 1; }
log "MAST written"
(cd $REPO/my_experiments/nn && CUDA_VISIBLE_DEVICES="" $PY -u postproc_qb.py $T $T-physqb --workers 6 > $T-physqb.log 2>&1) || { log "POSTPROC FAILED"; exit 1; }
log "physics q95/betaN done"
TAG=$TAG $PY - <<'PYEOF'
import numpy as np, os, pyarrow.parquet as pq
from pathlib import Path
tag = os.environ["TAG"]; S = Path("/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions")
files = sorted(Path("/fsx/hackathon_offsite/fusion_equilibrium_challenge/data/mast_public_test").glob("*.parquet"))
T = [len(pq.read_table(f, columns=["efit_times"]).column(0)[0]) for f in files]
z = np.load(S / f"tmp-{tag}-physqb" / "mast_public_test.npz"); out = S / tag; out.mkdir(exist_ok=True)
arr = {k: np.asarray(z[k] * (1.2 if k.endswith("_betaN") else 1.0), np.float32) for k in z.files}
bad = [i for i, t in enumerate(T) if arr[f"shot_{i:04d}_psirz"].shape != (t, 65, 65)
       or not all(np.isfinite(arr[f"shot_{i:04d}_{q}"]).all() for q in ("psirz", "q95", "betaN"))]
np.savez(out / "mast_public_test.npz", **arr)
d = out / "diii_d_public_test.npz"
if d.exists(): os.remove(d)
os.link(S / "c1-shape-ep270-fp32" / "diii_d_public_test.npz", d)
for f in ("mast_public_test", "diii_d_public_test"):
    zz = np.load(out / f"{f}.npz"); print(f"{tag} {f}: dtypes {sorted({str(zz[k].dtype) for k in zz.files})}")
q = np.concatenate([arr[f"shot_{i:04d}_q95"] for i in range(len(files))])
print(f"{tag}: {len(files)} shots, problems {len(bad)}; q95 10/50/90 {np.percentile(q,[10,50,90]).round(2)}", flush=True)
PYEOF
[ "${NO_UPLOAD:-0}" = 1 ] && { log "built locally, not uploaded"; exit 0; }
cd $REPO && export HF_READ_TOKEN="$(tr -d '[:space:]' < ft-read-token.txt)"
$PY push_predictions.py --repo beckhamc/fusion-eq-predictions --dir submissions/$TAG --out submissions/$TAG/$TAG.zip 2>&1 \
  | grep -E "Upload this|revision|ERROR|Error|limit" | sed -E 's/hf_[A-Za-z0-9]+/<token>/g'
TAG=$TAG $PY - <<'PYEOF' 2>&1 | sed -E 's/hf_[A-Za-z0-9]+/<token>/g'
import os, json, zipfile, hashlib
from huggingface_hub import HfApi
api = HfApi(); tok = os.environ["HF_READ_TOKEN"]; d = os.environ["TAG"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
m = json.loads(zipfile.ZipFile(f"submissions/{d}/{d}.zip").read("manifest.json"))
fs = {f.path: f for f in api.list_repo_tree(m["repo_id"], revision=m["revision"], repo_type="dataset", token=tok, expand=True)}
print(d, m["revision"][:8], *[n + (" MATCH" if n in fs and fs[n].lfs.sha256 == sha(f"submissions/{d}/{n}") else " MISMATCH")
                              for n in ("diii_d_public_test.npz", "mast_public_test.npz")], flush=True)
PYEOF
log "all done"
