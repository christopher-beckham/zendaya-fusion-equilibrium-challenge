#!/usr/bin/env bash
# Build an ensemble candidate: mean of several checkpoints' raw MAST predictions (psi, q95, betaN heads), then the same
# post-processing and Challenge 1 DIII-D file as build_candidate.sh.
# usage: [NO_UPLOAD=1] build_ensemble.sh TAG CKPT1 CKPT2 [...]   (set FEC_MAST_MASK as for the members)
# GPUS="2 5 6 7" picks the GPUs members run on (default 0 1 2 ...); members share them round-robin.
set -u
TAG=$1; shift
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter
MM=/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn
S=$REPO/submissions; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
log() { echo "[$(date +%T)] $*"; }
T=$S/tmp-$TAG; mkdir -p $T
i=0; read -ra G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
for CK in "$@"; do
  M=$T-m$i; mkdir -p $M
  (cd $MM && CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $PY -u predict.py $CK --out $M --device cuda --check-order 0 > $M/predict.log 2>&1) &
  echo "$CK" > $M/ckpt.txt; i=$((i+1))
done
# predict.py goes on to the DIII-D files; stop each member once its MAST file is written
for M in $T-m*; do
  until grep -q "Validating diii_d" $M/predict.log 2>/dev/null || grep -q Traceback $M/predict.log 2>/dev/null; do sleep 20; done
  grep -q Traceback $M/predict.log && { log "PREDICT FAILED in $M"; tail -5 $M/predict.log; exit 1; }
  pkill -f "predict.py $(cat $M/ckpt.txt | sed 's/[][\\.*^$+?(){}|]/\\&/g') --out $M" 2>/dev/null
done
log "members written"
T=$T $PY - <<'PYEOF'
import numpy as np, os, glob
T = os.environ["T"]; ms = sorted(glob.glob(T + "-m*"))
zs = [np.load(f"{m}/mast_public_test.npz") for m in ms]
assert all(z.files == zs[0].files and all(z[k].shape == zs[0][k].shape for k in z.files) for z in zs)
np.savez(f"{T}/mast_public_test.npz", **{k: np.mean([z[k] for z in zs], axis=0).astype(np.float32) for k in zs[0].files})
print(f"averaged {len(ms)} members over {len(zs[0].files)} arrays", flush=True)
PYEOF
ln -f $S/c1-shape-ep270-fp32/diii_d_public_test.npz $T/diii_d_public_test.npz   # postproc_qb expects one; replaced below anyway
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
bash $REPO/my_experiments/nn/upload_candidate.sh $TAG
