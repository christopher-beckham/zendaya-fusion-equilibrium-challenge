#!/usr/bin/env bash
# Upload an already-built submission folder and verify its hashes.  usage: upload_candidate.sh TAG
set -u
TAG=$1
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter
PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
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
echo "[$(date +%T)] all done"
