#!/usr/bin/env bash
# Z9: 7-member average of the Z7 recipe (V2, W1-W3, W4-W6 seeds 4-6), then the centre-column q95 rule (×0.932,
# q95_study/H_v4) and V2 βN (q95_study/F_betaN, REPORT_V4_DEEPDIVE.md). Builds Z9q (q95 rule only) and Z9qb (both).
set -u
REPO=/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter; NN=$REPO/my_experiments/nn; S=$REPO/submissions
R=/fsx/chris.beckham/github/fec-multimachine/my_experiments/nn/runs; PY=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
for n in scr-W4-20k5kv4-seed4 scr-W5-20k5kv4-seed5 scr-W6-20k5kv4-seed6; do until [ -f $R/$n/synthS_ep020.json ]; do sleep 60; done; done; sleep 60
for n in scr-W4-20k5kv4-seed4 scr-W5-20k5kv4-seed5 scr-W6-20k5kv4-seed6; do echo "$n best $($PY -c "import json;d=json.load(open('$R/$n/best_mast.json'));print(d['epoch'],round(d['synthS'],4))")"; done
B=mastZ9-mm-7xV2recipe-limiter-c1
cd $NN && FEC_MAST_MASK=limiter GPUS="0 1 2 3" NO_UPLOAD=1 bash build_ensemble.sh $B $R/scr-V2-20k5kv4/best_mast.pt \
  $R/scr-W1-20k5kv4-seed1/best_mast.pt $R/scr-W2-20k5kv4-seed2/best_mast.pt $R/scr-W3-20k5kv4-seed3/best_mast.pt \
  $R/scr-W4-20k5kv4-seed4/best_mast.pt $R/scr-W5-20k5kv4-seed5/best_mast.pt $R/scr-W6-20k5kv4-seed6/best_mast.pt > $S/build_Z9.log 2>&1
tail -n 2 $S/build_Z9.log
cd $NN/q95_study/H_v4 && CUDA_VISIBLE_DEVICES= FEC_MAST_MASK=limiter $PY apply_cc.py $S/$B $S/mastZ9q-mm-7xV2recipe-q95cc0932-limiter-c1 --factor 0.932 --workers 12 2>&1 | tail -2
cd $NN/q95_study/F_betaN && CUDA_VISIBLE_DEVICES= FEC_MAST_MASK=limiter $PY apply_bn_variant.py $S/mastZ9q-mm-7xV2recipe-q95cc0932-limiter-c1 $S/mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1 --variant V2 2>&1 | tail -3
cd $S && $PY -c "
import numpy as np
for t in ['$B','mastZ9q-mm-7xV2recipe-q95cc0932-limiter-c1','mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1']:
    z=np.load(t+'/mast_public_test.npz'); q=np.concatenate([z[k] for k in z.files if k.endswith('_q95')]); b=np.concatenate([z[k] for k in z.files if k.endswith('_betaN')])
    print(t, 'q95>12.44 %.2f%%'%(100*(q>12.44).mean()), 'betaN 10/50/90', np.percentile(b,[10,50,90]).round(2), sorted({str(z[k].dtype) for k in z.files}), len([k for k in z.files if k.endswith('_psirz')]), 'nonfinite', sum(int((~np.isfinite(z[k])).sum()) for k in z.files))
"
for t in mastZ9q-mm-7xV2recipe-q95cc0932-limiter-c1 mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1; do
  cd $NN && HF_TOKEN_PATH=$HOME/.cache/huggingface/token bash upload_candidate.sh $t 2>&1 | grep -E "MATCH|MISMATCH|ERROR|Error"; done
echo done
