#!/bin/bash
# Rebuild Zara's two within-budget team blends with her blend_build2.py and weights (read-only from her folder):
# once from the original model files (must equal her blend array for array) and once with Chris's members replaced by
# the files predicted from the retrained checkpoints (compared with her blend).
set -u
until grep -q "all done" /fsx/chris.beckham/fec_upload/step2/predict_all_rebuilt.out 2>/dev/null; do sleep 60; done
P=/fsx/chris.beckham/miniconda3/envs/fusion-eq/bin/python
ZT=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/my_experiments/ch2/g4; ZS=/fsx/zara.ercan/fusion-equilibrium-challenge-starter/submission
OUT=/fsx/chris.beckham/fec_upload/step2/blends; mkdir -p $OUT; S2=/fsx/chris.beckham/fec_upload/step2
W() { $P -c "import json,sys;d=json.load(open('$1'))[sys.argv[1]];t=sum(d.values());print(','.join(f'{k}={v/t:.4f}' for k,v in d.items()))" $2; }
for spec in "s1fc wS1_FC.json" "s2v7c wS2_V7C.json"; do
  set -- $spec; name=$1; wj=$ZT/final_push/$2
  for src in pred_public pred_rebuilt; do
    M=$S2/$src
    $P -I $ZT/team_blend/blend_build2.py --out $OUT/${name}_$src.npz --psi $(W $wj maps) --q95 $(W $wj q95) --betaN $(W $wj betaN) \
      r1=$ZS/nn_prod_v13_ch1_g4r1/mast_public_test.npz u4pp=/fsx/zara.ercan/cache/ch2/g4/old_subs/u4pp/mast_public_test.npz \
      u4all=/fsx/zara.ercan/cache/ch2/g4/old_subs/u4all/mast_public_test.npz \
      Z15=$M/Z15/mast_public_test.npz@-1 Z18=$M/Z18/mast_public_test.npz@-1 Z17=$M/Z17/mast_public_test.npz@-1 \
      Z=$M/Z/mast_public_test.npz@-1 Z2=$M/Z2/mast_public_test.npz@-1 FC=$M/FC/mast_public_test.npz@-1 V7C=$M/V7C/mast_public_test.npz@-1 \
      > $OUT/${name}_$src.log 2>&1 || { echo "$name $src: blend failed"; tail -3 $OUT/${name}_$src.log; continue; }
  done
  $P - <<PY
import numpy as np
a, b = np.load("/fsx/zara.ercan/cache/ch2/g4/blend/$name.npz"), np.load("$OUT/${name}_pred_public.npz")
same = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
print("$name from the original model files: " + ("IDENTICAL arrays to Zara's blend" if same else "DIFFERENT from Zara's blend"))
PY
  $P /fsx/chris.beckham/github/fec-final/chris/mast/checks/compare_predictions.py /fsx/zara.ercan/cache/ch2/g4/blend/$name.npz \
    $OUT/${name}_pred_rebuilt.npz --name "$name with retrained models vs Zara's blend"
done
echo "all done"
