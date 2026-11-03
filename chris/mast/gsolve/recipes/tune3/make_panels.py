"""Fixed shot panels: tuning panel (critic-train / critic-val) and a final test panel, disjoint by shot."""
import json
import numpy as np
import pandas as pd
from tune_lib import HERE, REAL_DIR, real_table

T = real_table().reset_index(drop=True)
ok = (abs(T.lag_prev - 20) < .6) & (abs(T.lag_next - 20) < .6) & (T["m:thomson_missing"] == 0) & (T.ip_kA.abs() >= 100)
cnt = T[ok].groupby("shot").size()
elig = sorted(cnt[cnt >= 12].index)
rng = np.random.default_rng(20261015)
pick = rng.permutation(elig)
panel = dict(train=sorted(pick[:72].tolist()), val=sorted(pick[72:120].tolist()), test=sorted(pick[120:240].tolist()),
             note="shots with >= 12 usable real frames (Thomson, 20 ms history); disjoint; seed 20261015")
(HERE / "panels.json").write_text(json.dumps(panel, indent=1))
print(len(elig), {k: len(v) for k, v in panel.items() if k != "note"})
