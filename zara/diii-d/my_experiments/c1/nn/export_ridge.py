"""Export a pickled C1Model's parameters to a plain .npz (portable across envs / numpy versions).

    python export_ridge.py ../models/holdout10_seed0_c1_field_shaf_eddy.pkl $FEQ_CACHE/nn/ridge_holdout.npz
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model import C1Model, Z_CLIP, ALPHAS  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
m = C1Model.load(src)
np.savez(dst, theta=np.asarray(m.theta), mu=np.asarray(m.mu), sd=np.asarray(m.sd),
         W=np.asarray(m.W), b=np.asarray(m.b), Ws=np.asarray(m.Ws), bs=np.asarray(m.bs),
         alpha_psi=m.alpha_psi, alpha_scal=m.alpha_scal, z_clip=Z_CLIP,
         names=np.asarray(list(map(str, m.names))), physics=np.asarray(list(m.physics_opts)))
print(f"{src.name}: theta={np.round(m.theta, 4)} alpha_psi={m.alpha_psi:g} alpha_scal={m.alpha_scal:g} "
      f"W {m.W.shape} physics={m.physics_opts} -> {dst}")
