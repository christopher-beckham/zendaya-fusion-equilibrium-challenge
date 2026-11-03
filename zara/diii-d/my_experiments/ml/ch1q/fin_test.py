"""Public-test (or any test fold) q95 / betaN set for a recipe evaluated with fin.py.

  q95 = wq_E q95_E + wq_N mean(q95 of --q runs) + wq_P q95_phys(x_b),  betaN = wb_E betaN_E + wb_N mean(betaN of --b runs)
  E and the q profile come from the production psi ensemble (--ens preds_test dir, --qc its qcurve file); runs are production
  scalar networks (ml/ch1q/runs/<name>/preds_test, "q:<name>"; ml/ch1s runs "s:<name>").

    python fin_test.py --out runs/<name> --q q:prod_x,... --b ... --x ... --wq a,b,c --wb d,e
Writes <out>/preds_test/<stem>.npz (q95, betaN float32), <out>/components_test/, <out>/fin_test.json (+ sanity numbers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1s"))
from qcurve import q95_from_curve  # noqa: E402
from s_common import OUTC, TEST_CACHE, load_scalars, test_stems  # noqa: E402

CH1 = HERE.parent / "ch1"


def rdir(spec):
    kind, name = spec.split(":", 1)
    return (HERE if kind == "q" else HERE.parent / "ch1s") / "runs" / name / "preds_test"


def mean_col(specs, stems, col):
    return np.mean([np.concatenate([np.load(rdir(r) / f"{s}.npz")[col] for s in stems]) for r in specs], 0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-cache", type=Path, default=TEST_CACHE, help="c1/nn per-shot cache of the test fold (stems, times)")
    ap.add_argument("--ens", type=Path, default=CH1 / "runs" / "prod_v6_ens" / "preds_test")
    ap.add_argument("--qc", type=Path, default=OUTC / "qc_test_v6.npz")
    ap.add_argument("--q", required=True); ap.add_argument("--b", required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--wq", required=True); ap.add_argument("--wb", required=True)
    ap.add_argument("--q2", default=""); ap.add_argument("--b2", default="")
    ap.add_argument("--w2q", type=float, default=0.0); ap.add_argument("--w2b", type=float, default=0.0)
    ap.add_argument("--e-bn-range", default="")
    ap.add_argument("--q3", default=""); ap.add_argument("--b3", default="")
    ap.add_argument("--w3q", type=float, default=0.0); ap.add_argument("--w3b", type=float, default=0.0)
    a = ap.parse_args()
    stems = sorted(p.stem for p in a.test_cache.glob("*.npz") if p.stem not in ("basis", "meta"))
    E = load_scalars(a.ens, stems)
    if a.e_bn_range:                                          # variant-D rule for E (as fin.py)
        lo_, hi_ = [float(v) for v in a.e_bn_range.split(",")]
        E[:, 1] = np.clip(E[:, 1], lo_, hi_)
    qc = np.load(a.qc)
    assert [str(s) for s in qc["stems"]] == stems
    qn = mean_col(a.q.split(","), stems, "q95"); bn = mean_col(a.b.split(","), stems, "betaN")
    if a.q2:                                                  # second network group (fixed share), as fin.py
        qn = (1 - a.w2q) * qn + a.w2q * mean_col(a.q2.split(","), stems, "q95")
        bn = (1 - a.w2b) * bn + a.w2b * mean_col(a.b2.split(","), stems, "betaN")
    if a.q3:                                                  # third group (e.g. sequence networks), share of the result
        qn = (1 - a.w3q) * qn + a.w3q * mean_col(a.q3.split(","), stems, "q95")
        bn = (1 - a.w3b) * bn + a.w3b * mean_col(a.b3.split(","), stems, "betaN")
    xb = np.minimum(mean_col(a.x.split(","), stems, "xb"), 1.0)
    qp = q95_from_curve(qc["Q"], xb)
    nfb = int((~np.isfinite(qp)).sum())
    qp = np.where(np.isfinite(qp), qp, qn)
    wq = [float(v) for v in a.wq.split(",")]; wb = [float(v) for v in a.wb.split(",")]
    q = wq[0] * E[:, 0] + wq[1] * qn + wq[2] * qp
    b = wb[0] * E[:, 1] + wb[1] * bn
    assert np.isfinite(q).all() and np.isfinite(b).all(), "non-finite test predictions"
    tt = [np.load(a.test_cache / f"{s}.npz")["t"] for s in stems]
    T = [len(x) for x in tt]
    off = np.r_[0, np.cumsum(T)]
    assert off[-1] == len(q)
    (a.out / "preds_test").mkdir(parents=True, exist_ok=True); (a.out / "components_test").mkdir(exist_ok=True)
    for i, s in enumerate(stems):
        sl = slice(off[i], off[i + 1])
        np.savez(a.out / "preds_test" / f"{s}.npz", q95=q[sl].astype(np.float32), betaN=b[sl].astype(np.float32),
                 t=np.asarray(tt[i], np.float32))
        np.savez(a.out / "components_test" / f"{s}.npz", q95_E=E[sl, 0].astype(np.float32), q95_net=qn[sl].astype(np.float32),
                 q95_phys=qp[sl].astype(np.float32), xb=xb[sl].astype(np.float32), betaN_E=E[sl, 1].astype(np.float32),
                 betaN_net=bn[sl].astype(np.float32))
    def agree(x, y):
        return float(1 - np.sum((x - y) ** 2) / np.sum((y - y.mean()) ** 2))
    info = {"q": a.q, "b": a.b, "x": a.x, "q2": a.q2, "b2": a.b2, "w2q": a.w2q, "w2b": a.w2b, "e_bn_range": a.e_bn_range, "q3": a.q3, "b3": a.b3, "w3q": a.w3q, "w3b": a.w3b, "wq": wq, "wb": wb, "ens": str(a.ens), "qc": str(a.qc), "shots": len(stems),
            "frames": int(len(q)), "phys_fallback_frames": nfb, "limited_pred_frac": float(np.mean(xb < 0.999)),
            "agree_r2_q95_vs_E": agree(q, E[:, 0]), "agree_r2_betaN_vs_E": agree(b, E[:, 1]),
            "q95_min": float(q.min()), "q95_max": float(q.max()), "betaN_min": float(b.min()), "betaN_max": float(b.max()),
            "betaN_lt0": int((b < 0).sum())}
    (a.out / "fin_test.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
