"""Train the mirror Ridge model (same recipe as control_vs_mirror_val.py) and run it on the
public leaderboard test inputs, writing submission-format .npz files.

Model: 21 coils (interpolated to efit_times, Ip time-axis fixed), PF circuit (ECOILA, F1A-F9B, Ip)
multiplied by s = sign(flat-top median Ip) -> StandardScaler -> RidgeCV -> 20 PCA comps of psi.
q95 / betaN: RidgeCV on the same scaled inputs, training labels clipped to [p0.1, p99.9].
Trained on the same 1,000-shot subsample (seed 0) of the baseline's 70% train split.
MAST: zeros (this model only knows DIII-D coils; Challenge 1 scores DIII-D only).

Keys are shot_{i:04d}_* in sorted-filename order, which must equal the Hub stream order
(checked with --check-order, and by validate_submission.py's per-shot T check).

Usage: uv run python my_experiments/infer_mirror_test.py [--out submission] [--check-order 20]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_stats import SIGS, OUT, REPO, process

DATA = REPO / "hf_dataset/data"
MODEL_PATH = OUT / "mirror_model.pkl"


def features(path, split):
    """(T, 21) coils on efit_times + sign of flat-top Ip, from the stats cache or computed fresh."""
    c = OUT / "cache" / split / f"{path.stem}.pkl"
    r = pickle.load(open(c, "rb")) if c.exists() else None
    if r is None or "error" in r["meta"]:
        r = process(path, split)
    s = np.sign(r["meta"]["ip_flattop_median"])
    return r, (s if s in (-1.0, 1.0) else 1.0)


def mirror(X, s):
    Xm = X.copy()
    Xm[:, [j for j, n in enumerate(SIGS) if n != "bcoil"]] *= s
    return Xm


def load_psi(path):
    col = pq.read_table(path, columns=["efit_psirz"]).column(0).combine_chunks()[0].values
    while hasattr(col, "flatten"):
        col = col.flatten()
    return col.to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, 65 * 65)


def check_order(config, n, data_dir=None):
    """First `n` Hub public-test rows must match local files sorted by name (efit_times)."""
    from datasets import load_dataset
    root = Path(data_dir) if data_dir is not None else DATA
    local = sorted((root / config).glob("*.parquet"))
    ds = load_dataset("Sophelio/fusion-equilibrium-challenge", config, split="public_test", streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        et_hub = np.asarray(row["efit_times"], dtype=np.float64)
        et_loc = pq.read_table(local[i], columns=["efit_times"]).column(0).combine_chunks()[0].values.to_numpy(zero_copy_only=False)
        if len(et_hub) != len(et_loc) or not np.allclose(et_hub, et_loc):
            raise SystemExit(f"ORDER MISMATCH {config} shot {i}: local {local[i].name} != Hub row {i}")
    print(f"  {config}: first {n} Hub rows match local sorted files (efit_times identical)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-pca", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "submission")
    ap.add_argument("--check-order", type=int, default=20, help="stream this many test rows from the Hub and compare efit_times to local files (0 = skip)")
    a = ap.parse_args()

    # ---- 1. train (same split + subsample as control_vs_mirror_val.py)
    t0 = time.time()
    files = sorted((DATA / "diii_d_train").glob("*.parquet"))
    N = len(files)
    n_train = N - max(1, int(N * 0.2)) - max(1, int(N * 0.1))
    train_idx = np.random.RandomState(42).permutation(N)[:n_train]
    sub_idx = np.random.RandomState(a.seed).choice(train_idx, min(a.n_train, n_train), replace=False)
    X, Y, Q, B = [], [], [], []
    for i in sub_idx:
        r, s = features(files[i], "diii_d_train")
        psi = load_psi(files[i])
        n = min(len(r["frames"]), len(psi))
        X.append(mirror(r["frames"][SIGS].to_numpy()[:n], s)); Y.append(psi[:n])
        Q.append(r["frames"]["efit_q95"].to_numpy()[:n]); B.append(r["frames"]["efit_beta_n"].to_numpy()[:n])
    X, Y, Q, B = map(np.concatenate, (X, Y, Q, B))
    print(f"train: {len(sub_idx)} shots, {len(X)} frames  [load {time.time() - t0:.0f}s]")

    mu = Y.mean(0)
    Y -= mu
    pca = PCA(a.n_pca, random_state=0).fit(Y)
    C = pca.transform(Y)
    del Y
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    alphas = np.logspace(-4, 4, 17)
    psi_model = RidgeCV(alphas=alphas).fit(Xs, C)
    scalar_models = {}
    for name, y in (("q95", Q), ("betaN", B)):
        ok = np.isfinite(y)
        lo, hi = np.percentile(y[ok], [0.1, 99.9])
        scalar_models[name] = RidgeCV(alphas=alphas).fit(Xs[ok], np.clip(y[ok], lo, hi))
        print(f"  {name}: labels clipped to [{lo:.3g}, {hi:.3g}], alpha={scalar_models[name].alpha_:g}")
    print(f"  psi: alpha={psi_model.alpha_:g}")
    model = {"sigs": SIGS, "scaler": sc, "pca": pca, "mu": mu, "psi": psi_model, "scalars": scalar_models,
             "note": f"mirror Ridge, {len(sub_idx)} train shots seed {a.seed}, {a.n_pca} PCA comps"}
    pickle.dump(model, open(MODEL_PATH, "wb"))
    print(f"saved model -> {MODEL_PATH}")

    # ---- 2. order check: local sorted files must match Hub stream order
    if a.check_order:
        print("\nChecking shot order against the Hub stream...")
        for config in ("diii_d_public_test", "mast_public_test"):
            check_order(config, a.check_order)

    # ---- 3. inference on the public test inputs
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    preds, n_rev = {}, 0
    test_files = sorted((DATA / "diii_d_public_test").glob("*.parquet"))
    for i, path in enumerate(test_files):
        r, s = features(path, "diii_d_public_test")
        T = len(pq.read_table(path, columns=["efit_times"]).column(0).combine_chunks()[0].values)
        Xs = sc.transform(mirror(r["frames"][SIGS].to_numpy(), s))
        assert len(Xs) == T and np.isfinite(Xs).all(), f"{path.name}: bad features"
        psi = pca.inverse_transform(psi_model.predict(Xs)) + mu
        preds[f"shot_{i:04d}_psirz"] = psi.reshape(T, 65, 65).astype(np.float16)
        for name, m in scalar_models.items():
            preds[f"shot_{i:04d}_{name}"] = m.predict(Xs).astype(np.float32)
        n_rev += s < 0
    np.savez_compressed(a.out / "diii_d_public_test.npz", **preds)
    print(f"\nDIII-D: {len(test_files)} shots ({n_rev} mirrored) -> {a.out / 'diii_d_public_test.npz'}  [{time.time() - t0:.0f}s]")

    preds = {}
    mast_files = sorted((DATA / "mast_public_test").glob("*.parquet"))
    for i, path in enumerate(mast_files):
        T = len(pq.read_table(path, columns=["efit_times"]).column(0).combine_chunks()[0].values)
        preds[f"shot_{i:04d}_psirz"] = np.zeros((T, 65, 65), dtype=np.float16)
        for name in scalar_models:
            preds[f"shot_{i:04d}_{name}"] = np.zeros(T, dtype=np.float32)
    np.savez_compressed(a.out / "mast_public_test.npz", **preds)
    print(f"MAST:   {len(mast_files)} shots (zeros) -> {a.out / 'mast_public_test.npz'}")


if __name__ == "__main__":
    main()
