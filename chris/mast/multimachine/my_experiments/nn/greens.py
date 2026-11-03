"""Vacuum Green's pictures for the physics-guidance baseline.

Each F-coil picture is poloidal flux (V·s/rad) per kA·turn on the 65×65 grid. The
rectangle is filled with a 5×5 filament grid and averaged. Parallelogram skew on
F5–F7 is ignored. ECOILA uses the same rectangle average, in flux per kA, because
that current is not in kA·turn. One plasma ring is a single filament at the
centroid of the in-envelope cells, in flux per kA·turn. bcoil has no poloidal rectangle.

    python my_experiments/nn/greens.py

Run this on the GPU allocation, not the login node. It writes `baseline_pictures.npz`
with `green` (18, 4225) in SIGS order F1A–F9B, `ecoil` (4225,), `ring` (4225,), and
the centroid. The check adds a plasma filament at the magnetic axis to the F-coil
sum and compares that with true ψ outside the envelope, after removing each frame's
mean. The axis is a label, used only in this check. The saved ring stays at the
centroid. The README's figure for the comparison is R² ≈ 0.94. If the correlation
is negative, the F-coil pictures, the E-coil picture, and the ring are multiplied
by −1, and the script says so. If the printed R² is far from 0.94, do not start
training: the 2π convention or the kA·turn scale is wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ellipe, ellipk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    CACHE, DATA_ROOT, F_COILS, F_SLICE, GRID, N_PSI, REPO, SCALAR_NAMES, SIGS,
    load_rz, train_files,
)

# μ₀ / 2π × 1000 A/kA. ψ = r A_φ, in V·s/rad, for a filament of 1 kA·turn.
_PER_KATURN = 2.0e-4
_N_FILAMENT = 5
DEFAULT_FRAMES = 48
DEFAULT_PATH = HERE / "fcoil_green.npy"
DEFAULT_BASELINE = HERE / "baseline_pictures.npz"


def filament_psi(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float) -> np.ndarray:
    """Poloidal flux (V·s/rad) of one filament carrying 1 kA·turn.

    k² is clipped just below 1 so a grid point that lands on a filament stays finite.
    """
    dist2 = (R + Rc) ** 2 + (Z - Zc) ** 2
    m = 4.0 * R * Rc / dist2
    m = np.clip(m, 1e-16, 1.0 - 1e-12)
    k = np.sqrt(m)
    bracket = (2.0 - m) * ellipk(m) - 2.0 * ellipe(m)
    return _PER_KATURN * np.sqrt(R * Rc) * bracket / k


def rectangle_psi(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float,
                  width: float, height: float, n: int = _N_FILAMENT) -> np.ndarray:
    """Average a uniform n×n filament grid over an axis-aligned rectangle.

    The offsets are cell centres, so the filaments fill the rectangle and do not
    sit on its boundary. Skew angles are not applied.
    """
    u = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5
    acc = np.zeros(R.shape, dtype=np.float64)
    for du in u:
        for dv in u:
            acc += filament_psi(R, Z, Rc + float(du) * width, Zc + float(dv) * height)
    return acc / (n * n)


def _cell(table, name) -> np.ndarray:
    values = table.column(name).combine_chunks()[0].values
    return values.to_numpy(zero_copy_only=False)


def read_rectangles(path: Path) -> list[tuple[float, float, float, float]]:
    """(R, Z, width, height) for F1A–F9B, in SIGS order, from one train parquet."""
    cols = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    table = pq.read_table(path, columns=cols)
    names = [str(v) for v in _cell(table, "coil_input_column")]
    centre_r = _cell(table, "coil_R")
    centre_z = _cell(table, "coil_Z")
    width = _cell(table, "coil_width")
    height = _cell(table, "coil_height")
    by_name: dict[str, tuple[float, float, float, float]] = {}
    for i, name in enumerate(names):
        key = name.removeprefix("magnetics_")
        if key not in F_COILS:
            continue
        if key in by_name:
            raise RuntimeError(f"{path} has two rows for {key}")
        rect = (float(centre_r[i]), float(centre_z[i]), float(width[i]), float(height[i]))
        if not np.isfinite(rect).all() or rect[2] <= 0 or rect[3] <= 0:
            raise RuntimeError(f"{key} rectangle is not usable: {rect}")
        by_name[key] = rect
    missing = [name for name in F_COILS if name not in by_name]
    if missing:
        raise RuntimeError(f"{path} is missing coil rectangles {missing}")
    return [by_name[name] for name in F_COILS]


def read_named_rectangle(path: Path, coil: str) -> tuple[float, float, float, float]:
    """(R, Z, width, height) for one coil, from one train parquet."""
    cols = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    table = pq.read_table(path, columns=cols)
    names = [str(v) for v in _cell(table, "coil_input_column")]
    centre_r = _cell(table, "coil_R")
    centre_z = _cell(table, "coil_Z")
    width = _cell(table, "coil_width")
    height = _cell(table, "coil_height")
    found: list[tuple[float, float, float, float]] = []
    for i, name in enumerate(names):
        if name.removeprefix("magnetics_") != coil:
            continue
        rect = (float(centre_r[i]), float(centre_z[i]), float(width[i]), float(height[i]))
        if not np.isfinite(rect).all() or rect[2] <= 0 or rect[3] <= 0:
            raise RuntimeError(f"{coil} rectangle is not usable: {rect}")
        found.append(rect)
    if len(found) != 1:
        raise RuntimeError(f"{path} has {len(found)} rectangles for {coil}")
    return found[0]


def envelope_centroid() -> tuple[float, float]:
    """Mean R and Z, in meters, of cells inside the DIII-D envelope.

    This is a fixed ring position. The magnetic axis is a label and is not used.
    """
    mask_path = REPO / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    inside = np.load(mask_path)["mask_coarse"].astype(bool)
    if inside.shape != (GRID, GRID):
        raise RuntimeError(f"{mask_path} mask_coarse {inside.shape}, expected {(GRID, GRID)}")
    if int(inside.sum()) == 0:
        raise RuntimeError(f"{mask_path} envelope mask is empty")
    r, z = load_rz()
    rr, zz = np.meshgrid(r, z, indexing="xy")
    return float(rr[inside].mean()), float(zz[inside].mean())


def grid_points() -> tuple[np.ndarray, np.ndarray]:
    """Flat R and Z of the 65×65 grid. Index iz * 65 + ir matches psi.npy."""
    r, z = load_rz()
    rr, zz = np.meshgrid(r, z, indexing="xy")
    R = np.ascontiguousarray(rr, dtype=np.float64).ravel()
    Z = np.ascontiguousarray(zz, dtype=np.float64).ravel()
    if R.size != N_PSI:
        raise RuntimeError(f"grid has {R.size} points, expected {N_PSI}")
    return R, Z


def compute_pictures(rectangles) -> np.ndarray:
    """(18, 4225) vacuum flux per kA·turn. Row order is F_COILS."""
    R, Z = grid_points()
    pictures = np.empty((len(rectangles), N_PSI), dtype=np.float64)
    for i, (rc, zc, width, height) in enumerate(rectangles):
        pictures[i] = rectangle_psi(R, Z, rc, zc, width, height)
    if not np.isfinite(pictures).all():
        raise RuntimeError("Green's pictures are not finite")
    return pictures


def compute_ecoil(rect: tuple[float, float, float, float]) -> np.ndarray:
    """(4225,) vacuum flux per kA for the ECOILA rectangle."""
    R, Z = grid_points()
    picture = rectangle_psi(R, Z, rect[0], rect[1], rect[2], rect[3])
    if picture.shape != (N_PSI,) or not np.isfinite(picture).all():
        raise RuntimeError("E-coil picture is not finite")
    return picture


def compute_ring(r0: float, z0: float) -> np.ndarray:
    """(4225,) vacuum flux per kA·turn of one filament at (r0, z0)."""
    R, Z = grid_points()
    picture = filament_psi(R, Z, r0, z0)
    if picture.shape != (N_PSI,) or not np.isfinite(picture).all():
        raise RuntimeError("ring picture is not finite")
    return picture


def outside_cells() -> np.ndarray:
    """Flat boolean, True on cells outside the DIII-D envelope."""
    mask_path = REPO / "fusion_scoring" / "masks" / "d3d_envelope.npz"
    mask = np.load(mask_path)["mask_coarse"].astype(bool)
    if mask.shape != (GRID, GRID):
        raise RuntimeError(f"{mask_path} mask_coarse {mask.shape}, expected {(GRID, GRID)}")
    return ~mask.reshape(-1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.square(a).sum() * np.square(b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    tot = np.square(true - true.mean()).sum()
    if tot <= 0:
        return float("nan")
    return float(1.0 - np.square(pred - true).sum() / tot)


def _frame_outside(pred: np.ndarray, psi: np.ndarray, outside: np.ndarray):
    """Outside-envelope values with each frame's mean removed.

    EFIT's zero of ψ is not the vacuum gauge, and it moves from frame to frame.
    A pooled R² without this step is mostly that offset.
    """
    p = pred[:, outside]
    t = psi[:, outside]
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    return p.ravel(), t.ravel()


def plasma_flux(ip: np.ndarray, r_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Vacuum ψ of a filament at the magnetic axis carrying Ip (kA = kA·turn).

    Positive Ip uses the same sign as `filament_psi`. The caller flips this with
    the F-coil pictures when the correlation is negative.
    """
    R, Z = grid_points()
    out = np.empty((len(ip), R.size), dtype=np.float64)
    for i in range(len(ip)):
        out[i] = filament_psi(R, Z, float(r_axis[i]), float(z_axis[i])) * float(ip[i])
    return out


def apply_sign(pictures: np.ndarray, currents: np.ndarray, psi: np.ndarray,
               outside: np.ndarray, plasma: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return pictures, correlation, and outside-envelope R².

    `currents` is (N, 18) physical kA·turn, `plasma` is the axis filament in the
    same sign convention as `pictures`, and `psi` is (N, 4225). The comparison
    removes each frame's mean. The overall sign is flipped when the correlation
    is negative.
    """
    pred = currents @ pictures + plasma
    p, t = _frame_outside(pred, psi, outside)
    corr = _corr(p, t)
    if corr < 0:
        print(
            f"correlation {corr:.4f} is negative; flipping the sign of all 18 pictures",
            flush=True,
        )
        pictures = -np.array(pictures, copy=True)
        p = -p
        corr = -corr
    return pictures, corr, _r2(p, t)


def _check_frames(cache: Path, n_frames: int):
    """F currents, true ψ, Ip (kA), and the magnetic axis for normal train frames."""
    ready = cache / "READY"
    if not ready.exists():
        raise SystemExit(f"cache is not ready ({ready}). The sign check needs cached train frames.")
    shots = pd.read_parquet(cache / "shots.parquet")
    train = shots[(shots["split"] == "train") & (shots["s"] > 0) & (shots["n"] > 0)]
    train = train.sort_values("filename")
    if train.empty:
        raise RuntimeError("no normal train shots in the cache")
    n = min(int(n_frames), len(train))
    pick = np.unique(np.linspace(0, len(train) - 1, n).astype(int))
    rows = train.iloc[pick]
    coils = np.load(cache / "coils.npy", mmap_mode="r")
    psi = np.load(cache / "psi.npy", mmap_mode="r")
    scal = np.load(cache / "scal.npy", mmap_mode="r")
    ip_col = SIGS.index("plasma_current")
    r_col = SCALAR_NAMES.index("R_axis")
    z_col = SCALAR_NAMES.index("Z_axis")
    currents = np.empty((len(rows), len(F_COILS)), dtype=np.float64)
    truth = np.empty((len(rows), N_PSI), dtype=np.float64)
    ip = np.empty(len(rows), dtype=np.float64)
    r_axis = np.empty(len(rows), dtype=np.float64)
    z_axis = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows.itertuples(index=False)):
        mid = int(row.start) + int(row.n) // 2
        currents[i] = np.asarray(coils[mid, F_SLICE], dtype=np.float64)
        truth[i] = np.asarray(psi[mid], dtype=np.float64)
        ip[i] = float(coils[mid, ip_col])
        r_axis[i] = float(scal[mid, r_col])
        z_axis[i] = float(scal[mid, z_col])
    ok = (
        np.isfinite(currents).all(axis=1)
        & np.isfinite(truth).all(axis=1)
        & np.isfinite(ip)
        & np.isfinite(r_axis)
        & np.isfinite(z_axis)
    )
    if not ok.all():
        currents, truth, ip, r_axis, z_axis = (
            currents[ok], truth[ok], ip[ok], r_axis[ok], z_axis[ok]
        )
    if len(currents) == 0:
        raise RuntimeError("no check frames with a finite axis and finite ψ")
    return currents, truth, ip, r_axis, z_axis


def score_pictures(pictures: np.ndarray, cache: Path, n_frames: int) -> tuple[np.ndarray, float]:
    """Sign-correct `pictures` and print the outside-envelope R². Returns (pictures, R²)."""
    currents, truth, ip, r_axis, z_axis = _check_frames(cache, n_frames)
    outside = outside_cells()
    plasma = plasma_flux(ip, r_axis, z_axis)
    pictures, corr, r2 = apply_sign(pictures, currents, truth, outside, plasma)
    print(
        f"outside-envelope R² {r2:.4f}  correlation {corr:.4f}  "
        f"({currents.shape[0]} normal train frames, {int(outside.sum())} cells, "
        "per-frame mean removed). "
        "Sum is the 18 F-coils plus a plasma filament at the axis. "
        "The README comparison is R² ≈ 0.94.",
        flush=True,
    )
    if not np.isfinite(r2) or r2 < 0.5:
        print(
            "This R² is far from 0.94. Do not start training: "
            "the 2π convention or the kA·turn scale is wrong.",
            flush=True,
        )
    return pictures, r2


def load_pictures(path: Path) -> np.ndarray:
    path = Path(path)
    pictures = np.load(path)
    expected = (len(F_COILS), N_PSI)
    if pictures.shape != expected or not np.isfinite(pictures).all():
        raise RuntimeError(
            f"{path} should be finite with shape {expected}, got {pictures.shape} {pictures.dtype}"
        )
    return np.ascontiguousarray(pictures, dtype=np.float32)


def build(data_root: Path, cache: Path, out: Path, frames: int = DEFAULT_FRAMES) -> np.ndarray:
    """Compute the pictures, sign-check them, and write `out`. Does not train."""
    path = train_files(data_root)[0]
    rectangles = read_rectangles(path)
    print(f"18 F-coil rectangles from {path.name} (skew on F5–F7 ignored)", flush=True)
    pictures = compute_pictures(rectangles)
    pictures, _r2 = score_pictures(pictures, cache, frames)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stored = np.ascontiguousarray(pictures, dtype=np.float32)
    np.save(out, stored)
    print(f"wrote {out} {stored.shape} {stored.dtype}", flush=True)
    return stored


def load_or_build(path: Path, data_root: Path, cache: Path,
                  frames: int = DEFAULT_FRAMES) -> np.ndarray:
    """Load `fcoil_green.npy`, or build it when the file is absent."""
    path = Path(path)
    if path.exists():
        pictures = load_pictures(path)
        print(f"loaded F-coil pictures {tuple(pictures.shape)} from {path}", flush=True)
        return pictures
    print(f"F-coil pictures are absent ({path}); building them", flush=True)
    return build(data_root, cache, path, frames)


def _picture_sign(original: np.ndarray, signed: np.ndarray) -> float:
    """+1 when `signed` matches `original`, −1 when the check flipped it."""
    dot = float(np.dot(np.asarray(signed, dtype=np.float64).ravel(), np.asarray(original, dtype=np.float64).ravel()))
    if dot > 0:
        return 1.0
    if dot < 0:
        return -1.0
    raise RuntimeError("could not read the sign of the F-coil pictures")


def load_baseline(path: Path) -> dict:
    """`baseline_pictures.npz` -> green (18, 4225), ecoil (4225,), ring (4225,), centroid."""
    path = Path(path)
    with np.load(path) as data:
        keys = set(data.files)
        need = {"green", "ecoil", "ring", "centroid_r", "centroid_z"}
        if keys != need and not need.issubset(keys):
            raise RuntimeError(f"{path} is missing {sorted(need - keys)}")
        green = np.ascontiguousarray(data["green"], dtype=np.float32)
        ecoil = np.ascontiguousarray(data["ecoil"], dtype=np.float32)
        ring = np.ascontiguousarray(data["ring"], dtype=np.float32)
        centroid_r = float(data["centroid_r"])
        centroid_z = float(data["centroid_z"])
    expected = (len(F_COILS), N_PSI)
    if green.shape != expected or ecoil.shape != (N_PSI,) or ring.shape != (N_PSI,):
        raise RuntimeError(
            f"{path} shapes green {green.shape} ecoil {ecoil.shape} ring {ring.shape}, "
            f"expected {expected}, {(N_PSI,)}, {(N_PSI,)}"
        )
    if not (np.isfinite(green).all() and np.isfinite(ecoil).all() and np.isfinite(ring).all()):
        raise RuntimeError(f"{path} has non-finite pictures")
    if not (np.isfinite(centroid_r) and np.isfinite(centroid_z)):
        raise RuntimeError(f"{path} centroid is not finite")
    return {
        "green": green,
        "ecoil": ecoil,
        "ring": ring,
        "centroid_r": centroid_r,
        "centroid_z": centroid_z,
    }


def build_baseline(data_root: Path, cache: Path, out: Path, frames: int = DEFAULT_FRAMES) -> dict:
    """Compute the three pictures, sign-check them, and write `out`. Does not train.

    The sign check flips the F-coil pictures when their correlation with true ψ is
    negative. The E-coil picture and the ring use the same filament convention, so
    they take that same sign.
    """
    path = train_files(data_root)[0]
    rectangles = read_rectangles(path)
    ecoil_rect = read_named_rectangle(path, "ECOILA")
    r0, z0 = envelope_centroid()
    print(
        f"18 F-coil rectangles and ECOILA {ecoil_rect} from {path.name}; "
        f"ring at envelope centroid ({r0:.4f}, {z0:.4f}) m",
        flush=True,
    )
    green = compute_pictures(rectangles)
    ecoil = compute_ecoil(ecoil_rect)
    ring = compute_ring(r0, z0)
    signed, r2 = score_pictures(green, cache, frames)
    sign = _picture_sign(green, signed)
    payload = {
        "green": np.ascontiguousarray(signed, dtype=np.float32),
        "ecoil": np.ascontiguousarray(ecoil * sign, dtype=np.float32),
        "ring": np.ascontiguousarray(ring * sign, dtype=np.float32),
        "centroid_r": np.float32(r0),
        "centroid_z": np.float32(z0),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(
        f"wrote {out}  green {payload['green'].shape}  ecoil {payload['ecoil'].shape}  "
        f"ring {payload['ring'].shape}  sign {sign:+.0f}  outside-envelope R² {r2:.4f}",
        flush=True,
    )
    return {
        "green": payload["green"],
        "ecoil": payload["ecoil"],
        "ring": payload["ring"],
        "centroid_r": r0,
        "centroid_z": z0,
    }


def load_or_build_baseline(path: Path, data_root: Path, cache: Path,
                           frames: int = DEFAULT_FRAMES) -> dict:
    """Load `baseline_pictures.npz`, or build it when the file is absent."""
    path = Path(path)
    if path.exists():
        pictures = load_baseline(path)
        print(
            f"loaded physics pictures from {path}  "
            f"centroid ({pictures['centroid_r']:.4f}, {pictures['centroid_z']:.4f}) m",
            flush=True,
        )
        return pictures
    print(f"physics pictures are absent ({path}); building them", flush=True)
    return build_baseline(data_root, cache, path, frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="normal train frames in the R² check")
    args = ap.parse_args()
    build_baseline(args.data_root, args.cache, args.out, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
