"""Merge several synthetic MAST caches (same layout, same MAST region) into one training set, so that one batch per
step samples all of them in proportion to their size (several --synth sets would each add their own batch).

    python synth_merge.py OUT_TAG PRIMARY_TAG OTHER_TAG [...]
    e.g. FEC_MAST_MASK=limiter python synth_merge.py zara_g4_all_limiter zara_g4_20000_limiter zara_g4_5000_limiter

Training frames: every set's training frames. Held-out frames: the PRIMARY set's held-out frames keep split "heldout"
(what SynthSet logs every epoch); the others are kept as "heldout_<tag>", which SynthSet ignores. Stems and cases get a
prefix per set (file names repeat across sets). shape_targets.npz (synth_shape_targets.py) is merged when every set has it.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import CACHE
from synth_cache import _sha

PAD = ("te", "ne", "tr", "tz")


def main() -> int:
    out_tag, tags = sys.argv[1], sys.argv[2:]
    srcs = [CACHE / "synth" / t for t in tags]
    zs = [np.load(s / "arrays.npz") for s in srcs]
    mids = {int(z["mid"]) for z in zs}
    assert len(mids) == 1, f"different input windows: {mids}"
    metas = [json.loads((s / "manifest.json").read_text()) for s in srcs]
    assert len({m.get("mast_mask") for m in metas}) == 1, "different MAST regions"
    width = {k: max(z[k].shape[1] for z in zs) for k in PAD}
    arrays = {}
    for k in zs[0].files:
        if k == "mid":
            continue
        parts = []
        for i, (t, z) in enumerate(zip(tags, zs)):
            v = z[k]
            if k == "split" and i > 0:
                v = np.where(v == "heldout", f"heldout_{t}", v)
            elif k in ("stem", "case"):
                v = np.char.add(f"{t}:", v.astype(str))
            elif k in PAD and v.shape[1] < width[k]:
                v = np.concatenate([v, np.full((len(v), width[k] - v.shape[1]), np.nan, v.dtype)], 1)
            parts.append(v)
        arrays[k] = np.concatenate(parts)
    out = CACHE / "synth" / out_tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arrays.npz", mid=zs[0]["mid"], **arrays)
    if all((s / "shape_targets.npz").exists() for s in srcs):
        st = [np.load(s / "shape_targets.npz") for s in srcs]
        np.savez(out / "shape_targets.npz", **{k: np.concatenate([x[k] for x in st]) for k in st[0].files})
    split = arrays["split"]
    meta = {k: metas[0][k] for k in ("machine", "layout", "label_sign", "pictures", "mast_mask", "window") if k in metas[0]}
    meta.update({"merged_from": tags, "primary_heldout": tags[0], "n_frames": int(len(split)),
                 "train_frames_per_set": {t: int((z["split"] == "train").sum()) for t, z in zip(tags, zs)},
                 "split_counts": {s: int((split == s).sum()) for s in sorted(set(split.tolist()))},
                 "arrays_sha256": _sha(out / "arrays.npz")})
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {meta['split_counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
