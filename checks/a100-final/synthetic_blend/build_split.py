"""Pool labelled synthetic MAST held-out sets into one Final-like split for predict.sh B, labels stripped.

    python build_split.py ROOT --workers 48

ROOT/data/mast_private_test/mast_shot_<id>.parquet  inputs only: the Final files' columns, in their order and types;
                                                    source "MAST", as in the Final files (r1 skips other sources)
ROOT/labels/mast_shot_<id>.parquet                  symlink to the original labelled file (scoring only)
ROOT/stems/<set>.txt, ROOT/map.tsv                  set membership; id = sha1("<set>/<stem>")[:10]
"""
import argparse, hashlib, os, sys
from multiprocessing import Pool
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

FINAL = Path("/fsx/chris.beckham/fec_pkgtest/work/data/final/data/mast_private_test")
M2, GS = Path("/fsx/zara.ercan/more_data/m2"), Path("/fsx/chris.beckham/gsolve_data")
SETS = {
    "zval": Path("/fsx/zara.ercan/cache/ch2/g4/eval/data"),       # Zara's internal validation (shipped Thomson)
    "g4_20000": M2 / "gen4_set20000/mast_g4s20000_heldout",
    "g4_5000": M2 / "gen4_set5000/mast_g4s5000_heldout",
    "g4_v3": M2 / "gen4_set5000v3/mast_g4s5000v3_heldout",
    "g4_v4": M2 / "gen4_set5000v4/mast_g4s5000v4_heldout",
    "g4_v4b": M2 / "gen4_set5000v4/mast_g4s5000v4b_heldout",
    "g4_v2": M2 / "gen4_set500v2/mast_g4s500v2_heldout",
    "g4_v6": M2 / "gen4_set1500v6/mast_g4s1500v6_heldout",
    "g4_v7": M2 / "gen4_set1500v7/mast_g4s1500v7_heldout",
    "gs1000": GS / "s1000/gen4_layout/heldout",
    "gst1000": GS / "tuned1000/gen4_layout/heldout",
    "gsc1000": GS / "cond1000/gen4_layout/heldout",
    "gsc2z1000": GS / "cond2z1000/gen4_layout/heldout",
    "gsc3_1000": GS / "cond3_1000/gen4_layout/heldout",
    "gsx8": GS / "early_x8/gen4_layout/heldout",
}
SCHEMA = pq.read_schema(next(FINAL.glob("*.parquet")))


def one(job):
    name, src, out_data, out_lab = job
    t = pq.read_table(src, columns=SCHEMA.names).cast(SCHEMA)
    t = t.set_column(t.schema.get_field_index("source"), "source", pa.array(["MAST"] * t.num_rows, pa.string()))  # as the Final files
    pq.write_table(t, out_data)
    if not os.path.lexists(out_lab):
        os.symlink(src, out_lab)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    D, L, S = a.root / "data/mast_private_test", a.root / "labels", a.root / "stems"
    for d in (D, L, S):
        d.mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    for s, folder in SETS.items():
        names = []
        for f in sorted(folder.glob("*.parquet")):
            fields = pq.read_schema(f).names
            if "efit_psirz" not in fields or any(n not in fields for n in SCHEMA.names):
                continue                                   # metadata files, not shots
            i = hashlib.sha1(f"{s}/{f.stem}".encode()).hexdigest()[:10]
            n = f"mast_shot_{i}"
            names.append(n); rows.append(f"{n}\t{s}\t{f}")
            jobs.append((n, str(f), str(D / f"{n}.parquet"), str(L / f"{n}.parquet")))
        (S / f"{s}.txt").write_text("".join(x + "\n" for x in names))
        print(f"{s:10s} {len(names):5d} {folder}", flush=True)
    assert len({j[0] for j in jobs}) == len(jobs), "id collision"
    (a.root / "map.tsv").write_text("".join(r + "\n" for r in rows))
    with Pool(a.workers) as p:
        for k, _ in enumerate(p.imap_unordered(one, jobs, chunksize=16)):
            if k % 2000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    print(f"done: {len(jobs)} shots", flush=True)


if __name__ == "__main__":
    sys.exit(main())
