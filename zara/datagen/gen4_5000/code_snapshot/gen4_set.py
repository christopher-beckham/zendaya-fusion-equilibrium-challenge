"""gen4 case set: N short synthetic MAST sequences in the release layout, drawn in proportion to the real input mix.

    python gen4_set.py --build-candidates                      # strata, quotas, candidate list (inputs only)
    python gen4_set.py --run --node-tag s2 --n-workers 40      # watchdog pool; workers claim candidates
    python gen4_set.py --assemble                              # pick the cases, write the parquet sets + summary

A case is centred on a real mast_public_test EFIT frame (|Ip| >= 150 kA). Its 9 consecutive EFIT frames (centre +-4) are
solved with gen4's plasma model (gen4.py docstring): the centre frame first, then a sweep forward and backward with warm
starts. The case file holds the real input series in +-100 ms around the centre (coil groups scaled by small per-case
factors), synthetic Thomson at every real Thomson time near a solved frame (own channel noise and a ~1 cm radial shift
per slice, the real slice's missing channels kept missing), and labels at the solved frames.

Sampling follows the real frames: strata = Ip quintile x normalised dIp/dt tercile (both from the real inputs), quota per
stratum = N x the stratum's share of real frames. A failed candidate is replaced by the next candidate of the same stratum,
so solver failures do not shift the mix. Train cases come from train-pool shots, held-out cases from the gen_m2 val-hash
shots (never train on them). MAST inputs only (Rule 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen4 as G  # noqa: E402

OUT = "/fsx/zara.ercan/more_data/m2/gen4_set500"
POOL_PATH = "/fsx/zara.ercan/more_data/generated/cache/mast_input_pool.npz"
N_CASE = {"train": 500, "heldout": 64}
SET_NAME = "mast_g4s500"        # assembled folders: <SET_NAME>_train, <SET_NAME>_heldout
OVERSAMPLE = 8                  # candidates per quota slot
MIN_FRAMES = 1                  # labelled frames for a case to count (recipe v2: 3)
RECIPE = "v1"
HALF = 4                        # frames each side of the centre
WINDOW_MS = 100.0
SYN_DT_CASE = 30.0              # ms: Thomson slices synthesised from a solved frame up to this far away
IP_MIN = 150.0


def pool_frames():
    """Real frames (shot file, t, Ip, normalised dIp/dt) from the input pool (EFIT times of every public-test shot)."""
    p = np.load(POOL_PATH)
    cols = list(p["columns"])
    ip = p["values"][:, cols.index("magnetics_plasma_current")]
    shot, t = p["shot"].astype(str), p["t"].astype(float)
    order = np.lexsort((t, shot))
    shot, t, ip = shot[order], t[order], ip[order]
    didt = np.full(len(t), np.nan)
    for s in np.unique(shot):
        m = np.where(shot == s)[0]
        if len(m) >= 3:
            didt[m] = np.gradient(ip[m], t[m]) * 100.0 / np.maximum(np.abs(ip[m]), 1.0)   # fractional change per 100 ms
    keep = np.abs(ip) >= IP_MIN
    return shot[keep], t[keep], np.abs(ip[keep]), np.nan_to_num(didt[keep])


def build_candidates():
    shot, t, ip, didt = pool_frames()
    split = np.array(["train" if G.split_of(s) == "train" else "heldout" for s in shot])
    qi = np.quantile(ip, [0.2, 0.4, 0.6, 0.8])
    qd = np.quantile(didt, [1 / 3, 2 / 3])
    stratum = np.digitize(ip, qi) * 3 + np.digitize(didt, qd)
    share = np.bincount(stratum, minlength=15) / len(stratum)
    rng = np.random.default_rng([G.SEED, 500] if N_CASE["train"] == 500 else [G.SEED, 500, N_CASE["train"]])
    cands, quotas = [], {}
    for sp, n in N_CASE.items():
        q = np.floor(share * n).astype(int)
        for i in np.argsort(-(share * n - q))[: n - q.sum()]:            # largest remainders
            q[i] += 1
        quotas[sp] = q.tolist()
        for st in range(15):
            idx = np.where((stratum == st) & (split == sp))[0]
            if not len(idx) or not q[st]:
                continue
            pick = rng.choice(idx, size=min(len(idx), OVERSAMPLE * q[st]), replace=False)
            for rank, i in enumerate(pick):
                cands.append(dict(split=sp, stratum=int(st), rank=rank, file=str(shot[i]), t=float(t[i]), ip=float(ip[i]),
                                  didt=float(didt[i])))
    # processing order: by rank, so every stratum fills at the same pace
    cands.sort(key=lambda c: (c["rank"], c["split"] != "train", c["stratum"]))
    for i, c in enumerate(cands):
        c["id"] = i
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(quotas=quotas, ip_edges=qi.tolist(), didt_edges=qd.tolist(), share=share.tolist(), n_real_frames=int(len(t)),
                   candidates=cands), open(os.path.join(OUT, "candidates.json"), "w"))
    print(f"{len(cands)} candidates; real frames {len(t)}; quotas {quotas}")


def extend(stratum, split, n_more):
    """Append n_more candidates of one stratum (frames not yet in the list; same strata edges, new seed)."""
    C = load_cands()
    shot, t, ip, didt = pool_frames()
    sp = np.array(["train" if G.split_of(s) == "train" else "heldout" for s in shot])
    st = np.digitize(ip, C["ip_edges"]) * 3 + np.digitize(didt, C["didt_edges"])
    used = {(c["file"], round(c["t"], 3)) for c in C["candidates"]}
    idx = [i for i in np.where((st == stratum) & (sp == split))[0] if (str(shot[i]), round(float(t[i]), 3)) not in used]
    rng = np.random.default_rng([G.SEED, 501, stratum, len(C["candidates"])])
    pick = rng.choice(idx, size=min(len(idx), n_more), replace=False)
    r0 = 1 + max(c["rank"] for c in C["candidates"] if c["split"] == split and c["stratum"] == stratum)
    i0 = len(C["candidates"])
    for j, i in enumerate(pick):
        C["candidates"].append(dict(split=split, stratum=int(stratum), rank=r0 + j, file=str(shot[i]), t=float(t[i]),
                                    ip=float(ip[i]), didt=float(didt[i]), id=i0 + j))
    json.dump(C, open(os.path.join(OUT, "candidates.json"), "w"))
    print(f"stratum {stratum} {split}: +{len(pick)} candidates (ids {i0}-{i0 + len(pick) - 1}; {len(idx)} unused frames)")


def load_cands():
    return json.load(open(os.path.join(OUT, "candidates.json")))


def n_accepted(split, stratum):
    n = 0
    for f in glob.glob(os.path.join(OUT, "done", f"{split}_s{stratum:02d}_*.json")):
        try:
            n += json.load(open(f)).get("ok", 0) >= MIN_FRAMES
        except Exception:  # noqa: BLE001
            pass
    return n


def case_path(c):
    return os.path.join(OUT, "raw", f"cand_{c['id']:05d}.parquet")


def done_path(c):
    return os.path.join(OUT, "done", f"{c['split']}_s{c['stratum']:02d}_{c['id']:05d}.json")


def solve_case(M, c, lib, beat=None, hang=(), verbose=False):
    d = G.load_shot(c["file"])
    kc = int(np.argmin(np.abs(d["t_efit"] - c["t"])))
    rng_tag = 100000 + c["id"]                                       # per-case draws (realisation index slot)
    lo, hi = max(0, kc - HALF), min(len(d["t_efit"]) - 1, kc + HALF)
    te = d["t_efit"]                                                  # sparse EFIT times: stay inside the stored input window
    while lo < kc and te[kc] - te[lo] > WINDOW_MS - 5.0:
        lo += 1
    while hi > kc and te[hi] - te[kc] > WINDOW_MS - 5.0:
        hi -= 1
    starts = [kc] + [k for k in (kc - 1, kc + 1, kc - 2, kc + 2) if lo <= k <= hi]
    d, P, sig, frames = G.solve_shot(M, c["file"], rng_tag, hang_frames=hang, beat=beat, verbose=verbose, lib=lib,
                                     frame_range=(lo, hi), start_cands=starts[:3])
    return d, P, sig, frames, kc


def write_case(path, d, P, sig, frames, th, c, kc):
    """Release layout: one file per case; inputs cropped to +-WINDOW_MS around the centre frame."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tc = float(d["t_efit"][kc])
    tm = d["tm"]
    w = (tm >= tc - WINDOW_MS) & (tm <= tc + WINDOW_MS)
    ok = [f for f in frames if f["status"] == "ok"]
    t_ok = np.array([f["t"] for f in ok])
    tab = d["tab"]
    cols = {}
    for name in tab.schema.names:
        if name.startswith("efit_") and name not in ("efit_grid_R", "efit_grid_Z"):
            continue
        cols[name] = tab.column(name)
    cols["magnetics_time"] = pa.array([tm[w].tolist()], pa.list_(pa.float32()))
    for col in G.SIG:
        cols[col] = pa.array([sig[col][w].tolist()], pa.list_(pa.float64()))
    for side in ("core", "edge"):
        tt = d[f"{side}_t"]
        te, ne = th[side]
        m = (tt >= t_ok.min() - SYN_DT_CASE) & (tt <= t_ok.max() + SYN_DT_CASE) & (np.abs(tt - tc) <= WINDOW_MS)
        cols[f"thomson_{side}_times"] = pa.array([tt[m].tolist()], pa.list_(pa.float64()))
        cols[f"thomson_{side}_Te"] = pa.array([te[m].tolist()], pa.list_(pa.list_(pa.float64())))
        cols[f"thomson_{side}_ne"] = pa.array([ne[m].tolist()], pa.list_(pa.list_(pa.float64())))
    cols["efit_times"] = pa.array([t_ok.tolist()], pa.list_(pa.float32()))
    cols["efit_psirz"] = pa.array([[f["psi"].tolist() for f in ok]], pa.list_(pa.list_(pa.list_(pa.float64()))))
    for col, key in G.LABEL_KEYS:
        cols[col] = pa.array([[float(f["labels"][key]) for f in ok]], pa.list_(pa.float64()))
    cols["efit_lcfs_n"] = pa.array([[float(len(f["lcfs"])) for f in ok]], pa.list_(pa.float64()))
    nl = max(len(f["lcfs"]) for f in ok)                 # challenge convention: rows padded with zeros, valid count in lcfs_n
    pad = lambda v: list(v) + [0.0] * (nl - len(v))
    cols["efit_lcfs_r"] = pa.array([[pad(f["lcfs"][:, 0].tolist()) for f in ok]], pa.list_(pa.list_(pa.float64())))
    cols["efit_lcfs_z"] = pa.array([[pad(f["lcfs"][:, 1].tolist()) for f in ok]], pa.list_(pa.list_(pa.float64())))
    cols["source"] = pa.array(["MAST_synthetic_freegsnke_gen4_case"], pa.string())
    meta = dict(version=G.VERSION, recipe=RECIPE, case=c, centre_t=tc, sol_factor=G.SOL_FACTOR, jitter=P["jit"], nf=P["nf"],
                am=P["am"], lam_ne=P["lam_ne"], lam_te=P["lam_te"], te_sep=P["te_sep"], tn_c=P.get("tn_c", 0.0),
                w_max=G.W_MAX, r_jitter=G.R_JITTER)
    cols["synth_meta"] = pa.array([json.dumps(meta)], pa.string())
    diag = []
    for f in frames:
        if f["status"] == "outside":
            continue
        g = {k: (v.item() if isinstance(v, (np.floating, np.bool_)) else v) for k, v in f.items() if k in G.FRAME_DIAG}
        if f["status"] == "ok":
            g.update(volume=f["labels"]["volume"], minor_radius=f["labels"]["minor_radius"], beta_t=f["labels"]["beta_t"])
        if "tries" in f:
            g["tries"] = f["tries"]
        diag.append(g)
    cols["synth_frames"] = pa.array([json.dumps(diag)], pa.string())
    tmp = path + ".tmp"
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    os.replace(tmp, path)


def acc_path(c):
    """Empty marker of an accepted case (counted by directory listing, without reading the outcome files)."""
    return os.path.join(OUT, "done", f"acc_{c['split']}_s{c['stratum']:02d}_{c['id']:05d}")


class Progress:
    """Finished candidate ids and accepted counts per split and stratum, from one listing of done/, refreshed every 60 s."""

    def __init__(self):
        self.t, self.done, self.acc = 0.0, set(), {}

    def refresh(self, force=False):
        if not force and time.time() - self.t < 60:
            return
        done, acc = set(), {}
        for name in os.listdir(os.path.join(OUT, "done")):
            if name.startswith("acc_"):
                key = name[4:].rsplit("_", 1)[0]
                acc[key] = acc.get(key, 0) + 1
            elif name.endswith(".json"):
                done.add(int(name[:-5].rsplit("_", 1)[1]))
        self.t, self.done, self.acc = time.time(), done, acc

    def full(self, c, quotas):
        return self.acc.get(f"{c['split']}_s{c['stratum']:02d}", 0) >= quotas[c["split"]][c["stratum"]] + 2   # small surplus

    def add(self, c, accepted):
        self.done.add(c["id"])
        if accepted:
            key = f"{c['split']}_s{c['stratum']:02d}"
            self.acc[key] = self.acc.get(key, 0) + 1


def worker(k, nw, node_tag):
    from pool import beat as pool_beat
    hb = os.path.join(OUT, "hb", node_tag)
    me = f"{node_tag}:{k}"
    C = load_cands()
    quotas = C["quotas"]
    M = G.Machine()
    lib = G.SeedLib()
    PR = Progress()
    PR.refresh(force=True)
    t0 = time.time(); n = n_ok = 0
    claims = os.path.join(OUT, "claims")
    # resume the candidate this worker slot was solving when the watchdog restarted it (its heartbeat holds "<id>|<frame>")
    resume = []
    try:
        cid, stage = open(os.path.join(hb, f"hb.w{k}")).read().strip().split("|")
        if stage not in ("skip", "write") and int(cid) not in PR.done:
            resume = [c for c in C["candidates"] if c["id"] == int(cid)][:1]
    except (OSError, ValueError):
        pass
    pool_beat(hb, k, "-1|skip")
    for i, c in enumerate(resume + C["candidates"]):
        if i % 200 == 0:                       # skipping thousands of candidates of full strata must not look like a hang
            PR.refresh()
            pool_beat(hb, k, f"{c['id']}|skip")
        if c["id"] in PR.done:
            continue
        cp = os.path.join(claims, str(c["id"]))
        if not (resume and c is resume[0]):
            if PR.full(c, quotas):
                continue
            try:
                fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, me.encode()); os.close(fd)
        if os.path.exists(done_path(c)):
            continue
        hang_file = os.path.join(OUT, "hangs", f"{c['id']}.txt")
        hang = {int(x) for x in open(hang_file).read().split()} if os.path.exists(hang_file) else set()
        tc = time.time()
        try:
            d, P, sig, frames, kc = solve_case(M, c, lib, beat=lambda fi: pool_beat(hb, k, f"{c['id']}|{fi}"), hang=hang)
            nok = sum(f["status"] == "ok" for f in frames)
            pool_beat(hb, k, f"{c['id']}|write")
            if nok:
                th = G.synth_thomson(M, d, P, frames, max_dt=SYN_DT_CASE)
                write_case(case_path(c), d, P, sig, frames, th, c, kc)
            st = {}
            for f in frames:
                if f["status"] != "outside":
                    st[f["status"]] = st.get(f["status"], 0) + 1
            json.dump(dict(id=c["id"], ok=nok, centre_ok=frames[kc]["status"] == "ok", status=st, wall=time.time() - tc, node=me),
                      open(done_path(c), "w"))
            if nok >= MIN_FRAMES:
                open(acc_path(c), "w").close()
            PR.add(c, nok >= MIN_FRAMES)
            n += 1; n_ok += nok > 0
            print(f"[{me}] cand {c['id']} {c['split']} s{c['stratum']:02d}: {nok} frames ok, {time.time() - tc:.0f}s | "
                  f"{n_ok}/{n} cases, {(time.time() - t0) / max(n, 1):.0f} s/case", flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            json.dump(dict(id=c["id"], ok=0, error=traceback.format_exc()[-400:], node=me), open(done_path(c), "w"))
            M = G.Machine()
    print(f"[{me}] finished: {n_ok}/{n} cases in {(time.time() - t0) / 3600:.2f} h", flush=True)


def run(a):
    from pool import run_pool
    for sub in ("done", "hangs", "claims", "raw", os.path.join("hb", a.node_tag)):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    # accepted-case markers for outcomes written before the markers existed (idempotent; several nodes may run it)
    done_dir = os.path.join(OUT, "done")
    names = os.listdir(done_dir)
    have = {x for x in names if x.startswith("acc_")}
    for name in names:
        if name.endswith(".json") and "acc_" + name[:-5] not in have:
            try:
                ok = json.load(open(os.path.join(done_dir, name))).get("ok", 0)
            except Exception:  # noqa: BLE001
                continue
            if ok >= MIN_FRAMES:
                open(os.path.join(done_dir, "acc_" + name[:-5]), "w").close()
    by_id = {c["id"]: c for c in load_cands()["candidates"]}

    def on_hang(k, tag):
        cid, fi = tag.split("|")
        if fi == "skip":
            return
        if fi == "write":                      # a stall after the solve: record the candidate as failed, never retry it
            c = by_id.get(int(cid))
            if c is not None and not os.path.exists(done_path(c)):
                json.dump(dict(id=c["id"], ok=0, error="stalled after the solve"), open(done_path(c), "w"))
            return
        with open(os.path.join(OUT, "hangs", f"{cid}.txt"), "a") as fh:
            fh.write(f"{fi}\n")

    n = run_pool(worker, a.n_workers, (a.node_tag,), os.path.join(OUT, "hb", a.node_tag), stall=a.stall, on_hang=on_hang)
    print(f"pool finished, {n} hang restarts", flush=True)


def pad_lcfs(src, dst):
    """Copy a case file with its LCFS rows padded with zeros to one length (valid count in efit_lcfs_n)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tab = pq.read_table(src)
    for col in ("efit_lcfs_r", "efit_lcfs_z"):
        rows = tab.column(col)[0].as_py()
        nl = max(len(r) for r in rows)
        rows = [r + [0.0] * (nl - len(r)) for r in rows]
        tab = tab.set_column(tab.schema.get_field_index(col), col, pa.array([rows], pa.list_(pa.list_(pa.float64()))))
    pq.write_table(tab, dst, compression="zstd")


def has_thomson(path, min_ch=20):
    """Usable case: at least one core Thomson slice with min_ch valid channels (cases without any are not calibrated on
    Thomson), and every labelled frame inside the stored input window (in shots with sparse EFIT times the 9-frame segment
    can reach beyond the +-WINDOW_MS of inputs written around the centre frame)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["thomson_core_Te", "efit_times", "magnetics_time"])
    te = t.column("thomson_core_Te")[0].as_py()
    tm = np.asarray(t.column("magnetics_time")[0].as_py(), float)
    tk = np.asarray(t.column("efit_times")[0].as_py(), float)
    in_window = len(tm) > 0 and bool(np.all((tk >= tm.min() + 2.0) & (tk <= tm.max() - 2.0)))
    return in_window and any(np.isfinite(np.asarray(r[1:], float)).sum() >= min_ch for r in te)


def assemble():
    """First `quota` accepted candidates (by rank) per stratum and split -> <SET_NAME>_{train,heldout}/."""
    C = load_cands()
    out = {}
    for sp in N_CASE:
        dst = os.path.join(OUT, f"{SET_NAME}_{sp}")
        os.makedirs(dst, exist_ok=True)
        for old in glob.glob(os.path.join(dst, "*.parquet")):
            os.remove(old)
        chosen = []
        for st in range(15):
            q = C["quotas"][sp][st]
            got = []
            for c in sorted([c for c in C["candidates"] if c["split"] == sp and c["stratum"] == st], key=lambda c: c["rank"]):
                if len(got) >= q:
                    break
                if (os.path.exists(done_path(c)) and json.load(open(done_path(c))).get("ok", 0) >= MIN_FRAMES
                        and has_thomson(case_path(c))):
                    got.append(c)
            chosen += got
            out[f"{sp}_s{st:02d}"] = (len(got), q)
        for i, c in enumerate(sorted(chosen, key=lambda c: c["id"])):
            pad_lcfs(case_path(c), os.path.join(dst, f"mast_synth_g4_{i:05d}.parquet"))
        print(f"{sp}: {len(chosen)} cases -> {dst}")
    short = {k: v for k, v in out.items() if v[0] < v[1]}
    print("strata below quota:", short or "none")
    json.dump(out, open(os.path.join(OUT, "assembly.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-candidates", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--extend", nargs=3, default=None, metavar=("STRATUM", "SPLIT", "N"))
    ap.add_argument("--node-tag", default=socket.gethostname())
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--stall", type=float, default=150.0)
    ap.add_argument("--out", default=None, help="set directory (default the 500-case set)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-heldout", type=int, default=None)
    ap.add_argument("--name", default=None, help="assembled folder prefix")
    ap.add_argument("--recipe", default="v1", choices=["v1", "v2"],
                    help="v2 = FEEDBACK_FOR_ZARA_TRAINING.md (2026-10-08 01:13): current shape independent of Thomson, "
                         "random Te/ne split at fixed pressure, no extra radial shift, >= 3 labelled frames per case")
    a = ap.parse_args()
    global OUT, SET_NAME, MIN_FRAMES, OVERSAMPLE, RECIPE
    RECIPE = a.recipe
    if a.recipe == "v2":
        G.W_MAX, G.TN_SPLIT, G.R_JITTER, G.SWEEP_BUDGET = 0.0, 0.6, 0.0, 14
        MIN_FRAMES, OVERSAMPLE = 3, 16
    if a.out:
        OUT = a.out
    if a.name:
        SET_NAME = a.name
    if a.n_train is not None:
        N_CASE["train"] = a.n_train
    if a.n_heldout is not None:
        N_CASE["heldout"] = a.n_heldout
    if a.extend:
        extend(int(a.extend[0]), a.extend[1], int(a.extend[2]))
    elif a.build_candidates:
        build_candidates()
    elif a.run:
        run(a)
    elif a.assemble:
        assemble()


if __name__ == "__main__":
    main()
