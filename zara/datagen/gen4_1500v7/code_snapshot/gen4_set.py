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
import hashlib
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
N_REAL = 1                      # realisations per real frame (new draws of the unmeasured parameters for each)
BLEND_P = 0.0                   # v4: share of cases whose coil currents blend two real shots
N_GROUPS, GROUP_SIZE = 0, 5     # v4: profile groups (identical coil currents, different profile draws)
RNG_BASE = 100000               # per-case seed = RNG_BASE + candidate id (the 500 / 5000 sets: 100000)
RECIPE = "v1"
HALF = 4                        # frames each side of the centre
WINDOW_MS = 100.0
SYN_DT_CASE = 30.0              # ms: Thomson slices synthesised from a solved frame up to this far away
IP_MIN = 150.0
# recipe v5: long cases, sampled by shot phase and vertical-control activity
CASE_MS = 0.0                   # ms: case window length (0 = the 9-frame cases of v1-v4)
MARGIN_MS = 100.0               # ms: inputs stored beyond the first and last labelled frame
MIN_SPAN_MS = 0.0               # ms: labelled frames must span at least this long for a case to count
PHASE_SHARE = (0.25, 0.55, 0.20)   # share of case centres in ramp-up / flat-top / ramp-down (first / last 20 % of the
                                   # time with |Ip| >= 150 kA)
PHASES = ("rampup", "flattop", "rampdown")
ACT_Q = 0.8                     # high vertical-control activity: 5 ms |change of P6u - P6l| above this real-frame quantile
P4_BINS = []                    # recipe v6: P4-factor bins, one per stratum copy (equal quotas)
LI_TABLE_SHA = None             # recipe v7: sha256 of the li target table
FILE_PREFIX = "mast_synth_g4v5"  # assembled file names (assemble_v5): <FILE_PREFIX>_<candidate id>.parquet
N_FULL = 100                    # v5 held-out: full-shot sequences (ramp-up to ramp-down), stratum 7
N_FULL_REAL = 6                 # realisations per held-out shot in the full-shot candidate list
STRATA_V5 = [f"{p}_{a}" for p in ("rampup", "flattop", "rampdown") for a in ("low", "high")] + ["heldout_uniform", "heldout_fullshot"]


def case_accepted(rec):
    if rec.get("full"):                # full shot: half of the solvable frames, 70 % of the |Ip| >= 150 kA span
        return rec.get("ok", 0) >= 0.5 * rec.get("n_elig", 1e9) and rec.get("span", 0.0) >= 0.7 * rec.get("plasma_span", 1e9)
    return rec.get("ok", 0) >= MIN_FRAMES and rec.get("span", 1e9) >= MIN_SPAN_MS


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
    """Strata = Ip quintile x dIp/dt tercile (x P4-factor bin when P4_BINS is set, recipe v6: equal quotas per bin, so the
    accepted P4 factors cover the range evenly even when acceptance depends on the factor)."""
    shot, t, ip, didt = pool_frames()
    split = np.array(["train" if G.split_of(s) == "train" else "heldout" for s in shot])
    qi = np.quantile(ip, [0.2, 0.4, 0.6, 0.8])
    qd = np.quantile(didt, [1 / 3, 2 / 3])
    stratum = np.digitize(ip, qi) * 3 + np.digitize(didt, qd)
    share = np.bincount(stratum, minlength=15) / len(stratum)
    nb = max(len(P4_BINS), 1)
    share_b = np.repeat(share / nb, nb)                                 # stratum st * nb + b
    rng = np.random.default_rng([G.SEED, 500] if N_CASE["train"] == 500 else [G.SEED, 500, N_CASE["train"]])
    cands, quotas = [], {}
    for sp, n in N_CASE.items():
        q = np.floor(share_b * n).astype(int)
        for i in np.argsort(-(share_b * n - q))[: n - q.sum()]:          # largest remainders
            q[i] += 1
        quotas[sp] = q.tolist()
        for sb in range(15 * nb):
            st, b = divmod(sb, nb)
            idx = np.where((stratum == st) & (split == sp))[0]
            if not len(idx) or not q[sb]:
                continue
            if N_REAL == 1:
                pick = [(i, 0) for i in rng.choice(idx, size=min(len(idx), OVERSAMPLE * q[sb]), replace=False)]
            else:                       # every frame once (random order), then every frame again with new draws, ...
                pick = [(i, rr) for rr in range(N_REAL) for i in rng.permutation(idx)][: OVERSAMPLE * q[sb]]
            for rank, (i, rr) in enumerate(pick):
                c = dict(split=sp, stratum=int(sb), rank=rank, file=str(shot[i]), t=float(t[i]), ip=float(ip[i]),
                         didt=float(didt[i]), real=rr)
                if P4_BINS:
                    c["p4_range"] = list(P4_BINS[b])
                cands.append(c)
    # processing order: by rank, so every stratum fills at the same pace
    cands.sort(key=lambda c: (c["rank"], c["split"] != "train", c["stratum"]))
    for i, c in enumerate(cands):
        c["id"] = i
        c["seed"] = RNG_BASE + i                                         # per-case draws; sets use disjoint seed ranges
    if BLEND_P > 0 or N_GROUPS > 0:
        rb = np.random.default_rng([G.SEED, 777, RNG_BASE])
        # coil-current blends: partner frame from ANOTHER shot of the same split pool and stratum, Ip within 10 %
        for c in cands:
            c["coil_seed"] = c["seed"]
            if rb.random() < BLEND_P:
                pool = np.where((split == c["split"]) & (stratum == c["stratum"] // nb) & (shot != c["file"]) &
                                (np.abs(ip - c["ip"]) <= 0.1 * c["ip"]))[0]
                if len(pool):
                    j = int(rb.choice(pool))
                    c["blend"] = [str(shot[j]), float(t[j]), float(rb.uniform(0.25, 0.75))]
        # profile groups: GROUP_SIZE cases with identical coil currents (same frame, partner and coil draws) and
        # different profile draws; built from early-ranked training candidates so they are solved first
        early = [k for k, c in enumerate(cands) if c["split"] == "train" and c["rank"] < 8]
        for g, k in enumerate(rb.choice(early, size=min(N_GROUPS, len(early)), replace=False)):
            cands[k]["group"] = g
            for _ in range(GROUP_SIZE - 1):
                c = dict(cands[k])
                c["id"] = len(cands)
                c["seed"] = RNG_BASE + c["id"]
                cands.append(c)
        cands.sort(key=lambda c: (c["rank"], c["split"] != "train", c["stratum"], c["id"]))
    os.makedirs(OUT, exist_ok=True)
    strata = [f"ip{st // 3}_didt{st % 3}" + (f"_p4bin{b}" if P4_BINS else "") for st in range(15) for b in range(nb)]
    json.dump(dict(quotas=quotas, strata=strata, p4_bins=[list(x) for x in P4_BINS], ip_edges=qi.tolist(), didt_edges=qd.tolist(),
                   share=share.tolist(), n_real_frames=int(len(t)), candidates=cands), open(os.path.join(OUT, "candidates.json"), "w"))
    print(f"{len(cands)} candidates; real frames {len(t)}; quotas {quotas}")


def pool_frames_v5():
    """Real frames with |Ip| >= IP_MIN from the input pool: shot, t, |Ip|, P6 activity (largest |change of P6u - P6l|
    to a neighbouring EFIT frame 5 ms away, kA), phase index and the shot's |Ip| >= IP_MIN span (t0, t1)."""
    p = np.load(POOL_PATH)
    cols = list(p["columns"])
    shot, t, V = p["shot"].astype(str), p["t"].astype(float), p["values"]
    o = np.lexsort((t, shot))
    shot, t, V = shot[o], t[o], V[o]
    ip = np.abs(V[:, cols.index("magnetics_plasma_current")])
    d6 = V[:, cols.index("magnetics_p6u_current")] - V[:, cols.index("magnetics_p6l_current")]
    act, phase = np.zeros(len(t)), np.full(len(t), -1)
    t0s, t1s = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for s in np.unique(shot):
        m = np.where(shot == s)[0]
        k = m[ip[m] >= IP_MIN]
        if len(k) < 2:
            continue
        t0, t1 = t[k[0]], t[k[-1]]
        span = t1 - t0
        tm = t[m]
        phase[m] = np.where(tm < t0 + 0.2 * span, 0, np.where(tm > t1 - 0.2 * span, 2, 1))
        dd = np.where(np.diff(tm) <= 6.0, np.abs(np.diff(d6[m])), 0.0)
        a = np.zeros(len(m))
        a[1:] = dd
        a[:-1] = np.maximum(a[:-1], dd)
        act[m], t0s[m], t1s[m] = a, t0, t1
    keep = (ip >= IP_MIN) & (phase >= 0)
    return shot[keep], t[keep], ip[keep], act[keep], phase[keep], t0s[keep], t1s[keep]


def build_candidates_v5():
    """Recipe v5 candidates. Strata = phase (ramp-up / flat-top / ramp-down) x P6 activity (high = above the ACT_Q real
    quantile); quota shares PHASE_SHARE, and within a phase twice the real share of high-activity frames (0.3-0.6).
    Each candidate has a CASE_MS window inside its shot's |Ip| >= IP_MIN span containing the centre frame. Blends (BLEND_P,
    low-activity strata only, so the real P6 activity is not diluted): partner of the same split and stratum, Ip within
    10 %, whose own span covers the aligned window."""
    shot, t, ip, act, phase, t0s, t1s = pool_frames_v5()
    split = np.array(["train" if G.split_of(s) == "train" else "heldout" for s in shot])
    thr = float(np.quantile(act, ACT_Q))
    high = (act > thr).astype(int)
    stratum = phase * 2 + high
    real_share = np.bincount(stratum, minlength=6) / len(stratum)
    share = np.zeros(6)
    for ph in range(3):
        hs = float(np.clip(2.0 * high[phase == ph].mean(), 0.3, 0.6))
        share[2 * ph], share[2 * ph + 1] = PHASE_SHARE[ph] * (1 - hs), PHASE_SHARE[ph] * hs
    ws = np.clip(t - CASE_MS / 2, t0s, np.maximum(t0s, t1s - CASE_MS))
    rng = np.random.default_rng([G.SEED, 505, N_CASE["train"], RNG_BASE])
    cands = []
    # train: phase x activity strata; held-out (ZARA_NEXT_DATA 5a): centres uniform over the |Ip| >= 150 kA time of the
    # held-out shots (stratum 6), plus N_FULL full-shot sequences (stratum 7)
    qt = np.floor(share * N_CASE["train"]).astype(int)
    for i in np.argsort(-(share * N_CASE["train"] - qt))[: N_CASE["train"] - qt.sum()]:
        qt[i] += 1
    quotas = {"train": qt.tolist() + [0, 0], "heldout": [0] * 6 + [N_CASE["heldout"], N_FULL]}
    for sp in N_CASE:
        for st in range(7):
            q = quotas[sp][st]
            idx = np.where(((stratum == st) if st < 6 else np.ones(len(t), bool)) & (split == sp))[0]
            if not len(idx) or not q:
                continue
            pick = [(i, rr) for rr in range(N_REAL) for i in rng.permutation(idx)][: OVERSAMPLE * q]
            for rank, (i, rr) in enumerate(pick):
                cands.append(dict(split=sp, stratum=int(st), phase=PHASES[phase[i]], p6_act=float(act[i]), rank=rank,
                                  file=str(shot[i]), t=float(t[i]), ip=float(ip[i]), real=rr, ws=float(ws[i]),
                                  we=float(ws[i] + CASE_MS), span=[float(t0s[i]), float(t1s[i])]))
    hshots = sorted({s for s, sp in zip(shot, split) if sp == "heldout"})
    pick = [(s, rr) for rr in range(N_FULL_REAL) for s in rng.permutation(hshots)]
    for rank, (s, rr) in enumerate(pick):
        m = np.where(shot == s)[0]
        i = m[int(np.argmin(np.abs(t[m] - 0.5 * (t0s[m[0]] + t1s[m[0]]))))]       # start at mid-shot
        cands.append(dict(split="heldout", stratum=7, phase="fullshot", p6_act=float(act[i]), rank=rank, file=str(s), t=float(t[i]),
                          ip=float(ip[i]), real=rr, ws=float(t0s[i]), we=float(t1s[i]), span=[float(t0s[i]), float(t1s[i])], full=True))
    cands.sort(key=lambda c: (c["rank"], c["split"] != "train", c["stratum"]))
    for i, c in enumerate(cands):
        c["id"] = i
        c["seed"] = RNG_BASE + i
    rb = np.random.default_rng([G.SEED, 778, RNG_BASE])
    for c in cands:
        c["coil_seed"] = c["seed"]
        if c["stratum"] in (0, 2, 4, 6) and rb.random() < BLEND_P:
            lo_off, hi_off = c["t"] - c["ws"], c["we"] - c["t"]
            pool = np.where((split == c["split"]) & ((stratum == c["stratum"]) if c["stratum"] < 6 else True) & (shot != c["file"]) &
                            (np.abs(ip - c["ip"]) <= 0.1 * c["ip"]) & (t - lo_off >= t0s - 10.0) & (t + hi_off <= t1s + 10.0))[0]
            if len(pool):
                j = int(rb.choice(pool))
                c["blend"] = [str(shot[j]), float(t[j]), float(rb.uniform(0.25, 0.75))]
    early = [k for k, c in enumerate(cands) if c["split"] == "train" and c["rank"] < 8]
    for g, k in enumerate(rb.choice(early, size=min(N_GROUPS, len(early)), replace=False)):
        cands[k]["group"] = g
        for _ in range(GROUP_SIZE - 1):
            c = dict(cands[k])
            c["id"] = len(cands)
            c["seed"] = RNG_BASE + c["id"]
            cands.append(c)
    cands.sort(key=lambda c: (c["rank"], c["split"] != "train", c["stratum"], c["id"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(quotas=quotas, strata=STRATA_V5,
                   act_threshold_kA=thr, share=share.tolist(), real_share=real_share.tolist(), n_real_frames=int(len(t)),
                   case_ms=CASE_MS, candidates=cands), open(os.path.join(OUT, "candidates.json"), "w"))
    print(f"{len(cands)} candidates; real frames {len(t)}; P6 activity threshold {thr:.3f} kA; real share {real_share.round(3).tolist()}; "
          f"target share {share.round(3).tolist()}; quotas {quotas}; blended {sum('blend' in c for c in cands)}")


def shot_phase(t, span):
    t0, t1 = span
    return PHASES[0] if t < t0 + 0.2 * (t1 - t0) else (PHASES[2] if t > t1 - 0.2 * (t1 - t0) else PHASES[1])


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
            n += case_accepted(json.load(open(f)))
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
    rng_tag = c.get("seed", 100000 + c["id"])                        # per-case draws (older sets: 100000 + id)
    if CASE_MS:                        # v5: every EFIT frame of the case window, started at the centre frame (as v4;
        te = d["t_efit"]               # starts at the highest Ip of a window rarely converge)
        idx = np.where((te >= c["ws"] - 0.1) & (te <= c["we"] + 0.1))[0]
        lo, hi = int(idx[0]), int(idx[-1])
        starts = [k for k in (kc, kc - 1, kc + 1, kc - 2, kc + 2) if lo <= k <= hi][:3]
        blend = (c["blend"][0], c["blend"][1], c["blend"][2], c["t"]) if c.get("blend") else None
        mcf = G.MAX_CONSEC_FAIL
        if c.get("full"):
            G.MAX_CONSEC_FAIL = 12
        try:
            d, P, sig, frames = G.solve_shot(M, c["file"], rng_tag, hang_frames=hang, beat=beat, verbose=verbose, lib=lib,
                                             frame_range=(lo, hi), start_cands=starts, coil_seed=c.get("coil_seed"), blend=blend,
                                             window=(c["ws"], c["we"]))
        finally:
            G.MAX_CONSEC_FAIL = mcf
        for f in frames:
            if f["status"] != "outside":
                f["shot_phase"] = shot_phase(f["t"], c["span"])
        return d, P, sig, frames, kc
    lo, hi = max(0, kc - HALF), min(len(d["t_efit"]) - 1, kc + HALF)
    te = d["t_efit"]                                                  # sparse EFIT times: stay inside the stored input window
    while lo < kc and te[kc] - te[lo] > WINDOW_MS - 5.0:
        lo += 1
    while hi > kc and te[hi] - te[kc] > WINDOW_MS - 5.0:
        hi -= 1
    starts = [kc] + [k for k in (kc - 1, kc + 1, kc - 2, kc + 2) if lo <= k <= hi]
    blend = (c["blend"][0], c["blend"][1], c["blend"][2], c["t"]) if c.get("blend") else None
    d, P, sig, frames = G.solve_shot(M, c["file"], rng_tag, hang_frames=hang, beat=beat, verbose=verbose, lib=lib,
                                     frame_range=(lo, hi), start_cands=starts[:3], coil_seed=c.get("coil_seed"), blend=blend,
                                     p4_range=c.get("p4_range"))
    return d, P, sig, frames, kc


def write_case(path, d, P, sig, frames, th, c, kc):
    """Release layout: one file per case; inputs cropped to +-WINDOW_MS around the centre frame."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tc = float(d["t_efit"][kc])
    tm = d["tm"]
    ok = [f for f in frames if f["status"] == "ok"]
    t_ok = np.array([f["t"] for f in ok])
    if CASE_MS:                        # v5: inputs MARGIN_MS beyond the first and last labelled frame
        w_lo, w_hi = float(t_ok.min()) - MARGIN_MS, float(t_ok.max()) + MARGIN_MS
    else:
        w_lo, w_hi = tc - WINDOW_MS, tc + WINDOW_MS
    w = (tm >= w_lo) & (tm <= w_hi)
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
        m = (tt >= t_ok.min() - SYN_DT_CASE) & (tt <= t_ok.max() + SYN_DT_CASE) & (tt >= w_lo) & (tt <= w_hi)
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
                w_max=G.W_MAX, r_jitter=G.R_JITTER, thomson_var=P.get("tv"), limited_ok=G.LIMITED_OK, jit_mode=G.JIT_MODE,
                bz_mode=G.BZ_MODE, profile_mode=G.PROFILE_MODE, tite_d=P.get("tite_d", 0.0), edge_bump=P.get("eb", 0.0),
                edge_bump_start=P.get("ex1", 0.8))
    if RECIPE in ("v6", "v7"):
        meta.update(p4_range=c.get("p4_range"), bz0_A=P.get("bz0"), bz0_mT=P.get("bz0", 0.0) * G.BZ_T_PER_A * 1e3, thomson_v6=P.get("t6"),
                    bz_range_A=list(G.BZ_RANGE), bz_range_mT=[x * G.BZ_T_PER_A * 1e3 for x in G.BZ_RANGE], tf_jit=G.TF_JIT,
                    pair_ul=G.PAIR_UL, p2_range=list(G.P2_RANGE), p5_range=list(G.P5_RANGE), limited_rule=G.LIMITED_RULE,
                    boundary_definition="q95 and every boundary label on the plasma's own last closed surface: the surface "
                    "touching the wall/limiter when limited, the separatrix when diverted")
    if RECIPE == "v7":
        meta.update(li_table=G.LI_TABLE, li_table_sha256=LI_TABLE_SHA, li_res_sd=G.LI_RES_SD, li_e=P.get("li_e"),
                    li_clip=list(G.LI_CLIP), li_tol=G.LI_TOL, pk_range=list(G.PK_RANGE), rounds=G.ROUNDS,
                    li_rule="li target per frame = li*(inputs) smoothed over +-2 EFIT frames + one N(0, li_res_sd) residual "
                    "per case; li* from a model fitted on DIII-D training labels (Chris, tune2/cond_model.pkl), applied "
                    "to real MAST input features; the current shape h -> h^pk is solved per frame to meet it")
    if CASE_MS:
        meta.update(case_ms=CASE_MS, input_window=[w_lo, w_hi], te_alpha=P.get("te_alpha"), drift=P.get("drift"),
                    limited_rule=G.LIMITED_RULE, p4_range=list(G.P4_RANGE), div23_range=list(G.DIV23_RANGE), neo_s=P.get("neo_s"), ni_an_lo=G.NI_AN_LO, te_alpha_range=list(G.TE_ALPHA_RANGE),
                    bz_range_A=list(G.BZ_RANGE), bz_cap_mT=G.BZ_RANGE[1] * G.BZ_T_PER_A * 1e3, pressure_raise_max=float(np.exp(G.LMS_MAX)),
                    z_jump=G.Z_JUMP, boundary_definition="q95 and every boundary label on the plasma's own last closed surface: "
                    "the surface touching the wall/limiter when limited, the separatrix when diverted")
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
            t_ok = [f["t"] for f in frames if f["status"] == "ok"]
            rec = dict(id=c["id"], ok=nok, span=float(max(t_ok) - min(t_ok)) if t_ok else 0.0, centre_ok=frames[kc]["status"] == "ok",
                       status=st, wall=time.time() - tc, node=me)
            if c.get("full"):
                rec.update(full=True, n_elig=sum(f["status"] not in ("outside", "low_ip") for f in frames),
                           plasma_span=float(c["span"][1] - c["span"][0]))
            json.dump(rec, open(done_path(c), "w"))
            if case_accepted(rec):
                open(acc_path(c), "w").close()
            PR.add(c, case_accepted(rec))
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
                rec = json.load(open(os.path.join(done_dir, name)))
            except Exception:  # noqa: BLE001
                continue
            if case_accepted(rec):
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


def read_done(c):
    """Outcome record of a candidate; {} if missing or unreadable (a worker stopped while writing it)."""
    try:
        return json.load(open(done_path(c)))
    except (OSError, ValueError):
        return {}


def assemble_v5(frac=1.0, tag=None):
    """Recipe v5. Per stratum and split: the cases already released (released.json) plus the next accepted candidates by
    rank, up to round(frac x quota). Files are named by candidate id, so a case keeps its name in every release.
    tag set: a partial release; only the cases not released before are written, to <SET_NAME>_<tag>_{train,heldout}/,
    and released.json grows. No tag: the full set, every chosen case, in <SET_NAME>_{train,heldout}/."""
    C = load_cands()
    rel_path = os.path.join(OUT, "released.json")
    released = set(json.load(open(rel_path))) if os.path.exists(rel_path) else set()
    out, chosen_all = {}, []
    for sp in N_CASE:
        dst = os.path.join(OUT, f"{SET_NAME}_{tag}_{sp}" if tag else f"{SET_NAME}_{sp}")
        os.makedirs(dst, exist_ok=True)
        for old in glob.glob(os.path.join(dst, "*.parquet")) + glob.glob(os.path.join(dst + "_fullshot", "*.parquet")):
            os.remove(old)
        chosen = []
        for st in range(len(C["quotas"][sp])):
            q = C["quotas"][sp][st]
            is_full = C.get("strata", [""] * 99)[st] == "heldout_fullshot" if st < len(C.get("strata", [])) else False
            qb = q if (frac >= 1.0 or is_full) else int(round(q * frac))       # full shots: all accepted, in any release
            cs = sorted([c for c in C["candidates"] if c["split"] == sp and c["stratum"] == st], key=lambda c: c["rank"])
            got = [c for c in cs if c["id"] in released]
            for c in cs:
                if len(got) >= qb:
                    break
                if (c["id"] not in released and os.path.exists(done_path(c)) and case_accepted(read_done(c))
                        and has_thomson(case_path(c))):
                    got.append(c)
            chosen += got
            out[f"{sp}_s{st:02d}"] = (len(got), qb)
        new = [c for c in chosen if c["id"] not in released] if tag else chosen
        dst_full = dst + "_fullshot"
        for c in sorted(new, key=lambda c: c["id"]):
            if c.get("full"):
                os.makedirs(dst_full, exist_ok=True)
            pad_lcfs(case_path(c), os.path.join(dst_full if c.get("full") else dst, f"{FILE_PREFIX}_{c['id']:06d}.parquet"))
        chosen_all += chosen
        print(f"{sp}: {len(chosen)} cases chosen, {len(new)} written -> {dst}")
    short = {k: v for k, v in out.items() if v[0] < v[1]}
    print("strata below target:", short or "none")
    if tag:
        json.dump(sorted(released | {c["id"] for c in chosen_all}), open(rel_path, "w"))
    json.dump(dict(frac=frac, tag=tag, strata=out), open(os.path.join(OUT, f"assembly_{tag or 'full'}.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-candidates", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0, help="v5 assembly: share of every quota to release")
    ap.add_argument("--tag", default=None, help="v5 assembly: partial-release tag (e.g. b1)")
    ap.add_argument("--bz-cap-mT", type=float, default=9.0, help="v5: largest hidden field (mT)")
    ap.add_argument("--p4-range", type=float, nargs=2, default=[0.6, 1.4], help="v5: P4 factor range (P5 compensates)")
    ap.add_argument("--div23-range", type=float, nargs=2, default=[0.85, 1.15], help="v5: P2 / P3 factor range")
    ap.add_argument("--pressure-raise-max", type=float, default=1.75, help="v5: largest pressure factor at the field cap")
    ap.add_argument("--extend", nargs=3, default=None, metavar=("STRATUM", "SPLIT", "N"))
    ap.add_argument("--node-tag", default=socket.gethostname())
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--stall", type=float, default=150.0)
    ap.add_argument("--out", default=None, help="set directory (default the 500-case set)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-heldout", type=int, default=None)
    ap.add_argument("--name", default=None, help="assembled folder prefix")
    ap.add_argument("--li-table", default="/fsx/zara.ercan/more_data/m2/gen4_set1500v7/li_targets.npz", help="v7: li_table.py output")
    ap.add_argument("--bz-lo-mT", type=float, default=None, help="v6/v7: smallest hidden field (mT; v6 0, v7 -10)")
    ap.add_argument("--bz-start-mT", type=float, nargs=2, default=None, help="v7: per-case starting hidden field range (mT)")
    ap.add_argument("--recipe", default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
                    help="v2 = FEEDBACK_FOR_ZARA_TRAINING.md (2026-10-08 01:13): current shape independent of Thomson, "
                         "random Te/ne split at fixed pressure, no extra radial shift, >= 3 labelled frames per case; "
                         "v3 = v2 + its 2026-10-11 requests: centre-column-limited plasmas where the real Thomson is hot at the "
                         "innermost channels, independent upper/lower coil jitter with a wider divertor range, synthetic Thomson "
                         "structure independent of the equilibrium (pedestal, hollow core, tilt, mis-calibrated block, lag)")
    ap.add_argument("--n-real", type=int, default=1, help="realisations per real frame in the candidate list")
    ap.add_argument("--rng-base", type=int, default=100000, help="per-case seed offset (use a new range for every new set)")
    a = ap.parse_args()
    global OUT, SET_NAME, MIN_FRAMES, OVERSAMPLE, RECIPE, N_REAL, RNG_BASE
    RECIPE = a.recipe
    N_REAL, RNG_BASE = a.n_real, a.rng_base
    global BLEND_P, N_GROUPS, CASE_MS, MIN_SPAN_MS, P4_BINS, FILE_PREFIX
    if a.recipe in ("v2", "v3", "v4", "v5", "v6", "v7"):
        G.W_MAX, G.TN_SPLIT, G.R_JITTER, G.SWEEP_BUDGET = 0.0, 0.6, 0.0, 14
        MIN_FRAMES, OVERSAMPLE = 3, 16
    if a.recipe == "v5":               # ZARA_NEXT_DATA_2026-10-14.md / FEEDBACK_FOR_ZARA_2026-10-14_0130.md
        G.LIMITED_OK, G.JIT_MODE, G.THOMSON_VAR, G.LIMITED_RULE = True, "shape", True, "not_cold"
        G.P4_RANGE, G.DIV23_RANGE = tuple(a.p4_range), tuple(a.div23_range)
        G.BZ_MODE, G.PROFILE_MODE, G.PRESSURE_DRIFT = True, "te", True
        G.BZ_RANGE = (G.BZ_RANGE[0], a.bz_cap_mT * 1e-3 / G.BZ_T_PER_A)
        G.LMS_MAX, G.Z_JUMP, G.MAX_CONSEC_FAIL = float(np.log(a.pressure_raise_max)), 0.05, 8
        G.TE_EDGE_BUMP, G.NI_AN_LO, G.TE_ALPHA_RANGE, G.NEO_S_MAX = False, 0.7, (1.2, 1.8), 0.5
        BLEND_P, N_GROUPS = 0.7, 80
        MIN_FRAMES, MIN_SPAN_MS, OVERSAMPLE, CASE_MS = 20, 100.0, 24, 200.0   # acceptance 5-20 % per stratum in the smoke test
        G.make_bz_machine()
    if a.recipe == "v3":
        G.LIMITED_OK, G.JIT_MODE, G.THOMSON_VAR = True, "updown", True
    if a.recipe == "v4":
        G.LIMITED_OK, G.JIT_MODE, G.THOMSON_VAR = True, "wide", True
        G.BZ_MODE, G.PROFILE_MODE = True, "wide"
        BLEND_P, N_GROUPS = 0.7, 80
        G.make_bz_machine()
    if a.recipe in ("v6", "v7"):     # ZARA_NEXT_DATA_2026-10-15.md: v4 + coil inputs like real MAST + hidden field 0-20 mT
        G.LIMITED_OK, G.JIT_MODE, G.THOMSON_VAR = True, "pairs", True
        G.BZ_MODE, G.PROFILE_MODE = True, "wide"
        G.TF_JIT, G.THOMSON_V6 = 0.003, True
        G.BZ_RANGE = ((a.bz_lo_mT if a.bz_lo_mT is not None else (-10.0 if a.recipe == "v7" else 0.0)) * 1e-3 / G.BZ_T_PER_A,
                      20e-3 / G.BZ_T_PER_A)
        G.BZ_START_RANGE = G.BZ_RANGE
        P4_BINS = [(0.85, 0.967), (0.967, 1.083), (1.083, 1.20)]
        BLEND_P, N_GROUPS = 0.7, 24
        OVERSAMPLE = 24                # some strata accept < 7 % (v4: ramp-up strata 6-10 %)
        FILE_PREFIX = "mast_synth_g4v6"
        G.make_bz_machine()
    if a.recipe == "v7":               # FOR_ZARA_TEAM_BLEND_2026-10-16.md section 3: v6 + per-frame li from the DIII-D-taught model
        global LI_TABLE_SHA
        G.LI_TABLE = a.li_table
        G.ROUNDS, G.FRAME_BUDGET, G.SWEEP_BUDGET = 14, 40, 26
        bs = a.bz_start_mT if a.bz_start_mT is not None else (-5.0, 15.0)    # centred on the field the higher li leaves
        G.BZ_START_RANGE = (bs[0] * 1e-3 / G.BZ_T_PER_A, bs[1] * 1e-3 / G.BZ_T_PER_A)
        LI_TABLE_SHA = hashlib.sha256(open(a.li_table, "rb").read()).hexdigest()
        FILE_PREFIX = "mast_synth_g4v7"
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
        build_candidates_v5() if CASE_MS else build_candidates()
    elif a.run:
        run(a)
    elif a.assemble:
        assemble_v5(a.frac, a.tag) if (CASE_MS or RECIPE in ("v6", "v7")) else assemble()


if __name__ == "__main__":
    main()
