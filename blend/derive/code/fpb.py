"""Final-push blend tools (MAST): exact candidate derivation + board-calibrated score estimate. No MAST label is read.

Members are board-scored MAST prediction files. A candidate is a frame-by-frame linear blend with weights per quantity
(maps / q95 / betaN). Its psi, q95 and betaN board R2 follow exactly from the members' board R2 and the pairwise file
differences (Gram method, SS_y from the scored b2 blend). Its seven derived shape scalars are computed here from the
blended maps with the scorer's own code (local_score.build_reference, limiter mask, MAST axis sign) and their board R2 is
estimated by affine projection onto every scored vector (members + the scored blends b2 and t2b), with SS_y chosen in the
PSD-feasible range by leave-one-out on b2 and t2b. D_LCFS: weighted member D minus gamma * 1/2 w'Hw (H = mean pairwise
LCFS Hausdorff / rgeo), gamma fitted on b2 and t2b.
"""
import itertools
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

STARTER = "/fsx/zara.ercan/fusion-equilibrium-challenge-starter"
G4 = "/fsx/zara.ercan/cache/ch2/g4"
TB = f"{STARTER}/my_experiments/ch2/g4/team_blend"
SUB = f"{STARTER}/submission"
CHS = "/fsx/chris.beckham/github/fusion-equilibrium-challenge-starter/submissions"
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]

# name -> (path, psi sign). Chris's files use DIII-D's psi sign: flip before mixing.
FILES = {
    "hyb": (f"{SUB}/nn_prod_v7_ch1_hm4fixhyb/mast_public_test.npz", 1.0),
    "c2": (f"{SUB}/nn_prod_v13_ch1_g4c2/mast_public_test.npz", 1.0),
    "r1": (f"{SUB}/nn_prod_v13_ch1_g4r1/mast_public_test.npz", 1.0),
    "u1x": (f"{SUB}/nn_prod_v7_ch1_ensu1x_nttvqr_pp2/mast_public_test.npz", 1.0),
    "u1xq": (f"{SUB}/nn_prod_v23_ensu1x_nttvq/mast_public_test.npz", 1.0),
    "u4pp": (f"{G4}/old_subs/u4pp/mast_public_test.npz", 1.0),
    "u4all": (f"{G4}/old_subs/u4all/mast_public_test.npz", 1.0),
    "Z3": (f"{G4}/chris_subs/Z3/mast_public_test.npz", -1.0),
    "Z8": (f"{G4}/chris_subs/Z8/mast_public_test.npz", -1.0),
    "Z9qb": (f"{CHS}/mastZ9qb-mm-7xV2recipe-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z10qb": (f"{CHS}/mastZ10-mm-v5mix-E1E2qb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z11qb": (f"{CHS}/mastZ11-mm-v5only-E4qb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z14": (f"{CHS}/mastZ14-mm-Z9dataPlusGsolve-x3qb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z15": (f"{CHS}/mastZ15-mm-v6mix-x4qb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z17": (f"{CHS}/mastZ17-mm-C3-condgsolveonlyqb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
    "Z18": (f"{CHS}/mastZ18-mm-Z9dataPlusCond-x3qb-q95cc0932-bnV2-limiter-c1/mast_public_test.npz", -1.0),
}
# Scored blends (already in MAST sign), used only as calibration points.
BLENDS = {"b2": f"{G4}/blend/b2.npz", "t2b": f"{G4}/blend/t2b.npz", "t3c": f"{G4}/blend/t3c.npz"}
WBLEND = {
    "b2": {"maps": {"hyb": .3, "c2": .1, "r1": .3, "u1x": .3}, "q95": {"hyb": .8, "r1": .1, "u1x": .1},
           "betaN": {"hyb": .4, "r1": .3, "u1x": .3}},
    "t2b": json.load(open(f"{TB}/wT2b.json")),
    "t3c": json.load(open(f"{TB}/wT3c.json")),
}
# Compliance (user rule, feedback-ch2-zero-efit): hyb's q95 used the EFIT-derived envelope mask; Z14 kept out by choice.
BARRED = {"q95": {"hyb", "Z14", "b2", "t2b", "t3c"}, "maps": {"Z14", "b2", "t2b", "t3c"},
          "betaN": {"Z14", "b2", "t2b", "t3c"}}


def load_board():
    bd = json.load(open(f"{TB}/budget/board_team3_exact.json"))
    out = {}
    for n, e in bd.items():
        if not isinstance(e, dict) or "psi" not in e:
            continue
        out[n] = dict(psi=e["psi"], q95=e.get("q95"), betaN=e.get("betaN"), D=e["D"], cons=e.get("cons"))
    return out


# ----------------------------------------------------------------------------------------------- derivation (workers)
_W = {}


def _init(paths, cands, R, Z, mask):
    _W.update(paths=paths, cands=cands, R=R, Z=Z, mask=mask, maskf=mask.astype(np.float64), npz={})
    sys.path.insert(0, STARTER)
    sys.path.insert(0, os.path.join(STARTER, "fusion_scoring"))
    import local_score as ls
    ls.MACHINE = "MAST"
    _W["ls"] = ls


def _psi(name, i):
    if name not in _W["npz"]:
        _W["npz"][name] = np.load(_W["paths"][name][0])
    return _W["paths"][name][1] * _W["npz"][name][f"shot_{i:04d}_psirz"].astype(np.float64)


def _one(i):
    used = sorted({n for c in _W["cands"] for n in c})
    psi = {n: _psi(n, i) for n in used}
    out = []
    for c in _W["cands"]:
        tot = sum(c.values())
        p = sum((w / tot) * psi[n] for n, w in c.items())
        refs = _W["ls"].build_reference(p, _W["R"], _W["Z"], _W["mask"], _W["maskf"])
        out.append((refs[1].astype(np.float32), refs[3]))
    return i, out


def limiter_mask(R, Z):
    sys.path.insert(0, "/fsx/zara.ercan/more_data/generated/code")
    import mast_machine as mm
    from matplotlib.path import Path as MPath
    RR, ZZ = np.meshgrid(R, Z)
    return MPath(mm.limiter_polygon()).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)


def derive(cands, workers, shots=1206):
    """cands: list of {member: map weight}. Returns array (n_cands, n_frames, 7) of derived scalars."""
    bas = np.load("/fsx/zara.ercan/cache/mast/mast_public_test/basis.npz")
    R, Z = bas["grid_R"].astype(float), bas["grid_Z"].astype(float)
    mask = limiter_mask(R, Z)
    paths = dict(FILES)
    res = {}
    with get_context("fork").Pool(workers, initializer=_init, initargs=(paths, cands, R, Z, mask)) as pool:
        for i, o in pool.imap_unordered(_one, range(shots), chunksize=1):
            res[i] = o
    return np.stack([np.concatenate([res[i][k][0] for i in range(shots)]) for k in range(len(cands))])


# ----------------------------------------------------------------------------------------------- estimator
def gram(E, D):
    return 0.5 * (E[:, None] + E[None, :] - D)


class Est:
    """Board-calibrated estimator. Scored vectors: members (FILES with board detail) + blends b2, t2b."""

    def __init__(self, verbose=True, extra_pairs=(), board_extra=None, extra_blends=None):
        """extra_pairs: dirs written by add_pairs.py (new members); board_extra: json {name: {psi, q95, betaN, D, cons}};
        extra_blends: {name: weights json} of further scored blends (vectors in G4/board_pairs_<name>, board numbers in
        board_extra) used as calibration points."""
        self.bd = load_board()
        if board_extra:
            self.bd.update(json.load(open(board_extra)))
        for B, wf in (extra_blends or {}).items():
            WBLEND[B] = json.load(open(wf))
            for k in BARRED:
                BARRED[k].add(B)
        fr = np.load(f"{G4}/board_pairs_team3/frames.npz")
        sm = json.load(open(f"{G4}/board_pairs_team3/summary.json"))
        self.names = [n for n in sm["files"]]                     # includes b2
        frs = [fr]
        pdist = dict(sm["psi_pair_d"])
        for d in extra_pairs:
            sm2 = json.load(open(f"{d}/summary.json"))
            frs.append(np.load(f"{d}/frames.npz"))
            pdist.update(sm2["psi_pair_d"])
            for n, spec in sm2["files"].items():
                path, sgn = spec.rsplit("@", 1)
                FILES[n] = (path, float(sgn))
                self.names.append(n)
        ix = {n: i for i, n in enumerate(self.names)}
        self.ix = ix
        self.Dpsi = np.zeros((len(self.names),) * 2)
        for k, v in pdist.items():
            p, q = k.split("|")
            if p in ix and q in ix:
                self.Dpsi[ix[p], ix[q]] = self.Dpsi[ix[q], ix[p]] = v
        self.vec = {}
        for n in self.names:
            src = next(f for f in frs if f"{n}__q95" in f.files)
            self.vec[(n, "q95")] = src[f"{n}__q95"].astype(float)
            self.vec[(n, "betaN")] = src[f"{n}__betaN"].astype(float)
            c = src[f"{n}__cons"].astype(float)
            for j, k in enumerate(CONS):
                self.vec[(n, k)] = c[:, j]
        self.blends = [B for B in ["t2b", "t3c"] + list(extra_blends or {})
                       if B in self.bd and os.path.exists(f"{G4}/board_pairs_{B}/frames.npz")]
        for B in self.blends:
            t2 = np.load(f"{G4}/board_pairs_{B}/frames.npz")
            for j, k in enumerate(CONS):
                self.vec[(B, k)] = t2[f"{B}__cons"][:, j].astype(float)
            self.vec[(B, "q95")] = t2[f"{B}__q95"].astype(float)
            self.vec[(B, "betaN")] = t2[f"{B}__betaN"].astype(float)
        self.blends = ["b2"] + self.blends
        self.haus = {}
        for f in frs:
            for key in f.files:
                if key.endswith("__haus"):
                    p, q = key[:-6].split("__vs__")
                    self.haus[(p, q)] = self.haus[(q, p)] = float(np.nanmean(f[key]))
        # SS_y for psi / q95 / betaN, exact from b2
        self.ss = {}
        wb = WBLEND["b2"]
        for key, wk in (("psi", wb["maps"]), ("q95", wb["q95"]), ("betaN", wb["betaN"])):
            names = list(wk)
            w = np.array([wk[n] for n in names])
            D = self.dmat(key, ["b2"] + names)
            wv = np.r_[0.0, w]
            self.ss[key] = 0.5 * (wv @ D @ wv) / (self.r2("b2", key) - sum(wk[n] * self.r2(n, key) for n in names))
        # members whose shape-term R2 were only solved from a scored blend ("pinned") are not calibration points;
        # the scored blends that contain them are
        self.det = [n for n in self.names if n != "b2" and self.bd.get(n, {}).get("cons") and not self.bd[n].get("pinned")]
        self.scored = self.det + self.blends
        self._cons_ss(verbose)
        self._fit_gamma(verbose)

    def r2(self, n, key):
        e = self.bd[n]
        return e[key] if key in ("psi", "q95", "betaN") else e["cons"][key]

    def dmat(self, key, names, extra=None):
        if key == "psi":
            ii = [self.ix[n] for n in names]
            return self.Dpsi[np.ix_(ii, ii)]
        x = [self.vec[(n, key)] for n in names]
        X = np.stack(x, 1)
        ok = np.all(np.isfinite(X), 1)
        X = X[ok]
        G = X.T @ X
        d = np.diag(G)
        return d[:, None] + d[None, :] - 2 * G

    # ---- exact psi / q95 / betaN
    def exact(self, key, w):
        names = [n for n, v in w.items() if v > 0]
        wv = np.array([w[n] for n in names]); wv = wv / wv.sum()
        D = self.dmat(key, names)
        C = gram(np.array([(1 - self.r2(n, key)) * self.ss[key] for n in names]), D)
        return 1 - wv @ C @ wv / self.ss[key]

    def opt_exact(self, key, pool):
        from scipy.optimize import minimize
        names = list(pool)
        D = self.dmat(key, names)
        C = gram(np.array([(1 - self.r2(n, key)) * self.ss[key] for n in names]), D)
        best = None
        for s in range(30):
            w0 = np.random.default_rng(s).dirichlet(np.ones(len(names)))
            r = minimize(lambda w: w @ C @ w, w0, jac=lambda w: 2 * C @ w, bounds=[(0, 1)] * len(names),
                         constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                         options=dict(ftol=1e-15, maxiter=1000))
            if best is None or r.fun < best.fun:
                best = r
        w = {n: float(v) for n, v in zip(names, best.x) if v > 1e-4}
        return w, 1 - best.fun / self.ss[key]

    # ---- consistency
    def _proj(self, key, s, J, ssv):
        X = np.stack([self.vec[(n, key)] for n in J], 1)
        ok = np.all(np.isfinite(X), 1) & np.isfinite(s)
        X, s = X[ok], s[ok]
        k = len(J)
        A = np.zeros((k + 1, k + 1)); A[:k, :k] = 2 * X.T @ X; A[:k, k] = 1; A[k, :k] = 1
        b = np.r_[2 * X.T @ s, 1.0]
        c = np.linalg.lstsq(A, b, rcond=None)[0][:k]
        r = s - X @ c
        a = np.array([1 - self.r2(n, key) for n in J])
        d = np.diag(X.T @ X)
        D = d[:, None] + d[None, :] - 2 * X.T @ X
        E = c @ gram(a * ssv, D) @ c + r @ r
        return 1 - E / ssv, c, float(np.sqrt(r @ r / len(r)))

    def _cons_ss(self, verbose):
        self.css, self.css_rng, self.loo = {}, {}, {}
        for key in CONS:
            J = self.scored
            X = np.stack([self.vec[(n, key)] for n in J], 1)
            ok = np.all(np.isfinite(X), 1)
            X = X[ok]
            d = np.diag(X.T @ X)
            D = d[:, None] + d[None, :] - 2 * X.T @ X
            a = np.array([1 - self.r2(n, key) for n in J])
            spread = np.mean([((X[:, j] - X[:, j].mean()) ** 2).sum() for j in range(len(self.det))])
            grid = np.exp(np.linspace(np.log(0.02), np.log(50), 3000)) * spread
            feas = [s for s in grid if np.linalg.eigvalsh(gram(a * s, D))[0] >= -1e-9 * s]
            lo, hi = min(feas), max(feas)
            cand = np.exp(np.linspace(np.log(lo), np.log(hi), 60))
            errs = []
            for sv in cand:
                e = []
                for B in self.blends:
                    J2 = [n for n in J if n != B]
                    est = self._proj(key, self.vec[(B, key)], J2, sv)[0]
                    e.append(est - self.r2(B, key))
                errs.append(e)
            errs = np.array(errs)
            j = int(np.argmin((errs ** 2).sum(1)))
            self.css[key], self.css_rng[key], self.loo[key] = cand[j], (lo, hi), errs[j]
            mid = np.sqrt(lo * hi)
            jm = int(np.argmin(np.abs(np.log(cand / mid))))
            if verbose:
                print(f"  {key:8s} SS range {lo / mid:.2f}..{hi / mid:.2f} x mid; chosen {cand[j] / mid:.2f} x mid; LOO err "
                      + " ".join(f"{B} {e:+.4f}" for B, e in zip(self.blends, errs[j]))
                      + " (at mid: " + " ".join(f"{e:+.4f}" for e in errs[jm]) + ")", flush=True)

    def cons_est(self, svecs):
        """svecs: (n_frames, 7) derived scalars of a candidate -> dict of estimated board R2 per scalar."""
        return {key: self._proj(key, svecs[:, j].astype(float), self.scored, self.css[key])[0]
                for j, key in enumerate(CONS)}

    def _first_cache(self):
        """Gram matrices of every scored member (first-order consistency, psi), computed once."""
        if getattr(self, "_fc", None) is None:
            names = [n for n in self.names if n != "b2" and self.bd.get(n, {}).get("cons")]
            fc = {"names": names, "ix": {n: i for i, n in enumerate(names)}}
            for key in CONS:
                D = self.dmat(key, names)
                fc[key] = gram(np.array([(1 - self.r2(n, key)) * self.css[key] for n in names]), D)
            Dp = self.dmat("psi", names)
            fc["psi"] = gram(np.array([(1 - self.r2(n, "psi")) * self.ss["psi"] for n in names]), Dp)
            fc["Dl"] = np.array([self.bd[n]["D"] for n in names])
            fc["H"] = np.array([[0 if p == q else self.haus[(p, q)] for q in names] for p in names])
            self._fc = fc
        return self._fc

    def first_vec(self, w):
        """weights dict -> full-pool weight vector (normalised)."""
        fc = self._first_cache()
        v = np.zeros(len(fc["names"]))
        for n, x in w.items():
            if x > 0:
                v[fc["ix"][n]] = x
        return v / v.sum()

    def cons_first(self, w):
        """first-order (no candidate derivation): Gram over members only."""
        fc = self._first_cache()
        wv = w if isinstance(w, np.ndarray) else self.first_vec(w)
        return {key: 1 - wv @ fc[key] @ wv / self.css[key] for key in CONS}

    def fast_parts(self, wv):
        """psi R2 (exact), D model, first-order consistency for a full-pool weight vector."""
        fc = self._first_cache()
        psi = 1 - wv @ fc["psi"] @ wv / self.ss["psi"]
        d = wv @ fc["Dl"] - self.gamma * 0.5 * wv @ fc["H"] @ wv
        cons = np.mean([max(0.0, 1 - wv @ fc[k] @ wv / self.css[k]) for k in CONS])
        return psi, d, cons

    # ---- D_LCFS
    def _dq(self, w):
        names = [n for n, v in w.items() if v > 0]
        wv = np.array([w[n] for n in names]); wv = wv / wv.sum()
        H = np.array([[0 if p == q else self.haus[(p, q)] for q in names] for p in names])
        return wv @ np.array([self.bd[n]["D"] for n in names]), 0.5 * wv @ H @ wv

    def _fit_gamma(self, verbose):
        rows = []
        for B in self.blends:
            lin, quad = self._dq(WBLEND[B]["maps"])
            rows.append((lin - self.bd[B]["D"], quad))
        rows = np.array(rows)
        self.gamma = float((rows[:, 0] * rows[:, 1]).sum() / (rows[:, 1] ** 2).sum())
        if verbose:
            for B, (num, q) in zip(self.blends, rows):
                print(f"  D_LCFS {B}: board {self.bd[B]['D']:.4f}, model {self._dq(WBLEND[B]['maps'])[0] - self.gamma * q:.4f} "
                      f"(gamma {self.gamma:.3f}, single-blend gamma {num / q:.3f})")

    def D(self, w):
        lin, quad = self._dq(w)
        return lin - self.gamma * quad

    def score(self, wmaps, cons, r2q, r2b):
        psi = self.exact("psi", wmaps)
        d = self.D(wmaps)
        c = np.mean([max(0.0, v) for v in cons.values()])
        S = 0.55 * max(0.0, psi) + 0.15 * max(0.0, 0.5 * (r2q + r2b)) + 0.10 * (1 - d) + 0.20 * c
        return dict(S=S, psi=psi, D=d, cons=c, q95=r2q, betaN=r2b, **{f"c_{k}": v for k, v in cons.items()})
