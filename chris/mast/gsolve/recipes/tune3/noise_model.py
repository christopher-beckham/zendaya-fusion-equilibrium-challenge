"""Thomson channel noise of the tuned recipe, shared by the generator (gen_param.py), the critic's per-slice emulation
(tune_lib.py) and the per-slice converter (convert_perslice.py). Same random-number order as the generator."""
import numpy as np


def apply(cT, cN, relT, relN, kn, inside, rng):
    """Noise-free remapped Te (keV) / ne (1e19 m^-3) on the channels -> noisy copies (NaN channels stay NaN).
    relT / relN: the origin frame's own relative scatter (noise_model rule); inside: channels with rho < 0.9."""
    C = len(cT)
    ns = kn["noise_scale"]
    tT = cT * (1 + ns * relT * rng.standard_normal(C))
    tN = cN * (1 + ns * kn.get("noise_ne", 1.0) * relN * rng.standard_normal(C))
    if kn["out_frac"] > 0:
        hit = rng.random(C) < kn["out_frac"]
        tT = np.where(hit, tT * np.exp(kn["out_sd"] * rng.standard_normal(C)), tT)
        hit = rng.random(C) < kn["out_frac"]
        tN = np.where(hit, tN * np.exp(kn["out_sd"] * rng.standard_normal(C)), tN)
    if kn["drop_frac"] > 0:
        hit = inside & (rng.random(C) < kn["drop_frac"])
        tT = np.where(hit, rng.uniform(0.005, 0.02, C), tT)
    return np.clip(tT, 0.001, None), np.clip(tN, 0.01, None)
