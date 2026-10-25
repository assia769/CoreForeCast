"""Statistical baselines. Each takes a 1-D history array, returns a 1-step forecast."""
import numpy as np

def naive(h):            return float(h[-1])
def seasonal_naive(h):   return float(h[-12]) if len(h) >= 12 else float(h[-1])
def moving_avg(h, k=6):  return float(np.mean(h[-k:]))

def croston(h, alpha=0.1, variant="classic"):
    h = np.asarray(h, float)
    nz = np.flatnonzero(h > 0)
    if len(nz) == 0:
        return 0.0
    z = h[nz[0]]                       # demand size
    p = nz[0] + 1.0                    # interval
    q = 1
    for t in range(nz[0] + 1, len(h)):
        if h[t] > 0:
            z += alpha * (h[t] - z)
            p += alpha * (q - p)
            q = 1
        else:
            q += 1
    f = z / p
    return float(f * (1 - alpha / 2)) if variant == "sba" else float(f)

def sba(h):  return croston(h, variant="sba")

def tsb(h, alpha=0.2, beta=0.2):
    h = np.asarray(h, float)
    nz = np.flatnonzero(h > 0)
    if len(nz) == 0:
        return 0.0
    z = h[nz[0]]
    prob = 1.0 / (nz[0] + 1)
    for t in range(nz[0] + 1, len(h)):
        occ = 1.0 if h[t] > 0 else 0.0
        prob += beta * (occ - prob)
        if occ:
            z += alpha * (h[t] - z)
    return float(prob * z)

BASELINES = {"naive": naive, "seasonal_naive": seasonal_naive,
             "moving_avg_6": moving_avg, "croston": croston,
             "sba": sba, "tsb": tsb}
