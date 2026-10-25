"""Step 1 - Simulated intermittent demand for remanufacturable parts (cores).
Four demand patterns (Syntetos-Boylan): smooth, erratic, intermittent, lumpy."""
import numpy as np
import pandas as pd

PATTERNS = {            # (prob of demand, mean size, size variability)
    "smooth":       (0.90, 20, 0.15),
    "erratic":      (0.90, 20, 0.90),
    "intermittent": (0.25, 6,  0.20),
    "lumpy":        (0.25, 6,  1.10),
}

def simulate(n_skus=300, n_months=60, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    names = list(PATTERNS)
    for i in range(n_skus):
        kind = names[i % 4]
        p, mu, cv = PATTERNS[kind]
        season_amp = rng.uniform(0, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        for t, d in enumerate(dates):
            season = 1 + season_amp * np.sin(2 * np.pi * t / 12 + phase)
            if rng.random() < min(p * season, 1):
                sigma = np.sqrt(np.log(1 + cv ** 2))
                y = rng.lognormal(np.log(mu) - sigma ** 2 / 2, sigma)
                y = max(1, int(round(y)))
            else:
                y = 0
            rows.append((f"SKU_{i:03d}", d, y, kind))
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y", "true_pattern"])

def classify(s):
    """ADI / CV2 classification (thresholds 1.32 and 0.49)."""
    nz = s[s > 0]
    if len(nz) < 2:
        return "insufficient"
    adi = len(s) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"

if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/demand.csv", index=False)
    seg = df.groupby("unique_id")["y"].apply(classify)
    print(df.shape, "| zeros: %.0f%%" % (100 * (df.y == 0).mean()))
    print(seg.value_counts())
