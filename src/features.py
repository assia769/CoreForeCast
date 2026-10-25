"""Features use ONLY past info (shift(1) minimum) -> no data leakage."""
import pandas as pd

def make_features(df):
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    g = df.groupby("unique_id")["y"]
    for k in (1, 2, 3, 6, 12):
        df[f"lag_{k}"] = g.shift(k)
    past = g.shift(1)
    for w in (3, 6, 12):
        df[f"roll_mean_{w}"] = past.groupby(df["unique_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    nz = (past > 0).astype(float).where(past.notna())
    df["nonzero_share_6"] = nz.groupby(df["unique_id"]).transform(
        lambda s: s.rolling(6, min_periods=1).mean())

    def since(s):
        out, c = [], 0
        for v in s.shift(1).fillna(0).values:
            c = 0 if v > 0 else c + 1
            out.append(c)
        return pd.Series(out, index=s.index)
    df["months_since_demand"] = g.transform(since)
    df["month"] = df["ds"].dt.month
    return df

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
            "roll_mean_3", "roll_mean_6", "roll_mean_12",
            "nonzero_share_6", "months_since_demand", "month"]
