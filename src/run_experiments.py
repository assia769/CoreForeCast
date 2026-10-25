"""Steps 4-6 - Rolling-origin temporal validation: baselines vs XGBoost, tracked in MLflow."""
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from src.data import classify
from src.baselines import BASELINES
from src.features import make_features, FEATURES
from src.metrics import wape, bias

N_TEST = 12   # last 12 months = test; each month forecast 1 step ahead (rolling origin)

def main():
    df = pd.read_csv("data/demand.csv", parse_dates=["ds"])
    dates = sorted(df.ds.unique())
    test_dates = dates[-N_TEST:]

    # Segment each SKU using TRAIN period only (no leakage)
    train_only = df[df.ds < test_dates[0]]
    seg = train_only.groupby("unique_id")["y"].apply(classify).rename("segment")

    feat = make_features(df).merge(seg, on="unique_id")
    series = {k: v.sort_values("ds")["y"].values for k, v in df.groupby("unique_id")}
    idx = {d: i for i, d in enumerate(dates)}

    records = []
    for d in test_dates:
        t = idx[d]
        # --- baselines: history strictly before month d
        for sku, y in series.items():
            hist = y[:t]
            for name, fn in BASELINES.items():
                records.append((sku, d, name, fn(hist), y[t]))
        # --- XGBoost: train on rows with ds < d, predict rows with ds == d
        tr = feat[(feat.ds < d)].dropna(subset=["lag_12"])
        te = feat[feat.ds == d]
        model = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3,
                                 n_estimators=200, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, random_state=0)
        model.fit(tr[FEATURES], tr["y"])
        pred = np.clip(model.predict(te[FEATURES]), 0, None)
        for sku, p, a in zip(te.unique_id, pred, te.y):
            records.append((sku, d, "xgboost", float(p), a))

    res = pd.DataFrame(records, columns=["unique_id", "ds", "model", "yhat", "y"]).merge(seg, on="unique_id")
    res.to_csv("results/predictions.csv", index=False)

    mlflow.set_experiment("coreforecast")
    rows = []
    for (m, s), g in res.groupby(["model", "segment"]):
        rows.append((m, s, wape(g.y, g.yhat), bias(g.y, g.yhat)))
    table = pd.DataFrame(rows, columns=["model", "segment", "wape", "bias"])

    for m, g in res.groupby("model"):
        with mlflow.start_run(run_name=m):
            mlflow.log_param("n_test_months", N_TEST)
            mlflow.log_metric("wape_all", wape(g.y, g.yhat))
            mlflow.log_metric("bias_all", bias(g.y, g.yhat))
            for s, gs in g.groupby("segment"):
                mlflow.log_metric(f"wape_{s}", wape(gs.y, gs.yhat))
                mlflow.log_metric(f"bias_{s}", bias(gs.y, gs.yhat))

    w = table.pivot(index="segment", columns="model", values="wape").round(3)
    w["BEST"] = w.idxmin(axis=1)
    w.to_csv("results/wape_by_segment.csv")
    print("WAPE by segment (lower is better):\n", w.to_string())
    print("\nBias by segment (+ over / - under forecast):\n",
          table.pivot(index="segment", columns="model", values="bias").round(2).to_string())
    print("\nOverall WAPE:\n", res.groupby("model").apply(lambda g: wape(g.y, g.yhat)).round(3).sort_values().to_string())

if __name__ == "__main__":
    main()
