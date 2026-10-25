# CoreForecast - Intermittent demand forecasting for remanufactured parts (cores)

## Problem
Cores (used parts collected for remanufacturing) arrive irregularly: many months with zero, then spikes.
Under-forecasting means missing cores (lost production); over-forecasting means excess stock.

## Data
Simulated (see `src/data.py`): 300 SKUs x 60 months, four demand patterns. **Not real  entreprise data.**

## Approach
1. Segment each SKU with ADI / CV2 (smooth, erratic, intermittent, lumpy), using training period only.
2. Baselines: naive, seasonal naive, moving average, Croston, SBA, TSB.
3. ML: global XGBoost (Tweedie objective) with lag, rolling and intermittency features (past information only).
4. Rolling-origin temporal validation on the last 12 months (no random split, no leakage).
5. Metrics: WAPE and bias (MAPE rejected because of zero values).
6. All runs tracked with MLflow.
7. LLM agent explains forecast gaps from computed numbers only.

## Results
(Paste your WAPE-by-segment table from `results/wape_by_segment.csv` here.)

## Limitations and next steps
- Simulated data; real data would need cleaning and business validation.
- One-step-ahead forecast; extend to multi-horizon.
- Production: scheduled job, model registry, drift monitoring.

## Run
    pip install -r requirements.txt
    python -m src.data
    python -m src.run_experiments
    mlflow ui
    python -m src.agent
# CoreForeCast
