import numpy as np

def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.abs(y - yhat).sum() / max(y.sum(), 1e-9)

def bias(y, yhat):
    """>0 over-forecast (excess stock), <0 under-forecast (missing cores)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return (yhat - y).sum() / max(y.sum(), 1e-9)
