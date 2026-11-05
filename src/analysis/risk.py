import numpy as np


def maximum_drawdown(profits, initial_capital=1000):
    equity = np.r_[initial_capital, initial_capital + np.cumsum(profits)]
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def calibration(probabilities, outcomes, bins=5):
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    result = []
    edges = np.linspace(0, 1, bins+1)
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if mask.any():
            result.append({'predicted': float(p[mask].mean()), 'observed': float(y[mask].mean()), 'n': int(mask.sum())})
    return result
