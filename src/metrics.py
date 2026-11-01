"""
metrics.py - evaluation functions used by train.py.
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (mg/dL). Large errors count extra."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error (mg/dL): the typical size of an error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke error grid zone of ONE prediction (mg/dL).
    A = clinically accurate, B = benign error, C = overcorrection,
    D = dangerous failure to detect, E = erroneous (treat low as high or vice versa).
    """
    x, y = ref, pred
    if (x <= 70 and y <= 70) or (0.8 * x <= y <= 1.2 * x):
        return "A"
    if (x >= 180 and y <= 70) or (x <= 70 and y >= 180):
        return "E"
    if (70 <= x <= 290 and y >= x + 110) or (130 <= x <= 180 and y <= 7 / 5 * x - 182):
        return "C"
    if (x >= 240 and 70 <= y <= 180) or (x <= 175 / 3 and 70 <= y <= 180) \
            or (175 / 3 <= x <= 70 and y >= 6 / 5 * x):
        return "D"
    return "B"


def clarke_ab(y_true, y_pred) -> float:
    """Share of predictions in Clarke zones A+B (%). Target for a good model: > 99 %."""
    zones = [clarke_zone(a, b) for a, b in zip(y_true, y_pred)]
    return 100 * np.mean([z in ("A", "B") for z in zones])


def clarke_a(y_true, y_pred) -> float:
    """Share of predictions in zone A only (%)."""
    zones = [clarke_zone(a, b) for a, b in zip(y_true, y_pred)]
    return 100 * np.mean([z == "A" for z in zones])