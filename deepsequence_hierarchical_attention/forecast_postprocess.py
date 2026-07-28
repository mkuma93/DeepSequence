"""Post-processing helpers for intermittent demand forecasts."""

from __future__ import annotations

import numpy as np


def round_forecast(yhat, minimum: float = 0.0) -> np.ndarray:
    """
    Round forecasts to the nearest integer (count demand).

    Clips at ``minimum`` (default 0) before rounding so tiny negative
    numerical noise does not become -1.
    """
    y = np.asarray(yhat, dtype=np.float64)
    y = np.maximum(y, minimum)
    return np.rint(y)
