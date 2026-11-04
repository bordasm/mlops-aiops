"""Shared regression evaluation without training dependencies."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for a prediction set."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
