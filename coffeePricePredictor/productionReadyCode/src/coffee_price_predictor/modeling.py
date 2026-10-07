"""Model-building and evaluation utilities for the coffee quality benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical columns."""
    numerical_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    if not numerical_columns:
        raise ValueError("No numeric columns were found in the feature set.")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_models() -> dict[str, Any]:
    """Define the candidate regression models used in the benchmark study."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
    }


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
