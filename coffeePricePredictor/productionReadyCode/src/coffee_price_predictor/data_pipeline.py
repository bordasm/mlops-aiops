"""Utilities for loading and preparing the coffee benchmark dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the benchmark CSV file and validate that it exists."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame,
    target_column: str,
    leakage_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature and target arrays after dropping leakage fields."""
    required_columns = {target_column, *leakage_columns}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    drop_columns = [*leakage_columns, target_column]
    features = dataframe.drop(columns=drop_columns, errors="ignore")
    target = dataframe[target_column]
    return features, target
