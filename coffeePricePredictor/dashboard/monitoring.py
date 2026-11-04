"""Validated feature monitoring and local Evidently report persistence."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.core.report import Snapshot
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset,
    RegressionPreset,
)

TARGET = "coffee_quality_score"
PREDICTION = "predicted_coffee_quality_score"
LEAKAGE_COLUMNS = {"record_id", "lot_id", "quality_grade", "price_usd_per_kg"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class MonitoringResult:
    """Durable report files and the number of evaluated target values."""

    html_path: Path
    json_path: Path
    evaluated_rows: int
    excluded_target_rows: int
    excluded_drift_columns: tuple[str, ...] = ()


def _persist_report(
    snapshot: Snapshot,
    reports_dir: Path,
    kind: str,
    *,
    evaluated_rows: int = 0,
    excluded_target_rows: int = 0,
    excluded_drift_columns: tuple[str, ...] = (),
) -> MonitoringResult:
    """Save a unique report pair including its monitoring limitations."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_" + uuid4().hex[:12] + "_" + kind
    )
    html_path = reports_dir / f"{name}.html"
    json_path = reports_dir / f"{name}.json"
    snapshot.save_html(str(html_path))
    payload = json.loads(snapshot.json())
    payload["coffee_monitoring"] = {
        "evaluated_rows": evaluated_rows,
        "excluded_target_rows": excluded_target_rows,
        "excluded_drift_columns": list(excluded_drift_columns),
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return MonitoringResult(
        html_path, json_path, evaluated_rows, excluded_target_rows,
        excluded_drift_columns,
    )


def read_csv(content: bytes) -> pd.DataFrame:
    """Read bounded UTF-8 CSV data without accepting empty input."""
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds the 20 MB limit.")
    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(f"Invalid UTF-8 CSV: {error}") from error
    if dataframe.empty:
        raise ValueError("The CSV contains no data rows.")
    return dataframe


def feature_columns(reference: pd.DataFrame) -> list[str]:
    """Match the training pipeline's target and leakage exclusions."""
    return [
        column for column in reference.columns
        if column not in LEAKAGE_COLUMNS | {TARGET}
    ]


def prepare_current(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Validate feature types, keeping missing values for quality checks."""
    columns = feature_columns(reference)
    missing = sorted(set(columns) - set(current.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    result = current.copy()
    numeric = reference[columns].select_dtypes(include="number").columns
    for column in numeric:
        try:
            values = pd.to_numeric(result[column]).astype("float64")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            ) from error
        if np.isinf(values).any():
            raise ValueError(f"Column '{column}' contains infinite values.")
        result[column] = values
    for column in set(columns) - set(numeric):
        values = result[column]
        result[column] = values.astype(str).where(values.notna(), np.nan)
    if TARGET in result:
        values = pd.to_numeric(result[TARGET], errors="coerce")
        invalid = result[TARGET].notna() & values.isna()
        if invalid.any() or np.isinf(values).any():
            raise ValueError(
                f"Column '{TARGET}' must contain finite numeric values "
                "or missing values."
            )
        result[TARGET] = values
    return result


def attach_predictions(
    current: pd.DataFrame,
    response: dict,
) -> pd.DataFrame:
    """Align predictions by API row index, never by optional record IDs."""
    predictions = pd.DataFrame(response["predictions"])
    expected = list(range(len(current)))
    if predictions.empty or "row_index" not in predictions:
        raise ValueError("The API returned no indexed predictions.")
    predictions = predictions.sort_values("row_index")
    if predictions["row_index"].tolist() != expected:
        raise ValueError("API prediction row indices do not match the CSV.")
    values = pd.to_numeric(predictions[PREDICTION], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("API predictions contain non-finite values.")
    result = current.copy()
    result[PREDICTION] = values.to_numpy()
    return result


def create_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reports_dir: Path,
    *,
    drift_share: float = 0.5,
) -> MonitoringResult:
    """Persist feature drift and quality against the training reference."""
    if not 0 < drift_share <= 1:
        raise ValueError("The drifted-feature share must be in (0, 1].")
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Monitoring requires at least two rows per dataset.")
    current = prepare_current(reference, current)
    columns = feature_columns(reference)
    numeric = reference[columns].select_dtypes(include="number").columns
    categorical = [column for column in columns if column not in numeric]
    definition = DataDefinition(
        numerical_columns=list(numeric),
        categorical_columns=categorical,
    )
    drift_columns = [
        column for column in columns
        if reference[column].notna().any() and current[column].notna().any()
    ]
    excluded = tuple(
        column for column in columns if column not in drift_columns
    )
    report = Report([
        DataSummaryPreset(),
        *([DataDriftPreset(
            columns=drift_columns, drift_share=drift_share,
        )] if drift_columns else []),
    ])
    snapshot = report.run(
        current_data=Dataset.from_pandas(
            current[columns], data_definition=definition,
        ),
        reference_data=Dataset.from_pandas(
            reference[columns], data_definition=definition,
        ),
    )
    return _persist_report(
        snapshot, reports_dir, "data", excluded_drift_columns=excluded,
    )


def create_regression_report(
    current: pd.DataFrame,
    reports_dir: Path,
) -> MonitoringResult:
    """Measure only labeled current rows; never use training as a test set."""
    if TARGET not in current or PREDICTION not in current:
        raise ValueError("Regression requires target and prediction columns.")
    values = current[[TARGET, PREDICTION]].apply(
        pd.to_numeric, errors="raise",
    )
    mask = np.isfinite(values).all(axis=1)
    labeled = values.loc[mask]
    if len(labeled) < 2:
        raise ValueError(
            "Regression monitoring requires at least two labeled rows "
            "with finite predictions."
        )
    dataset = Dataset.from_pandas(
        labeled,
        data_definition=DataDefinition(
            regression=[Regression(target=TARGET, prediction=PREDICTION)],
        ),
    )
    snapshot = Report([RegressionPreset()]).run(dataset, None)
    return _persist_report(
        snapshot, reports_dir, "regression",
        evaluated_rows=len(labeled), excluded_target_rows=int((~mask).sum()),
    )
