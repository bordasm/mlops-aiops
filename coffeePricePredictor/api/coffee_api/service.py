"""Training and inference services used by the REST API."""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from coffee_price_predictor.config import TrainingConfig
from coffee_price_predictor.training import train_and_save_model

from coffee_api import config


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested without a trained model."""


class TrainingInProgressError(RuntimeError):
    """Raised when a training run is requested while another one runs."""


class InvalidInputError(ValueError):
    """Raised when the uploaded inference data is invalid."""


@dataclass(frozen=True)
class LoadedModel:
    """Model artifacts held in memory for inference."""

    model: Any
    preprocessor: Any
    summary: dict[str, Any]
    feature_columns: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]

    @property
    def model_name(self) -> str:
        """Return the name of the selected best model."""
        return str(self.summary.get("best_model", type(self.model).__name__))


def _get_column_groups(preprocessor: Any) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical columns of a fitted transformer."""
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            numerical_columns = list(columns)
        elif name == "cat":
            categorical_columns = list(columns)
    return numerical_columns, categorical_columns


class CoffeeQualityService:
    """Train the coffee quality model and serve predictions."""

    def __init__(
        self,
        artifacts_dir: Path = config.ARTIFACTS_DIR,
        data_path: Path = config.DATA_PATH,
    ) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._data_path = Path(data_path)
        self._training_lock = threading.Lock()
        self._loaded_model: LoadedModel | None = None

    @property
    def is_model_loaded(self) -> bool:
        """Return True when model artifacts are available in memory."""
        return self._loaded_model is not None

    @property
    def is_training(self) -> bool:
        """Return True while a training run is in progress."""
        return self._training_lock.locked()

    def load_artifacts(self) -> bool:
        """Load model artifacts from disk; return False if they are missing."""
        model_path = self._artifacts_dir / config.MODEL_FILE_NAME
        preprocessor_path = self._artifacts_dir / config.PREPROCESSOR_FILE_NAME
        summary_path = self._artifacts_dir / config.SUMMARY_FILE_NAME
        if not all(
            path.exists()
            for path in (model_path, preprocessor_path, summary_path)
        ):
            return False

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        numerical_columns, categorical_columns = _get_column_groups(
            preprocessor
        )
        self._loaded_model = LoadedModel(
            model=model,
            preprocessor=preprocessor,
            summary=summary,
            feature_columns=list(preprocessor.feature_names_in_),
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
        return True

    def train(self, test_size: float, random_state: int) -> dict[str, Any]:
        """Train the candidate models, persist and load the best one."""
        if not self._training_lock.acquire(blocking=False):
            raise TrainingInProgressError(
                "A training run is already in progress."
            )
        try:
            training_config = TrainingConfig(
                data_path=self._data_path,
                output_dir=self._artifacts_dir,
                target_column=config.TARGET_COLUMN,
                test_size=test_size,
                random_state=random_state,
            )
            start = time.perf_counter()
            summary = train_and_save_model(training_config)
            elapsed = time.perf_counter() - start
            self.load_artifacts()
        finally:
            self._training_lock.release()

        return {
            **summary,
            "training_time_seconds": round(elapsed, 3),
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata of the loaded model."""
        loaded = self._require_model()
        return {
            "model_name": loaded.model_name,
            "target_column": config.TARGET_COLUMN,
            "feature_columns": loaded.feature_columns,
            "best_model_metrics": loaded.summary["best_model_metrics"],
            "all_models": loaded.summary["all_models"],
        }

    def predict_from_csv(self, content: bytes) -> dict[str, Any]:
        """Predict the coffee quality score for every row of a CSV file."""
        loaded = self._require_model()
        dataframe = self._read_csv(content)
        features = self._prepare_features(dataframe, loaded)
        predictions = loaded.model.predict(
            loaded.preprocessor.transform(features)
        )

        if config.ID_COLUMN in dataframe.columns:
            record_ids = [
                None if pd.isna(value) else str(value)
                for value in dataframe[config.ID_COLUMN]
            ]
        else:
            record_ids = [None] * len(dataframe)

        items = [
            {
                "row_index": row_index,
                "record_id": record_id,
                "predicted_coffee_quality_score": float(prediction),
            }
            for row_index, (record_id, prediction) in enumerate(
                zip(record_ids, predictions)
            )
        ]
        return {
            "model_name": loaded.model_name,
            "target_column": config.TARGET_COLUMN,
            "record_count": len(items),
            "predictions": items,
        }

    def _require_model(self) -> LoadedModel:
        """Return the loaded model or raise if no model is available."""
        if self._loaded_model is None:
            raise ModelNotLoadedError(
                "No trained model is available. Call POST /train first."
            )
        return self._loaded_model

    @staticmethod
    def _read_csv(content: bytes) -> pd.DataFrame:
        """Parse the uploaded CSV content into a DataFrame."""
        if not content:
            raise InvalidInputError("The uploaded file is empty.")
        try:
            dataframe = pd.read_csv(io.BytesIO(content))
        except (
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as error:
            raise InvalidInputError(
                f"The uploaded file is not a valid CSV: {error}"
            ) from error
        if dataframe.empty:
            raise InvalidInputError("The uploaded CSV contains no data rows.")
        return dataframe

    @staticmethod
    def _prepare_features(
        dataframe: pd.DataFrame,
        loaded: LoadedModel,
    ) -> pd.DataFrame:
        """Validate the input columns and cast them to the training types."""
        missing = [
            column
            for column in loaded.feature_columns
            if column not in dataframe.columns
        ]
        if missing:
            raise InvalidInputError(
                f"The CSV is missing required feature columns: {missing}"
            )

        features = dataframe[loaded.feature_columns].copy()
        for column in loaded.numerical_columns:
            try:
                features[column] = pd.to_numeric(features[column])
            except (TypeError, ValueError) as error:
                raise InvalidInputError(
                    f"Column '{column}' must contain numeric values."
                ) from error
        for column in loaded.categorical_columns:
            series = features[column]
            features[column] = (
                series.astype(str).where(series.notna(), np.nan).astype(object)
            )
        return features
