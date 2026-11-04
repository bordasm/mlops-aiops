"""Inference-only service using remote MLflow models and artifacts."""

from __future__ import annotations

import io
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline
from coffee_price_predictor.metrics import evaluate_predictions

from coffee_mlflow_api import config
from coffee_mlflow_api.registry import ModelNotLoadedError, RegistryService

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


class InvalidInputError(ValueError):
    """Raised when the uploaded inference data is invalid."""


@dataclass(frozen=True)
class ChampionModel:
    """Champion model loaded from the MLflow Model Registry."""

    pipeline: Pipeline
    version: ModelVersion
    feature_columns: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]

    @property
    def model_name(self) -> str:
        """Return the name of the regressor (e.g. LightGBM)."""
        return self.version.tags.get(
            "model_name",
            type(self.pipeline.named_steps["model"]).__name__,
        )


def _model_uri() -> str:
    """Return the registry URI of the champion model."""
    return f"models:/{config.REGISTERED_MODEL_NAME}@{config.MODEL_ALIAS}"


def _get_column_groups(pipeline: Pipeline) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical columns of the preprocessor."""
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    preprocessor = pipeline.named_steps["preprocessor"]
    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            numerical_columns = list(columns)
        elif name == "cat":
            categorical_columns = list(columns)
    return numerical_columns, categorical_columns


class ServingService(RegistryService):
    """Serve and log predictions without importing the training pipeline."""

    experiment_names = (config.INFERENCE_EXPERIMENT_NAME,)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._champion_lock = threading.Lock()
        self._champion: ChampionModel | None = None
        self._prediction_lock = threading.Lock()

    def is_champion_available(self) -> bool:
        """Return True when a champion model exists in the registry."""
        try:
            self._get_champion()
        except ModelNotLoadedError:
            return False
        return True

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata and metrics of the champion model."""
        champion = self._get_champion()
        run = self._require_client().get_run(champion.version.run_id)
        return {
            **self._registered_model_info(champion.version),
            "model_name": champion.model_name,
            "model_run_id": champion.version.run_id,
            "training_run_id": champion.version.tags.get("training_run_id"),
            "target_column": config.TARGET_COLUMN,
            "feature_columns": champion.feature_columns,
            "metrics": run.data.metrics,
        }

    def predict_from_csv(
        self,
        content: bytes,
        file_name: str | None,
    ) -> dict[str, Any]:
        """Predict a CSV file and log the inference run in MLflow."""
        champion = self._get_champion()
        dataframe = self._read_csv(content)
        features = self._prepare_inference_features(dataframe, champion)
        predictions = champion.pipeline.predict(features)

        output = self._build_output(dataframe, predictions)
        evaluation = self._evaluate(dataframe, predictions)
        with self._prediction_lock:
            run_id = self._log_inference_run(
                champion, dataframe, output, predictions,
                evaluation, file_name,
            )

        items = [
            {
                "row_index": row_index,
                "record_id": (
                    None if pd.isna(record_id) else str(record_id)
                ),
                "predicted_coffee_quality_score": float(prediction),
            }
            for row_index, (record_id, prediction) in enumerate(
                zip(output[config.ID_COLUMN], predictions)
            )
        ]
        return {
            **self._registered_model_info(champion.version),
            "model_name": champion.model_name,
            "inference_run_id": run_id,
            "predictions_artifact_uri": (
                f"runs:/{run_id}/{config.PREDICTIONS_FILE_NAME}"
            ),
            "target_column": config.TARGET_COLUMN,
            "record_count": len(items),
            "evaluation": evaluation or None,
            "predictions": items,
        }

    def _get_champion(self) -> ChampionModel:
        """Return the champion model, reloading it if the alias moved.

        The alias is checked on every call, so a model registered by the
        training notebook is picked up without restarting the API.
        """
        try:
            version = self._require_client().get_model_version_by_alias(
                config.REGISTERED_MODEL_NAME, config.MODEL_ALIAS
            )
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            raise ModelNotLoadedError(
                "No champion model is registered. Submit a training job first."
            ) from error

        with self._champion_lock:
            if (
                self._champion is None
                or self._champion.version.version != version.version
            ):
                pipeline = mlflow.sklearn.load_model(
                    f"models:/{config.REGISTERED_MODEL_NAME}/"
                    f"{version.version}"
                )
                numerical_columns, categorical_columns = _get_column_groups(
                    pipeline
                )
                self._champion = ChampionModel(
                    pipeline=pipeline,
                    version=version,
                    feature_columns=list(pipeline.feature_names_in_),
                    numerical_columns=numerical_columns,
                    categorical_columns=categorical_columns,
                )
            return self._champion

    def _refresh_champion(self) -> None:
        """Drop the cached champion and load the current one."""
        with self._champion_lock:
            self._champion = None
        self._get_champion()

    def _log_inference_run(
        self,
        champion: ChampionModel,
        dataframe: pd.DataFrame,
        output: pd.DataFrame,
        predictions: np.ndarray,
        evaluation: dict[str, float],
        file_name: str | None,
    ) -> str:
        """Log parameters, metrics and the predictions CSV to MLflow."""
        source = file_name or "uploaded.csv"
        with mlflow.start_run(
            experiment_id=self._experiment_ids[
                config.INFERENCE_EXPERIMENT_NAME
            ],
            run_name="inference-api",
        ) as run:
            mlflow.set_tags(
                {
                    "pipeline": "inference",
                    "source": "rest_api",
                    "registered_model_name": config.REGISTERED_MODEL_NAME,
                    "registered_model_version": champion.version.version,
                    "model_alias": config.MODEL_ALIAS,
                }
            )
            mlflow.log_params(
                {
                    "model_uri": self._registered_model_info(
                        champion.version,
                    )["model_uri"],
                    "model_version": champion.version.version,
                    "input_file_name": source,
                    "n_rows": len(dataframe),
                }
            )
            mlflow.log_input(
                mlflow.data.from_pandas(
                    dataframe,
                    source=source,
                    name=Path(source).stem,
                ),
                context="inference",
            )
            mlflow.log_metrics(
                {
                    "prediction_mean": float(np.mean(predictions)),
                    "prediction_std": float(np.std(predictions)),
                    "prediction_min": float(np.min(predictions)),
                    "prediction_max": float(np.max(predictions)),
                    **evaluation,
                }
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                predictions_path = (
                    Path(temp_dir) / config.PREDICTIONS_FILE_NAME
                )
                output.to_csv(predictions_path, index=False)
                mlflow.log_artifact(str(predictions_path))
        return run.info.run_id

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
    def _prepare_inference_features(
        dataframe: pd.DataFrame,
        champion: ChampionModel,
    ) -> pd.DataFrame:
        """Validate the input columns and cast them to the training types."""
        missing = [
            column
            for column in champion.feature_columns
            if column not in dataframe.columns
        ]
        if missing:
            raise InvalidInputError(
                f"The CSV is missing required feature columns: {missing}"
            )

        features = dataframe[champion.feature_columns].copy()
        for column in champion.numerical_columns:
            try:
                features[column] = pd.to_numeric(features[column]).astype(
                    "float64"
                )
            except (TypeError, ValueError) as error:
                raise InvalidInputError(
                    f"Column '{column}' must contain numeric values."
                ) from error
        for column in champion.categorical_columns:
            series = features[column]
            features[column] = (
                series.astype(str).where(series.notna(), np.nan).astype(object)
            )
        return features

    @staticmethod
    def _build_output(
        dataframe: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Combine the record identifiers with the predictions."""
        if config.ID_COLUMN in dataframe.columns:
            output = dataframe[[config.ID_COLUMN]].copy()
        else:
            output = pd.DataFrame({config.ID_COLUMN: [None] * len(dataframe)})
        output[config.PREDICTION_COLUMN] = predictions
        return output

    @staticmethod
    def _evaluate(
        dataframe: pd.DataFrame,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Return evaluation metrics if the CSV contains the target."""
        if config.TARGET_COLUMN not in dataframe.columns:
            return {}
        target = pd.to_numeric(
            dataframe[config.TARGET_COLUMN], errors="coerce"
        )
        mask = np.isfinite(target).to_numpy()
        if not mask.any():
            return {}
        if mask.sum() == 1:
            absolute_error = float(
                np.abs(target[mask].to_numpy() - predictions[mask])[0]
            )
            return {
                "eval_rmse": absolute_error,
                "eval_mae": absolute_error,
            }
        return {
            f"eval_{name}": value
            for name, value in evaluate_predictions(
                target[mask], predictions[mask]
            ).items()
        }
