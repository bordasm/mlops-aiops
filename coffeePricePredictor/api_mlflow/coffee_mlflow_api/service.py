"""MLflow based training and inference services used by the REST API."""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from coffee_price_predictor.data_pipeline import (
    load_dataset,
    prepare_features,
)
from coffee_price_predictor.modeling import (
    build_models,
    build_preprocessor,
    evaluate_predictions,
)
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from coffee_mlflow_api import config

LEAKAGE_COLUMNS = ("record_id", "lot_id", "quality_grade", "price_usd_per_kg")

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


class ModelNotLoadedError(RuntimeError):
    """Raised when no champion model is available in the registry."""


class TrainingInProgressError(RuntimeError):
    """Raised when a training run is requested while another one runs."""


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


def _to_float_features(features: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to float64 so they accept missing values."""
    features = features.copy()
    numerical_columns = features.select_dtypes(include=["number"]).columns
    features[numerical_columns] = features[numerical_columns].astype(
        "float64"
    )
    return features


def _prefixed_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Return regression metrics with a split prefix (e.g. test_r2)."""
    return {
        f"{prefix}_{name}": value
        for name, value in evaluate_predictions(y_true, y_pred).items()
    }


def _nest_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    """Convert flat MLflow metrics into the API evaluation structure."""
    return {
        "train": {
            name: metrics[f"train_{name}"] for name in ("rmse", "mae", "r2")
        },
        "test": {
            name: metrics[f"test_{name}"] for name in ("rmse", "mae", "r2")
        },
        "fit_time_seconds": metrics["fit_time_seconds"],
    }


class CoffeeQualityMlflowService:
    """Train, register and serve the coffee quality model with MLflow."""

    def __init__(
        self,
        tracking_uri: str = config.TRACKING_URI,
        data_path: Path = config.DATA_PATH,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._data_path = Path(data_path)
        self._training_lock = threading.Lock()
        self._champion_lock = threading.Lock()
        self._champion: ChampionModel | None = None
        self._experiment_ids: dict[str, str] = {}
        self._client: MlflowClient | None = None

    @property
    def tracking_uri(self) -> str:
        """Return the MLflow tracking URI."""
        return self._tracking_uri

    @property
    def is_training(self) -> bool:
        """Return True while a training run is in progress."""
        return self._training_lock.locked()

    def setup(self) -> None:
        """Configure MLflow and create the experiments if needed."""
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        for name in (
            config.TRAINING_EXPERIMENT_NAME,
            config.INFERENCE_EXPERIMENT_NAME,
        ):
            self._experiment_ids[name] = self._get_or_create_experiment(name)

    def is_champion_available(self) -> bool:
        """Return True when a champion model exists in the registry."""
        try:
            self._get_champion()
        except ModelNotLoadedError:
            return False
        return True

    def train(self, test_size: float, random_state: int) -> dict[str, Any]:
        """Train the candidates, log them and register the best model."""
        if not self._training_lock.acquire(blocking=False):
            raise TrainingInProgressError(
                "A training run is already in progress."
            )
        try:
            start = time.perf_counter()
            result = self._run_training(test_size, random_state)
            result["training_time_seconds"] = round(
                time.perf_counter() - start, 3
            )
        finally:
            self._training_lock.release()
        return result

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
        run_id = self._log_inference_run(
            champion, dataframe, output, predictions, evaluation, file_name
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

    def _run_training(
        self,
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        """Execute the training pipeline inside MLflow runs."""
        dataframe = load_dataset(self._data_path)
        features, target = prepare_features(
            dataframe=dataframe,
            target_column=config.TARGET_COLUMN,
            leakage_columns=LEAKAGE_COLUMNS,
        )
        features = _to_float_features(features)
        target = target.astype("float64")
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        preprocessor = build_preprocessor(x_train)
        x_train_processed = preprocessor.fit_transform(x_train)
        x_test_processed = preprocessor.transform(x_test)

        experiment_id = self._experiment_ids[config.TRAINING_EXPERIMENT_NAME]
        results: dict[str, dict[str, float]] = {}
        model_uris: dict[str, str] = {}
        candidate_run_ids: dict[str, str] = {}

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="training-api",
        ) as parent_run:
            mlflow.set_tags(
                {
                    "pipeline": "training",
                    "source": "rest_api",
                    "target_column": config.TARGET_COLUMN,
                }
            )
            mlflow.log_params(
                {
                    "data_path": str(self._data_path),
                    "target_column": config.TARGET_COLUMN,
                    "leakage_columns": ",".join(LEAKAGE_COLUMNS),
                    "test_size": test_size,
                    "random_state": random_state,
                    "selection_metric": config.SELECTION_METRIC,
                    "n_rows": len(dataframe),
                    "n_features": features.shape[1],
                    "n_train_rows": len(x_train),
                    "n_test_rows": len(x_test),
                }
            )
            mlflow.log_input(
                mlflow.data.from_pandas(
                    dataframe,
                    source=str(self._data_path),
                    targets=config.TARGET_COLUMN,
                    name=self._data_path.stem,
                ),
                context="training",
            )

            for model_name, model in build_models().items():
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=model_name,
                    nested=True,
                ) as child_run:
                    fit_start = time.perf_counter()
                    model.fit(x_train_processed, y_train)
                    fit_time = time.perf_counter() - fit_start

                    metrics = {
                        **_prefixed_metrics(
                            y_train,
                            model.predict(x_train_processed),
                            "train",
                        ),
                        **_prefixed_metrics(
                            y_test,
                            model.predict(x_test_processed),
                            "test",
                        ),
                        "fit_time_seconds": round(fit_time, 3),
                    }
                    mlflow.set_tags(
                        {"pipeline": "training", "model_name": model_name}
                    )
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(
                        {
                            f"model__{key}": value
                            for key, value in model.get_params().items()
                        }
                    )
                    mlflow.log_metrics(metrics)

                    model_pipeline = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("model", model),
                        ]
                    )
                    input_example = x_train.head(5)
                    model_info = mlflow.sklearn.log_model(
                        model_pipeline,
                        name=config.MODEL_ARTIFACT_NAME,
                        serialization_format="cloudpickle",
                        signature=infer_signature(
                            input_example,
                            model_pipeline.predict(input_example),
                        ),
                        input_example=input_example,
                    )
                results[model_name] = metrics
                model_uris[model_name] = model_info.model_uri
                candidate_run_ids[model_name] = child_run.info.run_id

            best_model_name = max(
                results,
                key=lambda name: results[name][config.SELECTION_METRIC],
            )
            best_metrics = results[best_model_name]
            comparison = (
                pd.DataFrame(results)
                .T.rename_axis("model")
                .sort_values(config.SELECTION_METRIC, ascending=False)
                .reset_index()
            )
            mlflow.log_table(comparison, "model_comparison.json")
            mlflow.log_dict(
                {
                    "best_model": best_model_name,
                    "selection_metric": config.SELECTION_METRIC,
                    "best_model_metrics": best_metrics,
                    "all_models": results,
                    "feature_columns": features.columns.tolist(),
                    "best_model_uri": model_uris[best_model_name],
                },
                "model_summary.json",
            )
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metrics(
                {f"best_{key}": value for key, value in best_metrics.items()}
            )

            model_version = self._register_champion(
                model_uri=model_uris[best_model_name],
                model_name=best_model_name,
                training_run_id=parent_run.info.run_id,
                selection_score=best_metrics[config.SELECTION_METRIC],
            )
            mlflow.set_tags(
                {
                    "registered_model_name": config.REGISTERED_MODEL_NAME,
                    "registered_model_version": model_version.version,
                }
            )

        self._refresh_champion()
        return {
            **self._registered_model_info(model_version),
            "message": "Training completed and champion model registered.",
            "best_model": best_model_name,
            "best_model_metrics": _nest_metrics(best_metrics),
            "all_models": {
                name: _nest_metrics(metrics)
                for name, metrics in results.items()
            },
            "training_run_id": parent_run.info.run_id,
            "candidate_run_ids": candidate_run_ids,
        }

    def _register_champion(
        self,
        model_uri: str,
        model_name: str,
        training_run_id: str,
        selection_score: float,
    ) -> ModelVersion:
        """Register a model version and move the champion alias to it."""
        client = self._require_client()
        model_version = mlflow.register_model(
            model_uri,
            config.REGISTERED_MODEL_NAME,
            tags={
                "model_name": model_name,
                "training_run_id": training_run_id,
                config.SELECTION_METRIC: f"{selection_score:.6f}",
            },
        )
        client.set_registered_model_alias(
            config.REGISTERED_MODEL_NAME,
            config.MODEL_ALIAS,
            model_version.version,
        )
        client.update_model_version(
            config.REGISTERED_MODEL_NAME,
            model_version.version,
            description=(
                f"{model_name} predicting {config.TARGET_COLUMN}; "
                f"{config.SELECTION_METRIC}={selection_score:.4f}"
            ),
        )
        return client.get_model_version(
            config.REGISTERED_MODEL_NAME, model_version.version
        )

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
            raise ModelNotLoadedError(
                "No champion model is registered. Call POST /train first."
            ) from error

        with self._champion_lock:
            if (
                self._champion is None
                or self._champion.version.version != version.version
            ):
                pipeline = mlflow.sklearn.load_model(_model_uri())
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
                    "model_uri": _model_uri(),
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

    def _get_or_create_experiment(self, name: str) -> str:
        """Return the id of an experiment, creating it if needed."""
        client = self._require_client()
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return client.create_experiment(
            name,
            artifact_location=(config.MLRUNS_DIR / name).as_uri(),
        )

    def _require_client(self) -> MlflowClient:
        """Return the MLflow client or raise if setup was not called."""
        if self._client is None:
            raise RuntimeError("Call setup() before using the service.")
        return self._client

    @staticmethod
    def _registered_model_info(version: ModelVersion) -> dict[str, str]:
        """Return the registry reference of a model version."""
        return {
            "registered_model_name": config.REGISTERED_MODEL_NAME,
            "model_version": str(version.version),
            "model_alias": config.MODEL_ALIAS,
            "model_uri": (
                f"models:/{config.REGISTERED_MODEL_NAME}/{version.version}"
            ),
        }

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
        mask = target.notna().to_numpy()
        if not mask.any():
            return {}
        return {
            f"eval_{name}": value
            for name, value in evaluate_predictions(
                target[mask], predictions[mask]
            ).items()
        }
