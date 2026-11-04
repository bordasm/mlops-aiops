"""Training-only pipeline reusing production-ready modeling code."""

from __future__ import annotations

import threading
import os
import time
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from coffee_price_predictor.data_pipeline import load_dataset, prepare_features
from coffee_price_predictor.modeling import build_models, build_preprocessor
from coffee_price_predictor.metrics import evaluate_predictions
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from coffee_mlflow_api import config
from coffee_mlflow_api.registry import RegistryService

LEAKAGE_COLUMNS = ("record_id", "lot_id", "quality_grade", "price_usd_per_kg")


class TrainingInProgressError(RuntimeError):
    """Raised when a training run is requested while another one runs."""


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


class TrainingService(RegistryService):
    """Train and promote models; never serve predictions."""

    experiment_names = (config.TRAINING_EXPERIMENT_NAME,)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._training_lock = threading.Lock()

    @property
    def is_training(self) -> bool:
        """Return True while a training run is in progress."""
        return self._training_lock.locked()

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
            job_id = os.environ.get("COFFEE_JOB_ID")
            if job_id:
                mlflow.set_tag("training_job_id", job_id)
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
