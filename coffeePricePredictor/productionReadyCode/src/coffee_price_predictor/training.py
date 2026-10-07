"""Training and artifact management for the coffee quality benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from coffee_price_predictor.config import TrainingConfig
from coffee_price_predictor.data_pipeline import load_dataset, prepare_features
from coffee_price_predictor.modeling import (
    build_models,
    build_preprocessor,
    evaluate_predictions,
)


def train_and_save_model(config: TrainingConfig) -> dict[str, Any]:
    """Train benchmark models, select the best candidate, and persist artifacts."""
    dataframe = load_dataset(config.data_path)
    features, target = prepare_features(
        dataframe=dataframe,
        target_column=config.target_column,
        leakage_columns=config.leakage_columns,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    result_map: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    for model_name, model in build_models().items():
        start = time.time()
        model.fit(x_train_processed, y_train)
        elapsed = time.time() - start

        train_pred = model.predict(x_train_processed)
        test_pred = model.predict(x_test_processed)

        result_map[model_name] = {
            "train": evaluate_predictions(y_train, train_pred),
            "test": evaluate_predictions(y_test, test_pred),
            "time_seconds": round(elapsed, 3),
        }
        trained_models[model_name] = model

    best_model_name = max(
        result_map,
        key=lambda name: result_map[name]["test"]["r2"],
    )
    best_model = trained_models[best_model_name]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "coffee_quality_model.joblib"
    preprocessor_path = output_dir / "preprocessor.joblib"
    summary_path = output_dir / "model_summary.json"

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    summary = {
        "best_model": best_model_name,
        "best_model_metrics": result_map[best_model_name],
        "all_models": result_map,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
