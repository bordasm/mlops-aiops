"""Verify container configuration and training in an isolated MLflow store."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as RegistryStore,
)
from mlflow.store.tracking.sqlalchemy_store import (
    SqlAlchemyStore as TrackingStore,
)
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api_mlflow"))
sys.path.insert(0, str(ROOT / "productionReadyCode" / "src"))

from coffee_mlflow_api import config  # noqa: E402
from coffee_mlflow_api.service import (  # noqa: E402
    CoffeeQualityMlflowService,
    ModelNotLoadedError,
)


class ConfigurationTests(unittest.TestCase):
    """Ensure environment overrides do not change notebook/local defaults."""

    def test_custom_paths_and_tracking_uri(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env.update({
                "PYTHONPATH": str(ROOT / "api_mlflow"),
                "COFFEE_ARTIFACTS_DIR": directory,
                "COFFEE_DATA_PATH": str(Path(directory) / "data.csv"),
                "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            })
            script = (
                "import json; from coffee_mlflow_api import config as c; "
                "print(json.dumps([str(c.ARTIFACTS_DIR), str(c.DATA_PATH), "
                "c.TRACKING_URI]))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env, check=True, capture_output=True, text=True,
            )
            paths = json.loads(result.stdout)
            self.assertEqual(Path(paths[0]), Path(directory))
            self.assertEqual(Path(paths[1]), Path(directory) / "data.csv")
            self.assertEqual(paths[2], "http://mlflow:5000")


class ServiceTests(unittest.TestCase):
    """Run real MLflow logging, registry, reload and inference."""

    def test_new_store_training_inference_and_reload(self) -> None:
        previous_uri = mlflow.get_tracking_uri()
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as cleanup,
            ):
                root = Path(directory)
                data = pd.read_csv(
                    ROOT / "data" / "coffee_value_benchmark.csv",
                ).head(100)
                data_path = root / "benchmark.csv"
                data.to_csv(data_path, index=False)
                uri = "sqlite:///" + (root / "mlflow.db").as_posix()
                with (
                    patch.object(config, "ARTIFACTS_DIR", root),
                    patch.object(config, "MLRUNS_DIR", root / "mlruns"),
                    patch(
                        "coffee_mlflow_api.training.build_models",
                        return_value={"RandomForest": RandomForestRegressor(
                            n_estimators=3, max_depth=3, random_state=42,
                        )},
                    ),
                ):
                    service = CoffeeQualityMlflowService(uri, data_path)
                    service.setup()
                    self.assertFalse(service.is_champion_available())
                    for store in (TrackingStore, RegistryStore):
                        cleanup.callback(store._engine_map[uri].dispose)
                    with self.assertRaises(ModelNotLoadedError):
                        service.get_model_info()
                    result = service.train(0.2, 42)
                    self.assertEqual(result["model_version"], "1")
                    self.assertTrue(service.is_champion_available())
                    content = data.head(3).to_csv(index=False).encode("utf-8")
                    prediction = service.predict_from_csv(content, "test.csv")
                    self.assertEqual(prediction["record_count"], 3)
                    self.assertIn("eval_r2", prediction["evaluation"])
                    json.dumps(prediction, allow_nan=False)
                    reloaded = CoffeeQualityMlflowService(uri, data_path)
                    reloaded.setup()
                    self.assertEqual(
                        reloaded.get_model_info()["model_version"], "1",
                    )
                    single = data.head(1).to_csv(index=False).encode("utf-8")
                    one = reloaded.predict_from_csv(single, "single.csv")
                    self.assertIn("eval_mae", one["evaluation"])
                    self.assertNotIn("eval_r2", one["evaluation"])
                    json.dumps(one, allow_nan=False)
        finally:
            mlflow.set_tracking_uri(previous_uri)

    def test_only_finite_targets_are_evaluated(self) -> None:
        result = CoffeeQualityMlflowService._evaluate(
            pd.DataFrame({
                config.TARGET_COLUMN: [80.0, np.inf, np.nan],
            }),
            np.array([81.0, 2.0, 3.0]),
        )
        self.assertEqual(result, {"eval_rmse": 1.0, "eval_mae": 1.0})


if __name__ == "__main__":
    unittest.main()
