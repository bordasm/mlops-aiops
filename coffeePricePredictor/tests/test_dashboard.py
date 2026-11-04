"""Tests for monitoring, HTTP errors, reports and Streamlit interactions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from client import ApiError, CoffeeApiClient  # noqa: E402
from monitoring import (  # noqa: E402
    PREDICTION,
    TARGET,
    attach_predictions,
    create_regression_report,
    create_report,
    feature_columns,
    prepare_current,
    read_csv,
)


class MonitoringTests(unittest.TestCase):
    """Verify schema validation and actual Evidently report outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = pd.read_csv(
            ROOT / "data" / "coffee_value_benchmark.csv",
        ).head(150)

    def test_feature_columns_match_training(self) -> None:
        columns = feature_columns(self.reference)
        self.assertEqual(len(columns), 31)
        for excluded in (
            TARGET, "record_id", "lot_id", "quality_grade", "price_usd_per_kg",
        ):
            self.assertNotIn(excluded, columns)

    def test_empty_csv_is_rejected(self) -> None:
        for content in (b"", b"column\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(content)

    def test_oversized_csv_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "20 MB"):
            read_csv(b"x" * (20 * 1024 * 1024 + 1))

    def test_missing_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required"):
            prepare_current(
                self.reference, self.reference.drop(columns="species"),
            )

    def test_invalid_numeric_and_target_are_rejected(self) -> None:
        for column, value in (
            ("farm_size_hectares", "not-a-number"),
            ("farm_size_hectares", np.inf),
            (TARGET, "not-a-number"),
            (TARGET, np.inf),
        ):
            current = self.reference.head(4).copy()
            current[column] = value
            with self.subTest(column=column, value=value):
                with self.assertRaises(ValueError):
                    prepare_current(self.reference, current)

    def test_missing_values_are_preserved_for_quality(self) -> None:
        current = self.reference.head(4).copy()
        current["species"] = np.nan
        current["farm_size_hectares"] = np.nan
        validated = prepare_current(self.reference, current)
        self.assertTrue(validated["species"].isna().all())
        self.assertTrue(validated["farm_size_hectares"].isna().all())

    def test_predictions_align_by_row_index(self) -> None:
        current = self.reference.head(2).copy()
        current.index = [80, 90]
        response = {"predictions": [
            {"row_index": 1, PREDICTION: 83.0},
            {"row_index": 0, PREDICTION: 82.0},
        ]}
        scored = attach_predictions(current, response)
        self.assertEqual(scored[PREDICTION].tolist(), [82.0, 83.0])
        self.assertEqual(scored.index.tolist(), [80, 90])

    def test_duplicate_prediction_indices_are_rejected(self) -> None:
        response = {"predictions": [
            {"row_index": 0, PREDICTION: 83.0},
            {"row_index": 0, PREDICTION: 82.0},
        ]}
        with self.assertRaisesRegex(ValueError, "row indices"):
            attach_predictions(self.reference.head(2), response)

    def test_actual_evidently_drift_and_quality_report(self) -> None:
        current = self.reference.head(100).copy()
        current["altitude_mean_meters"] += 5000
        current["farm_size_hectares"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            result = create_report(
                self.reference, current, Path(directory), drift_share=0.1,
            )
            self.assertIn(
                "farm_size_hectares", result.excluded_drift_columns,
            )
            self.assertGreater(result.html_path.stat().st_size, 1000)
            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            names = [metric["metric_name"] for metric in metrics]
            self.assertTrue(any("Drift" in name for name in names))
            self.assertTrue(any("Missing" in name for name in names))
            drift_metrics = [
                metric for metric in metrics
                if "DriftedColumnsCount" in metric["metric_name"]
            ]
            self.assertEqual(len(drift_metrics), 1)
            self.assertGreater(drift_metrics[0]["value"]["count"], 0)
            serialized = json.dumps(payload)
            self.assertNotIn('"column": "record_id"', serialized)
            self.assertNotIn('"column": "price_usd_per_kg"', serialized)

    def test_actual_evidently_regression_report_filters_missing_labels(self):
        current = pd.DataFrame({
            TARGET: [80.0, 82.0, 84.0, np.nan],
            PREDICTION: [81.0, 83.0, 85.0, 90.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            result = create_regression_report(current, Path(directory))
            self.assertEqual(result.evaluated_rows, 3)
            self.assertEqual(result.excluded_target_rows, 1)
            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            mae = [
                item for item in payload["metrics"]
                if item["metric_name"].startswith("MAE(")
            ]
            self.assertEqual(len(mae), 1)
            self.assertAlmostEqual(mae[0]["value"]["mean"], 1.0)
            self.assertGreater(result.html_path.stat().st_size, 1000)

    def test_single_labeled_row_is_not_reported_as_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least two"):
                create_regression_report(
                    pd.DataFrame({TARGET: [80.0], PREDICTION: [81.0]}),
                    Path(directory),
                )


class ClientTests(unittest.TestCase):
    """Verify errors are visible rather than success-shaped fallbacks."""

    @patch("client.requests.request")
    def test_timeout_is_explicit(self, request: Mock) -> None:
        request.side_effect = requests.Timeout("timeout")
        with self.assertRaisesRegex(ApiError, "may still be running"):
            CoffeeApiClient("http://api").request("POST", "/train")

    @patch("client.requests.request")
    def test_connection_error_is_explicit(self, request: Mock) -> None:
        request.side_effect = requests.ConnectionError("unavailable")
        with self.assertRaisesRegex(ApiError, "Cannot connect"):
            CoffeeApiClient("http://api").request("GET", "/health")

    @patch("client.requests.request")
    def test_http_error_is_explicit(self, request: Mock) -> None:
        request.return_value = Mock(
            ok=False, status_code=503,
            json=Mock(return_value={"detail": "Train a model first."}),
        )
        with self.assertRaisesRegex(ApiError, "503.*Train"):
            CoffeeApiClient("http://api").request("GET", "/model")

    @patch("client.requests.request")
    def test_successful_upload(self, request: Mock) -> None:
        request.return_value = Mock(
            ok=True, status_code=200,
            json=Mock(return_value={"predictions": []}),
        )
        CoffeeApiClient("http://api/").predict(b"a\n1", "sample.csv")
        self.assertEqual(request.call_args.args[1], "http://api/predict")
        self.assertIn("file", request.call_args.kwargs["files"])


class StreamlitTests(unittest.TestCase):
    """Run the actual Streamlit script without an external API dependency."""

    def test_same_report_can_render_in_current_and_saved_tabs(self) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "report.html"
            companion = html.with_suffix(".json")
            html.write_text("<html><body>Test report</body></html>")
            companion.write_text("{}")
            script = (
                "from app import display_report\n"
                "from monitoring import MonitoringResult\n"
                "from pathlib import Path\n"
                f"result = MonitoringResult(Path({str(html)!r}), "
                f"Path({str(companion)!r}), 0, 0)\n"
                "display_report(result, 'Data drift and quality')\n"
                "display_report(result, 'Saved report')\n"
            )
            app = AppTest.from_string(script).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("download_button")), 4)

    @patch("client.CoffeeApiClient.request")
    def test_new_store_and_unavailable_api(self, request: Mock) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"COFFEE_REPORTS_DIR": directory}):
                request.return_value = {
                    "status": "ok",
                    "champion_model_available": False,
                    "training_in_progress": False,
                }
                app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"))
                app.run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(
                    "New model store" in item.value for item in app.info
                ))
                request.side_effect = ApiError("API unreachable")
                app.run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.error[0].value, "API unreachable")

    @patch("client.CoffeeApiClient.request")
    def test_training_comparison_and_champion(self, request: Mock) -> None:
        from streamlit.testing.v1 import AppTest

        metrics = {"rmse": 1.0, "mae": 0.8, "r2": 0.9}
        trained = {"train": metrics, "test": metrics, "fit_time_seconds": 0.1}

        def respond(method, path, **kwargs):
            if path == "/health":
                return {
                    "status": "ok", "champion_model_available": True,
                    "training_in_progress": False,
                }
            if path == "/model":
                return {
                    "model_name": "RandomForest", "model_version": "1",
                    "model_run_id": "candidate", "metrics": {"test_r2": 0.9},
                }
            return {
                "message": "Training completed", "training_run_id": "parent",
                "best_model": "RandomForest",
                "all_models": {"RandomForest": trained},
            }

        request.side_effect = respond
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"COFFEE_REPORTS_DIR": directory}):
                app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"))
                app.run(timeout=30)
                train_button = next(
                    button for button in app.button
                    if button.label == "Train models"
                )
                train_button.click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertIn("training_result", app.session_state)
                self.assertTrue(any(
                    "test_r2" in frame.value.columns
                    for frame in app.dataframe
                ))


if __name__ == "__main__":
    unittest.main()
