"""Asynchronous Airflow client behavior without Airflow on Windows."""

import io
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "airflow" / "dags"))

from coffee_training import (  # noqa: E402
    RetrainingError,
    _request,
    retrain_coffee_model,
)

RESULT = {
    "training_run_id": "run-1",
    "registered_model_name": "coffee-quality-model",
    "model_version": "2", "model_alias": "champion",
    "best_model": "LightGBM",
}
SUCCESS = {"job_id": "job-1", "status": "succeeded", "result": RESULT}


class AirflowJobTests(unittest.TestCase):
    @patch("coffee_training.time.sleep")
    @patch("coffee_training._request")
    def test_submit_poll_and_verify(self, request, sleep):
        request.side_effect = [
            {"job_id": "job-1", "status": "queued"},
            {"job_id": "job-1", "status": "running"},
            SUCCESS, RESULT,
        ]
        result = retrain_coffee_model("airflow:stable-key")
        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(
            request.call_args_list[0].kwargs["payload"]["request_id"],
            "airflow:stable-key",
        )
        self.assertEqual(
            [call.args[1] for call in request.call_args_list],
            ["/jobs", "/jobs/job-1", "/jobs/job-1", "/model"],
        )
        self.assertEqual(sleep.call_count, 2)

    @patch("coffee_training._request")
    def test_existing_completed_job_is_not_resubmitted(self, request):
        request.side_effect = [SUCCESS, RESULT]
        retrain_coffee_model("same-request")
        self.assertEqual(request.call_count, 2)

    @patch("coffee_training._request")
    def test_failed_job_is_visible(self, request):
        request.return_value = {
            "job_id": "job-1", "status": "failed",
            "error": "worker interrupted",
        }
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "interrupted"):
                retrain_coffee_model("same-request")
        self.assertEqual(request.call_count, 1)

    @patch("coffee_training.time.monotonic", side_effect=[0, 1801])
    @patch("coffee_training._request")
    def test_polling_timeout_never_submits_twice(self, request, monotonic):
        request.return_value = {
            "job_id": "job-1", "status": "running",
        }
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "deadline"):
                retrain_coffee_model("same-request")
        self.assertEqual(request.call_count, 1)

    @patch("coffee_training._request")
    def test_invalid_job_id(self, request):
        request.return_value = {"status": "queued"}
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "job_id"):
                retrain_coffee_model("key")

    @patch("coffee_training._request")
    def test_invalid_success_result_fails_explicitly(self, request):
        for field in RESULT:
            result = dict(RESULT)
            result.pop(field)
            request.side_effect = [
                {**SUCCESS, "result": result},
            ]
            with self.subTest(field=field):
                with self.assertLogs("coffee_training", level="ERROR"):
                    with self.assertRaisesRegex(RetrainingError, field):
                        retrain_coffee_model("key")

    @patch("coffee_training._request")
    def test_mismatched_champion(self, request):
        request.side_effect = [SUCCESS, {**RESULT, "model_version": "3"}]
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "verification"):
                retrain_coffee_model("key")

    @patch("coffee_training.urlopen")
    def test_http_errors_are_visible(self, urlopen):
        for code in (409, 500, 503):
            urlopen.side_effect = HTTPError(
                "http://training/jobs", code, "error", {},
                io.BytesIO(b'{"detail":"unavailable"}'),
            )
            with self.subTest(code=code):
                with self.assertRaisesRegex(RetrainingError, f"HTTP {code}"):
                    _request("http://training", "/jobs", payload={})

    @patch("coffee_training.urlopen")
    def test_timeout_and_connection_errors_do_not_retry(self, urlopen):
        for error in (TimeoutError("timeout"), URLError("unavailable")):
            urlopen.reset_mock()
            urlopen.side_effect = error
            with self.subTest(error=error), self.assertRaises(RetrainingError):
                _request("http://training", "/jobs", payload={})
            self.assertEqual(urlopen.call_count, 1)

    @patch("coffee_training.urlopen")
    def test_response_shape_and_json_validation(self, urlopen):
        for raw in (b"not-json", b"[]"):
            context = Mock()
            context.__enter__ = Mock(return_value=io.BytesIO(raw))
            context.__exit__ = Mock(return_value=False)
            urlopen.return_value = context
            with self.subTest(raw=raw), self.assertRaises(RetrainingError):
                _request("http://training", "/jobs")


if __name__ == "__main__":
    unittest.main()
