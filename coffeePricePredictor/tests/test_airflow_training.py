"""Verify daily orchestration without installing Airflow on Windows."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

HEALTH = {"status": "ok", "training_in_progress": False}
TRAINING = {
    "training_run_id": "training-123",
    "registered_model_name": "coffee-quality-model",
    "model_version": "3",
    "model_alias": "champion",
    "best_model": "LightGBM",
}


class RetrainingTests(unittest.TestCase):
    """Make failures explicit and never automatically repeat POST /train."""

    @patch("coffee_training._request")
    def test_completed_training_is_verified_and_summarized(self, request):
        request.side_effect = [HEALTH, TRAINING, TRAINING]
        with patch.dict(os.environ, {"COFFEE_API_URL": "http://api:8000"}):
            result = retrain_coffee_model()
        self.assertEqual(result["training_run_id"], "training-123")
        self.assertEqual(result["model_version"], "3")
        self.assertEqual(
            [call.args[1] for call in request.call_args_list],
            ["/health", "/train", "/model"],
        )
        self.assertEqual(
            request.call_args_list[1].kwargs["payload"],
            {"test_size": 0.2, "random_state": 42},
        )

    @patch("coffee_training._request")
    def test_active_training_is_a_visible_failure_without_post(self, request):
        request.return_value = {"status": "ok", "training_in_progress": True}
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "active"):
                retrain_coffee_model()
        self.assertEqual(request.call_count, 1)

    @patch("coffee_training._request")
    def test_invalid_health_does_not_submit_training(self, request):
        for health in ({}, {"status": "ok", "training_in_progress": "false"}):
            request.reset_mock()
            request.return_value = health
            with self.subTest(health=health):
                with self.assertLogs("coffee_training", level="ERROR"):
                    with self.assertRaisesRegex(RetrainingError, "health"):
                        retrain_coffee_model()
                self.assertEqual(request.call_count, 1)

    @patch("coffee_training._request")
    def test_invalid_training_result_fails(self, request):
        for field in TRAINING:
            result = dict(TRAINING)
            result.pop(field)
            request.side_effect = [HEALTH, result]
            with self.subTest(field=field):
                with self.assertLogs("coffee_training", level="ERROR"):
                    with self.assertRaisesRegex(RetrainingError, field):
                        retrain_coffee_model()

    @patch("coffee_training._request")
    def test_missing_champion_alias_fails(self, request):
        request.side_effect = [
            HEALTH, {**TRAINING, "model_alias": "candidate"},
        ]
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "champion alias"):
                retrain_coffee_model()

    @patch("coffee_training._request")
    def test_champion_mismatch_fails_without_retraining(self, request):
        request.side_effect = [
            HEALTH, TRAINING, {**TRAINING, "model_version": "4"},
        ]
        with self.assertLogs("coffee_training", level="ERROR"):
            with self.assertRaisesRegex(RetrainingError, "verification"):
                retrain_coffee_model()
        self.assertEqual(request.call_count, 3)

    @patch("coffee_training.urlopen")
    def test_timeout_is_not_retried(self, urlopen):
        urlopen.side_effect = TimeoutError("read timed out")
        with self.assertRaisesRegex(RetrainingError, "may still"):
            _request("http://api", "/train", payload={})
        self.assertEqual(urlopen.call_count, 1)

    @patch("coffee_training.urlopen")
    def test_connection_error_is_not_retried(self, urlopen):
        urlopen.side_effect = URLError("unavailable")
        with self.assertRaisesRegex(RetrainingError, "connection failed"):
            _request("http://api", "/health")
        self.assertEqual(urlopen.call_count, 1)

    @patch("coffee_training.urlopen")
    def test_http_conflict_and_server_error_are_not_retried(self, urlopen):
        for status in (409, 500, 503):
            urlopen.reset_mock()
            urlopen.side_effect = HTTPError(
                "http://api/train", status, "failure", {},
                io.BytesIO(b'{"detail":"training unavailable"}'),
            )
            with self.subTest(status=status):
                with self.assertRaisesRegex(
                    RetrainingError, f"HTTP {status}.*unavailable",
                ):
                    _request("http://api", "/train", payload={})
            self.assertEqual(urlopen.call_count, 1)

    @patch("coffee_training.urlopen")
    def test_invalid_json_and_non_object_responses_fail(self, urlopen):
        for raw in (b"not-json", b"[]"):
            context = Mock()
            context.__enter__ = Mock(return_value=io.BytesIO(raw))
            context.__exit__ = Mock(return_value=False)
            urlopen.return_value = context
            with self.subTest(raw=raw), self.assertRaises(RetrainingError):
                _request("http://api", "/train", payload={})

    @patch("coffee_training.urlopen")
    def test_request_timeout_and_method(self, urlopen):
        context = Mock()
        context.__enter__ = Mock(return_value=io.BytesIO(b'{"status":"ok"}'))
        context.__exit__ = Mock(return_value=False)
        urlopen.return_value = context
        _request("http://api/", "/health")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://api/health")
        self.assertEqual(request.method, "GET")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30)
        context.__enter__.return_value = io.BytesIO(b"{}")
        _request("http://api/", "/train", payload={"test_size": 0.2})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"test_size": 0.2})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 1800)

    def test_real_http_transport_submits_training_once(self) -> None:
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(("GET", self.path, None))
                self.send_json(HEALTH if self.path == "/health" else TRAINING)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(("POST", self.path, json.loads(body)))
                self.send_json(TRAINING)

            def send_json(self, payload):
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with patch.dict(
                    os.environ,
                    {"COFFEE_API_URL": url, "NO_PROXY": "127.0.0.1"},
                ):
                    result = retrain_coffee_model()
                self.assertEqual(result["best_model"], "LightGBM")
                self.assertEqual(
                    [method for method, _, _ in calls],
                    ["GET", "POST", "GET"],
                )
                self.assertEqual(calls[1][1], "/train")
                self.assertEqual(calls[1][2]["test_size"], 0.2)
            finally:
                server.shutdown()
                thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
