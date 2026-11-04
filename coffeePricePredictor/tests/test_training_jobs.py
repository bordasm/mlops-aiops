"""Durability, idempotency and isolated HTTP service contracts."""

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api_mlflow"))

from coffee_mlflow_api.jobs import JobConflictError, JobStore  # noqa: E402


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jobs.db"
        self.store = JobStore(self.path)
        self.parameters = {"test_size": 0.2, "random_state": 42}

    def test_same_request_returns_same_job_after_restart(self):
        first = self.store.submit("stable-key", self.parameters)
        restarted = JobStore(self.path)
        second = restarted.submit("stable-key", self.parameters)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(len(restarted.list_jobs()), 1)

    def test_same_key_with_different_inputs_conflicts(self):
        self.store.submit("key", self.parameters)
        with self.assertRaises(JobConflictError):
            self.store.submit("key", {**self.parameters, "test_size": 0.3})

    def test_parallel_submissions_allow_one_active_job(self):
        def submit(index):
            try:
                return self.store.submit(str(index), self.parameters)
            except JobConflictError:
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(8)))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(len(self.store.list_jobs()), 1)

    def test_claim_finish_and_terminal_idempotency(self):
        first = self.store.submit("key", self.parameters)
        claimed = self.store.claim()
        self.assertEqual(claimed["job_id"], first["job_id"])
        self.assertEqual(self.store.get(first["job_id"])["status"], "running")
        self.store.finish(first["job_id"], result={"model_version": "1"})
        self.assertEqual(
            self.store.submit("key", self.parameters)["status"], "succeeded",
        )
        self.assertIsNotNone(self.store.submit("next", self.parameters))

    def test_interrupted_training_fails_but_queued_work_survives(self):
        first = self.store.submit("key", self.parameters)
        self.store.recover_interrupted()
        self.assertEqual(self.store.get(first["job_id"])["status"], "queued")
        self.store.claim()
        self.store.recover_interrupted()
        job = self.store.get(first["job_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("Inspect MLflow", job["error"])
        self.assertEqual(
            self.store.submit("key", self.parameters)["status"], "failed",
        )

    def test_worker_heartbeat(self):
        self.assertFalse(self.store.worker_ready())
        self.store.heartbeat()
        self.assertTrue(self.store.worker_ready())
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE worker SET heartbeat='2026-01-01T00:00:00+00:00'",
            )
        self.assertFalse(self.store.worker_ready())

    def test_invalid_completion_and_unknown_job(self):
        self.assertIsNone(self.store.get("unknown"))
        with self.assertRaises(ValueError):
            self.store.finish("unknown")
        with self.assertRaises(RuntimeError):
            self.store.finish("unknown", result={})

    def test_training_http_contract(self):
        from fastapi.testclient import TestClient
        from coffee_mlflow_api.training_app import app

        with patch(
            "coffee_mlflow_api.training_app.get_store",
            return_value=self.store,
        ):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 503)
                self.store.heartbeat()
                self.assertEqual(client.get("/health").status_code, 200)
                request = {"request_id": "key", **self.parameters}
                first = client.post("/jobs", json=request)
                self.assertEqual(first.status_code, 202)
                second = client.post("/jobs", json=request)
                self.assertEqual(
                    first.json()["job_id"], second.json()["job_id"])
                self.assertEqual(
                    client.post(
                        "/jobs", json={**request, "request_id": "other"},
                    ).status_code, 409,
                )
                self.assertEqual(
                    client.get("/jobs/missing").status_code, 404,
                )
                self.assertEqual(len(client.get("/jobs").json()["jobs"]), 1)
                self.assertEqual(
                    client.post("/jobs", json={"test_size": 0.2}).status_code,
                    422,
                )

    def test_inference_api_has_no_training_routes(self):
        from coffee_mlflow_api.inference_app import app
        from coffee_mlflow_api.serving import ServingService
        from coffee_mlflow_api.training import TrainingService

        self.assertNotIn("/train", app.openapi()["paths"])
        self.assertNotIn("/jobs", app.openapi()["paths"])
        self.assertFalse(hasattr(ServingService, "train"))
        self.assertFalse(hasattr(TrainingService, "predict_from_csv"))


if __name__ == "__main__":
    unittest.main()
