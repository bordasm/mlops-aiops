"""Submit an idempotent training job and poll it across service boundaries."""

from __future__ import annotations

import json
import logging
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class RetrainingError(RuntimeError):
    """The training job failed, timed out or could not be verified."""


def _request(
    base_url: str, path: str, *, payload: dict | None = None,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RetrainingError(
            f"API {path} failed (HTTP {error.code}): {detail}"
        ) from error
    except (URLError, TimeoutError, OSError, ValueError) as error:
        raise RetrainingError(
            f"API {path} request failed: {error}. Inspect the job; "
            "reuse the same request_id to recover an accepted request."
        ) from error
    if not isinstance(result, dict):
        raise RetrainingError(f"API {path} returned a non-object JSON result.")
    return result


def retrain_coffee_model(request_id: str) -> dict:
    """Wait for a persistent job, without repeating uncertain training."""
    training_url = os.environ.get(
        "COFFEE_TRAINING_URL", "http://training-api:8001",
    )
    inference_url = os.environ.get("COFFEE_API_URL", "http://api:8000")
    try:
        job = _request(training_url, "/jobs", payload={
            "request_id": request_id, "test_size": 0.2, "random_state": 42,
        })
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RetrainingError("Training service returned no job_id.")
        deadline = time.monotonic() + 1800
        while job.get("status") in ("queued", "running"):
            if time.monotonic() >= deadline:
                raise RetrainingError(
                    f"Job {job_id} exceeded the polling deadline; it may "
                    "still be running. Inspect the job before retrying."
                )
            time.sleep(5)
            job = _request(training_url, f"/jobs/{job_id}")
            if job.get("job_id") != job_id:
                raise RetrainingError("Polling returned a different job_id.")
        if job.get("status") != "succeeded":
            raise RetrainingError(
                f"Training job {job_id} failed: {job.get('error', job)}"
            )
        result = job.get("result")
        if not isinstance(result, dict):
            raise RetrainingError(f"Job {job_id} has no training result.")
        for field in (
            "training_run_id", "registered_model_name", "model_version",
            "model_alias", "best_model",
        ):
            if not isinstance(result.get(field), str) or not result[field]:
                raise RetrainingError(
                    f"Job {job_id} result is missing '{field}'."
                )
        if result["model_alias"] != "champion":
            raise RetrainingError("Job did not promote the champion alias.")
        model = _request(inference_url, "/model")
        for field in (
            "registered_model_name", "model_version", "model_alias",
        ):
            if not result.get(field) or model.get(field) != result[field]:
                raise RetrainingError(
                    f"Champion verification failed for '{field}'."
                )
        summary = {
            field: result[field] for field in (
                "training_run_id", "registered_model_name",
                "model_version", "best_model",
            )
        }
        LOGGER.info("Daily retraining job %s completed: %s", job_id, summary)
        return {"job_id": job_id, **summary}
    except RetrainingError:
        LOGGER.exception("Daily coffee model retraining failed.")
        raise
