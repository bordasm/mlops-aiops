"""Trigger and verify coffee retraining through the existing REST API."""

from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class RetrainingError(RuntimeError):
    """The API was unavailable, training failed or its result was invalid."""


def _request(
    base_url: str,
    path: str,
    *,
    payload: dict | None = None,
) -> dict:
    """Perform one request without retrying a possibly completed training."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(
            request, timeout=30 if payload is None else 1800,
        ) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RetrainingError(
            f"API {path} failed (HTTP {error.code}): {detail}. "
            "Check API/MLflow state before manually retrying training."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RetrainingError(
            f"API {path} connection failed: {error}. Training may still "
            "be running or completed; check API/MLflow before retrying."
        ) from error
    except (ValueError, UnicodeDecodeError) as error:
        raise RetrainingError(
            f"API {path} returned invalid JSON. Check MLflow before retrying."
        ) from error
    if not isinstance(result, dict):
        raise RetrainingError(f"API {path} returned a non-object JSON result.")
    return result


def retrain_coffee_model() -> dict:
    """Log a completed API training run and verify the serving champion."""
    base_url = os.environ.get("COFFEE_API_URL", "http://api:8000")
    try:
        health = _request(base_url, "/health")
        if (
            health.get("status") != "ok"
            or not isinstance(health.get("training_in_progress"), bool)
        ):
            raise RetrainingError("API health response is invalid.")
        if health["training_in_progress"]:
            raise RetrainingError(
                "Another training run is active. No training was submitted."
            )
        result = _request(
            base_url, "/train",
            payload={"test_size": 0.2, "random_state": 42},
        )
        for field in (
            "training_run_id", "registered_model_name", "model_version",
            "model_alias", "best_model",
        ):
            if not isinstance(result.get(field), str) or not result[field]:
                raise RetrainingError(
                    f"Training result is missing '{field}'. "
                    "Inspect MLflow before retrying."
                )
        if result["model_alias"] != "champion":
            raise RetrainingError(
                "Training did not return the champion alias."
            )
        model = _request(base_url, "/model")
        for field in ("registered_model_name", "model_version", "model_alias"):
            if model.get(field) != result[field]:
                raise RetrainingError(
                    f"Champion verification failed for '{field}'. "
                    "Inspect API/MLflow state; training is not retried."
                )
        summary = {
            field: result[field]
            for field in (
                "training_run_id", "registered_model_name",
                "model_version", "best_model",
            )
        }
        LOGGER.info("Daily retraining completed: %s", json.dumps(summary))
        return summary
    except RetrainingError:
        LOGGER.exception("Daily coffee model retraining failed.")
        raise
