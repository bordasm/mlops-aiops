"""HTTP client for the existing coffee quality REST API."""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """A connection, HTTP or response-format error from the API."""


class CoffeeApiClient:
    """Keep predictions and training behind the existing API contract."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 60,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return JSON or raise an explicit, user-facing API error."""
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=(10, timeout),
                **kwargs,
            )
        except requests.Timeout as error:
            raise ApiError(
                "API timeout. Training may still be running; refresh the "
                "status before submitting another request."
            ) from error
        except requests.ConnectionError as error:
            raise ApiError(f"Cannot connect to the API: {error}") from error
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise ApiError(
                f"API returned non-JSON data (HTTP {response.status_code})."
            ) from error
        if not response.ok:
            detail = (
                payload.get("detail", payload)
                if isinstance(payload, dict) else payload
            )
            raise ApiError(f"HTTP {response.status_code}: {detail}")
        if not isinstance(payload, dict):
            raise ApiError("API returned an unexpected JSON structure.")
        return payload

    def predict(self, content: bytes, filename: str) -> dict[str, Any]:
        """Score a CSV through the champion model and log the API run."""
        return self.request(
            "POST",
            "/predict",
            timeout=300,
            files={"file": (filename, content, "text/csv")},
        )
