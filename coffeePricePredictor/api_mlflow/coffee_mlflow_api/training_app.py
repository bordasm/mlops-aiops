"""Asynchronous training API; only the worker imports ML training code."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from coffee_mlflow_api.jobs import JobConflictError, JobStore
from coffee_mlflow_api import config
from coffee_mlflow_api.schemas import TrainRequest

STORE_PATH = Path(os.environ.get(
    "COFFEE_JOBS_PATH", config.ARTIFACTS_DIR / "jobs" / "training.db",
))
app = FastAPI(title="Coffee Training Jobs Service")


@lru_cache
def get_store() -> JobStore:
    """Initialize service-owned state only when the API first uses it."""
    return JobStore(STORE_PATH)


class TrainingJobRequest(TrainRequest):
    """The client supplies a stable key so network retries cannot duplicate."""

    request_id: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[\w.:-]+$"),
    ]


class JobResponse(BaseModel):
    job_id: str
    request_id: str
    parameters: dict
    status: str
    created_at: str
    updated_at: str
    result: dict | None
    error: str | None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


@app.get("/health")
def health() -> dict:
    store = get_store()
    ready = store.worker_ready()
    if not ready:
        raise HTTPException(503, "Training worker is not ready.")
    active = any(
        job["status"] in ("queued", "running") for job in store.list_jobs()
    )
    return {
        "status": "ok", "worker_ready": ready,
        "training_in_progress": active,
    }


@app.post("/jobs", status_code=202, response_model=JobResponse)
def submit(request: TrainingJobRequest) -> dict:
    """Accept durable work; never execute training on the HTTP thread."""
    try:
        return get_store().submit(
            request.request_id,
            request.model_dump(exclude={"request_id"}),
        )
    except JobConflictError as error:
        raise HTTPException(409, str(error)) from error


@app.get("/jobs", response_model=JobListResponse)
def jobs() -> dict:
    return {"jobs": get_store().list_jobs()}


@app.get("/jobs/{job_id}", response_model=JobResponse)
def job(job_id: str) -> dict:
    result = get_store().get(job_id)
    if result is None:
        raise HTTPException(404, "Training job not found.")
    return result
