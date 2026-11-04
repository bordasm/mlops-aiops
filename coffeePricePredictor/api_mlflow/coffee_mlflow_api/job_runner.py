"""Execute one claimed job and persist explicit success or failure."""

import logging
import os
import sys
from pathlib import Path

from coffee_mlflow_api.jobs import JobStore
from coffee_mlflow_api.schemas import TrainResponse
from coffee_mlflow_api.training import TrainingService


def run(job_id: str) -> None:
    """Exception boundary for a job: failures are logged and persisted."""
    logging.basicConfig(level=logging.INFO)
    store = JobStore(Path(os.environ["COFFEE_JOBS_PATH"]))
    job = store.get(job_id)
    if job is None or job["status"] != "running":
        raise RuntimeError(f"Job {job_id} is not claimed.")
    try:
        service = TrainingService()
        service.setup()
        result = TrainResponse(
            **service.train(**job["parameters"]),
        ).model_dump()
        store.finish(job_id, result=result)
        logging.info("Training job %s succeeded.", job_id)
    except Exception as error:
        logging.exception("Training job %s failed.", job_id)
        store.finish(
            job_id,
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    run(sys.argv[1])
