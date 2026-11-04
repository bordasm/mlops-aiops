"""Single durable training worker with isolated per-job ML processes."""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

from coffee_mlflow_api.jobs import JobStore

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Keep queued jobs, fail interrupted work, and publish readiness."""
    logging.basicConfig(level=logging.INFO)
    path = Path(os.environ.get("COFFEE_JOBS_PATH", "/app/jobs/training.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".worker.lock", timeout=0):
        store = JobStore(path)
        store.recover_interrupted()
        while True:
            store.heartbeat()
            job = store.claim()
            if job is None:
                time.sleep(2)
                continue
            LOGGER.info("Executing training job %s", job["job_id"])
            with subprocess.Popen([
                sys.executable, "-m", "coffee_mlflow_api.job_runner",
                job["job_id"],
            ], env={**os.environ, "COFFEE_JOB_ID": job["job_id"]}) as process:
                started = time.monotonic()
                while process.poll() is None:
                    store.heartbeat()
                    if time.monotonic() - started > 1800:
                        process.kill()
                        process.wait()
                        LOGGER.error("Training job exceeded 30 minutes.")
                        break
                    time.sleep(2)
            current = store.get(job["job_id"])
            if current["status"] == "running":
                message = (
                    f"Training process exited ({process.returncode}) without "
                    "a result. Inspect MLflow before submitting another job."
                )
                LOGGER.error(message)
                store.finish(job["job_id"], error=message)


if __name__ == "__main__":
    main()
