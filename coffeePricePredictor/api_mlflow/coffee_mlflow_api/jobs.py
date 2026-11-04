"""Service-owned durable training jobs with atomic deduplication."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4


class JobConflictError(RuntimeError):
    """Another job is active, or an idempotency key has different inputs."""


class JobStore:
    """SQLite belongs only to the training API/worker, not other services."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE NOT NULL,
                    parameters TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result TEXT,
                    error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_training
                ON jobs ((1)) WHERE status IN ('queued', 'running');
                CREATE TABLE IF NOT EXISTS worker (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    heartbeat TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Close connections explicitly, including on Windows."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def decode(row: sqlite3.Row) -> dict:
        result = dict(row)
        for field in ("parameters", "result"):
            if result[field] is not None:
                result[field] = json.loads(result[field])
        return result

    def submit(self, request_id: str, parameters: dict) -> dict:
        """Return the same durable job for repeated identical requests."""
        encoded = json.dumps(parameters, sort_keys=True, allow_nan=False)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT * FROM jobs WHERE request_id=?", (request_id,),
            ).fetchone()
            if previous:
                if previous["parameters"] != encoded:
                    raise JobConflictError(
                        "This request_id was already used "
                        "with different inputs."
                    )
                return self.decode(previous)
            active = connection.execute(
                "SELECT job_id FROM jobs WHERE status IN ('queued','running')",
            ).fetchone()
            if active:
                raise JobConflictError(
                    f"Training job {active['job_id']} is already active."
                )
            job_id = str(uuid4())
            now = self.now()
            connection.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,NULL,NULL)",
                (job_id, request_id, encoded, "queued", now, now),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            return self.decode(row)

    def get(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            return None if row is None else self.decode(row)

    def list_jobs(self) -> list[dict]:
        with self.connection() as connection:
            return [
                self.decode(row) for row in connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50",
                )
            ]

    def claim(self) -> dict | None:
        """Atomically claim queued work for the single training worker."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued' LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET status='running',updated_at=? WHERE job_id=?",
                (self.now(), row["job_id"]),
            )
            return self.decode(row)

    def finish(
        self, job_id: str, *, result: dict | None = None,
        error: str | None = None,
    ) -> None:
        if (result is None) == (error is None):
            raise ValueError("Specify exactly one training result or error.")
        encoded = (
            None if result is None else json.dumps(result, allow_nan=False)
        )
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status=?,result=?,error=?,updated_at=? "
                "WHERE job_id=? AND status='running'",
                (
                    "failed" if error else "succeeded",
                    encoded, error, self.now(), job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Job {job_id} is not running.")

    def recover_interrupted(self) -> None:
        """Never replay training with an uncertain MLflow outcome."""
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status='failed',error=?,updated_at=? "
                "WHERE status='running'",
                (
                    "Training worker interrupted. Inspect MLflow before "
                    "submitting a new request; no automatic retry.",
                    self.now(),
                ),
            )

    def heartbeat(self) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO worker VALUES (1,?) ON CONFLICT(id) "
                "DO UPDATE SET heartbeat=excluded.heartbeat",
                (self.now(),),
            )

    def worker_ready(self) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT heartbeat FROM worker WHERE id=1",
            ).fetchone()
        if row is None:
            return False
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(row[0])
        ).total_seconds()
        return 0 <= age < 30
