"""Run inside training-api against an isolated Compose project."""

import json
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pandas as pd
from mlflow import MlflowClient


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    query = Request(
        base + path, data=data, headers={"Content-Type": "application/json"},
    )
    with urlopen(query, timeout=30) as response:
        return json.load(response)


training = "http://training-api:8001"
inference = "http://api:8000"
assert request(training, "/health")["worker_ready"]
assert request(inference, "/health")["status"] == "ok"
try:
    request(inference, "/train", {})
except HTTPError as error:
    assert error.code == 404
else:
    raise AssertionError("Inference unexpectedly exposes training.")

state_path = Path("/app/jobs/microservice-smoke.json")
if state_path.exists():
    parameters = json.loads(state_path.read_text())
else:
    parameters = {
        "request_id": "smoke:" + str(uuid4()),
        "test_size": 0.2, "random_state": 42,
    }
    state_path.write_text(json.dumps(parameters))

start = time.monotonic()
job = request(training, "/jobs", parameters)
elapsed = time.monotonic() - start
assert elapsed < 5, f"Job submission blocked for {elapsed} seconds"
duplicate = request(training, "/jobs", parameters)
assert duplicate["job_id"] == job["job_id"]
print("Asynchronous/idempotent submission:", round(elapsed, 3), "seconds")

deadline = time.monotonic() + 1200
while job["status"] in ("queued", "running"):
    assert request(training, "/health")["status"] == "ok"
    assert request(inference, "/health")["status"] == "ok"
    if time.monotonic() > deadline:
        raise TimeoutError(job)
    time.sleep(3)
    job = request(training, "/jobs/" + job["job_id"])
assert job["status"] == "succeeded", job
assert len(job["result"]["all_models"]) == 3
client = MlflowClient(tracking_uri="http://mlflow:5000")
trained = client.get_model_version(
    job["result"]["registered_model_name"], job["result"]["model_version"],
)
assert trained.tags["training_run_id"] == job["result"]["training_run_id"]
model = request(inference, "/model")
champion = client.get_model_version_by_alias(
    model["registered_model_name"], "champion",
)
assert model["model_version"] == champion.version
assert len(model["feature_columns"]) == 31
print("All candidates trained; remote champion:", model["model_name"])

current = pd.read_csv("/app/data/coffee_value_benchmark.csv").head(20)
boundary = "coffee-" + uuid4().hex
content = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="sample.csv"\r\n'
    "Content-Type: text/csv\r\n\r\n"
    + current.to_csv(index=False)
    + f"\r\n--{boundary}--\r\n"
).encode()
query = Request(
    inference + "/predict", data=content,
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
)
with urlopen(query, timeout=120) as response:
    prediction = json.load(response)
assert prediction["record_count"] == 20
assert len(prediction["predictions"]) == 20
assert prediction["model_version"] == model["model_version"]
assert "eval_r2" in prediction["evaluation"]
experiment = client.get_experiment_by_name("coffee-quality-training")
assert experiment.artifact_location.startswith("mlflow-artifacts:")
with tempfile.TemporaryDirectory() as directory:
    artifact = client.download_artifacts(
        prediction["inference_run_id"],
        "coffee_quality_predictions.csv", directory,
    )
    output = pd.read_csv(artifact)
assert len(output) == 20
assert not Path("/app/mlflow-artifacts").exists()
print("Remote HTTP artifacts and labeled predictions verified.")
