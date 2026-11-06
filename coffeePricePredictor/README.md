# Coffee quality microservices

The application predicts **`coffee_quality_score`**, not price.
Training, serving and the monitoring UI are independently built/deployed
services. They communicate through HTTP; neither serving nor training opens
the MLflow database or mounts its artifact filesystem.

## Architecture

```text
Streamlit/Evidently ---- GET /model, POST /predict ----> Inference API
       |
       +---- POST /jobs, GET /jobs/{id} ----> Training API
                                              |
                                  service-owned durable job store
                                              |
Airflow ---- POST /jobs, poll status ----> Training worker
                                              |
Inference API ------ MLflow HTTP ------> MLflow service <--- Training worker
                                              |
                                    PostgreSQL + artifact volume
```

| Component | Image / responsibility | Local address |
| --- | --- | --- |
| `api` | `coffee-inference:local`: prediction only, no training code/data | <http://localhost:8000/docs> |
| `training-api` | `coffee-training:local`: durable asynchronous job API | <http://localhost:8001/docs> |
| `training-worker` | `coffee-training:local`: isolated training processes | No published port |
| `dashboard` | `coffee-dashboard:local`: Streamlit/Evidently, no model loading | <http://localhost:8501> |
| `mlflow` | `coffee-mlflow:local`: registry, tracking, HTTP artifacts | <http://localhost:5000> |
| `mlflow-db` | PostgreSQL, exclusively owned by MLflow | No published port |
| `airflow` | `coffee-airflow:local`: daily orchestration | <http://localhost:8080> |

Images use separate Docker build targets; only ML-library base layers are
shared between training and inference. Airflow has its own official image.
Services run as non-root except for the PostgreSQL image's standard bootstrap.
All published HTTP ports are bound to host loopback.

## Start

Requirements: running Docker/Moby Linux engine (Docker Desktop/Rancher Desktop)
and Docker Compose. Allow sufficient memory for training, Streamlit and Airflow.
From this directory in PowerShell:

```powershell
# Once: generates a cryptographically random password into ignored local .env.
# Preserves an existing .env and never prints its password.
& .\docker\initialize.ps1
docker compose up --build -d
docker compose ps
```

`MLFLOW_DB_PASSWORD` is mandatory. The generated value is URL-safe hexadecimal.
Keep `.env` private and backed up; it is not included in images or committed.
Do not change the database password in `.env` for an initialized PostgreSQL
volume without also rotating the PostgreSQL user's password.

If Rancher is running but Windows `docker info` fails, its Windows named-pipe
integration may be unavailable. Repair that integration for normal PowerShell
commands. The Linux engine can be checked without changing global settings:

```powershell
wsl --distribution rancher-desktop --exec docker info
```

If Rancher's Linux Compose plugin is not discovered automatically, use its
bundled executable from this directory (default Rancher installation):

```powershell
$compose = "/mnt/c/Program Files/Rancher Desktop/resources/resources/linux/docker-cli-plugins/docker-compose"
wsl --distribution rancher-desktop --cd (Get-Location).Path --exec $compose up --build -d
```

The first start creates an empty **new** central registry. Airflow may submit
the latest due training job immediately. Until a champion exists, prediction
returns HTTP 503; the service itself is healthy.

## Existing data and migration

The old `mlflow-data` volume and the host [artifacts](artifacts/) are **not**
mounted, migrated or deleted. Host notebooks still use their original local
store; they do not update the microservice registry. Retrain to populate the
new registry. Existing monitoring reports and Airflow state retain their
original volume names. The prior monolithic image is no longer used by Compose.

New service-owned volumes:

- `mlflow-postgres`: MLflow metadata and model registry.
- `mlflow-http-artifacts`: only MLflow mounts this volume; clients upload and
  download through its artifact HTTP endpoints (`mlflow-artifacts:` URIs).
- `training-jobs`: only the training API and worker mount this SQLite job store.
- `monitoring-reports`: only the dashboard mounts report HTML/JSON files.
- `airflow-state`: Airflow metadata, task logs and generated login credentials.

```powershell
docker compose stop
docker compose start
docker compose down
```

These commands preserve volumes. **Do not use `down --volumes`** unless deletion
of the new system's models, jobs, reports and scheduler history is intended.
Back up the volumes and `.env` with the relevant services stopped.
No automated deletion or migration of the old volume is performed.

## Asynchronous training

The inference API exposes `/health`, `/model` and `/predict` only.
**Breaking container API change:** `POST http://localhost:8000/train` is removed.
Submit a job to the independent training API instead:

```powershell
$request = @{
    request_id = [guid]::NewGuid().ToString()
    test_size = 0.2
    random_state = 42
} | ConvertTo-Json
$job = Invoke-RestMethod -Method Post http://localhost:8001/jobs `
    -ContentType "application/json" -Body $request
Invoke-RestMethod "http://localhost:8001/jobs/$($job.job_id)"
Invoke-RestMethod http://localhost:8001/jobs
```

- `POST /jobs` returns **202** with `job_id`, `request_id` and state immediately.
- `GET /jobs/{id}` returns `queued`, `running`, `succeeded` or `failed`;
  completed jobs contain `result` or `error`. Unknown jobs return 404.
- `GET /jobs` lists the latest 50 jobs inside a `jobs` object.
- Reusing a `request_id` with identical parameters returns the **same job**,
  even after success/failure or API restart. Reusing it with different inputs
  returns 409. Use a **new** key only when a new training run is intended.
- An atomic constraint admits one queued/running job. Another distinct request
  returns 409 instead of creating concurrent promotions.
- The worker trains RandomForest, XGBoost and LightGBM using the production
  pipeline and full configured benchmark. It selects maximum test R-squared,
  logs candidates to MLflow and promotes `champion`.
- Each job runs in its own ML process, without blocking the training HTTP
  server or inference service. MLflow records `training_job_id` for inspection.
- Queued jobs survive restart. Interrupted running jobs become explicit
  failures, **not automatic retries**: the champion may have been promoted
  before the interruption. Inspect MLflow before submitting a new key.
- The worker has a singleton filesystem lock and readiness heartbeat; the
  training API reports 503 if that worker is not ready. Training has a
  30-minute worker limit. Job status remains queryable during worker outages.

The job queue is owned by one logical training service. It is a single-node
SQLite implementation, not a distributed broker. Keep one worker replica.
For multi-host training, replace it with a service-owned database/broker and
distributed claiming; serving can already be deployed independently.

## Dashboard and monitoring

Open <http://localhost:8501>. The **Training** tab submits jobs, lists persisted
states/errors and displays candidate comparison when a selected job succeeds.
It does not hold a long-running training request open. **Refresh status**
updates job/model state. A failed submission retains its request key for safe
retry with the same parameters. Job history remains available after browser
restart. No polling refresh creates another training job.

The inference/monitoring tab still supports UTF-8 CSV uploads (max. 20 MB),
prediction CSV downloads, embedded Evidently reports and HTML/JSON downloads.
Only the 31 training features are required; extra columns are ignored by
inference. `record_id` is echoed. Uploading known `coffee_quality_score` labels
also enables current-data regression evaluation.

- Reference: the bundled full training benchmark; current: uploaded data.
- Drift excludes IDs, target, `quality_grade` and `price_usd_per_kg`, matching
  training. Numeric/categorical mappings come from the reference.
- Missing values remain in quality reports. Entirely empty columns are
  excluded from drift with a warning and JSON `coffee_monitoring` metadata.
- The slider controls the fraction of eligible drifted features required
  for dataset drift (default 50%), not a per-feature p-value. Evidently's
  default per-feature tests/thresholds are used.
- At least two rows are required; small batches give unreliable drift
  estimates (UI warns below 100 rows).
- Regression reports evaluate finite labeled **current** rows only, with
  missing-label exclusions counted. At least two labels are required for the
  report; the inference API returns only MAE/RMSE for one finite label.
- The full training reference is not an independent performance test set.
  Upload held-out/production labels for unbiased evaluation.
- Reports can contain sample values. Treat them as application data; no
  reports are uploaded to Evidently Cloud. Container telemetry is disabled.

Changing the upload, reference file or champion clears stale inference/report
results. The inference service loads the exact registry version selected for
the request and notices champion updates without restarting.

## Airflow daily retraining

The existing `coffee_daily_retraining` DAG remains **daily 02:00
Europe/Budapest**, `catchup=False`, one active run/task. Airflow follows local
daylight-saving transitions (one daily slot); downtime does not replay each
missed day, but the latest due slot may run on startup.

The DAG now submits `POST /jobs`, polls `GET /jobs/{id}` and verifies the
champion through the inference API. Its request key is a deterministic hash
of the Airflow run ID, so clearing/retrying that same run cannot duplicate the
training job. A new manually triggered DAG run creates a new key.
Failed jobs/polling timeouts fail visibly. No automatic task retries are set.
The polling deadline is 30 minutes, task timeout 35, DAG-run timeout 40.

Open <http://localhost:8080>; username `admin`. Retrieve the generated password:

```powershell
docker compose exec airflow cat /opt/airflow/state/simple_auth_manager_passwords.json.generated
docker compose exec airflow airflow dags list-import-errors
docker compose exec airflow airflow dags trigger coffee_daily_retraining
docker compose logs -f airflow training-worker
```

Airflow remains single-node standalone for local use. Pausing its DAG is
persisted; auto-enable applies only to first DAG creation. Stop it independently
with `docker compose stop airflow`.

## Deployment boundaries and scaling

Rebuild/update individual services without bundling the others:

```powershell
docker compose up -d --build --no-deps api
docker compose up -d --build --no-deps training-api training-worker
docker compose up -d --build --no-deps dashboard
```

Inference needs only MLflow HTTP and its ML libraries: no training job database
or artifact volume. Multiple inference replicas can use the same central
registry; each caches its own champion. Compose's fixed published port prevents
`--scale api=2` as written; use a reverse proxy/load balancer and remove the
fixed host port for replicas. Training remains a single promoter.

This is service-oriented/microservice deployment, **not** a public production
platform: API authentication, TLS, managed backups, load balancing and
multi-host job control still require deployment infrastructure. Do not expose
the local unauthenticated ML endpoints publicly.

## Local compatibility and tests

The original local `api_mlflow/main.py` API retains synchronous `/train` through
a backward-compatible facade over the extracted training/serving modules.
The production CLI and notebooks remain supported. It is deliberately **not**
the application used by the microservice Compose stack.

For the split services without Docker, set `MLFLOW_TRACKING_URI` to a running
central MLflow HTTP service, `COFFEE_JOBS_PATH` to a writable service-owned
SQLite path, and `COFFEE_DATA_PATH` to the benchmark. Start these in separate
terminals with the API package on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path api_mlflow).Path
python -m uvicorn coffee_mlflow_api.training_app:app --port 8001
python -m coffee_mlflow_api.worker
python -m uvicorn coffee_mlflow_api.inference_app:app --port 8000
```

For Streamlit set `COFFEE_API_URL=http://127.0.0.1:8000` and
`COFFEE_TRAINING_URL=http://127.0.0.1:8001`. Its report/reference path overrides
remain `COFFEE_REPORTS_DIR`, `COFFEE_DATA_PATH`; the browser MLflow link is
`COFFEE_MLFLOW_UI_URL`.

```powershell
& C:\Users\bordasm\.venv\Scripts\python.exe -m pip install -r tests\requirements.txt
& C:\Users\bordasm\.venv\Scripts\python.exe -m unittest discover -s tests -v
docker compose config --quiet
docker compose run --rm --no-deps airflow python -m unittest discover -s /opt/airflow/tests -v
```

Tests cover legacy training behavior, real Evidently outputs, Streamlit,
job concurrency/durability/idempotency, API boundaries and asynchronous Airflow
polling. Existing artifacts are not used as writable test storage.

On the `cloud_run` branch, GitHub Actions also runs the application tests,
PEP 8 checks, Compose configuration validation, image builds and native
Airflow DAG tests. A manually dispatched run on `cloud_run` publishes the
inference, training, dashboard, MLflow and Airflow images to Artifact Registry
only after these checks pass. The separately published images are tagged with
the Git commit SHA, not `latest`.

### Image publication prerequisites

- Enable the Artifact Registry API and create an existing **Docker-format**
  `mlsecops` repository in the region specified by the `GCP_REGION` secret.
- Configure GitHub Actions secrets `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_SA_KEY`
  (the service-account JSON key) and `SERVICE_NAME`. The service account needs
  Artifact Registry Writer on the repository and permission to read its
  metadata. `SERVICE_NAME` is a lowercase image-name prefix; for example, the
  inference image is `REGION-docker.pkg.dev/PROJECT/mlsecops/PREFIX-inference:SHA`.
- Create the GitHub environment `cloud-run-publication` and configure required
  reviewers and branch restrictions if approval is desired. An environment
  name alone does not enforce approval. Publication uses a service-account key;
  protect and rotate it, and prefer Workload Identity Federation when available.
- GitHub requires a manually dispatchable workflow to exist on the repository's
  default branch before it can be run manually. Until it is present there,
  this branch-only workflow cannot be manually dispatched via GitHub. Once it
  exists there, choose `cloud_run` as the ref when starting the workflow.

This workflow **does not deploy to Cloud Run**. The images are the existing
local Compose images: SQLite jobs, a continuously running worker, local report
and artifact volumes, network addresses and authentication still need a
Cloud Run-compatible redesign. Airflow is intended to move to Cloud Composer,
not run as the standalone Airflow image on Cloud Run. No public access or
Cloud Run infrastructure is configured by image publication.

For an **isolated test project only**, the end-to-end smoke test submits real
training, checks idempotency, predicts labeled rows and downloads HTTP artifacts:

```powershell
Get-Content tests\microservice_smoke.py -Raw |
    docker compose exec -T training-api python -
```

It persists a test request key in the test job volume. Repeating it after service
restart verifies the original job without retraining; a newer champion, such as
one promoted by Airflow, is checked separately.

***********************************************************************************

This project was developed with the help of GitHub Copilot.
