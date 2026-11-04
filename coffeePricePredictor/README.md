# Coffee quality: Docker, Streamlit, Evidently and Airflow

The application predicts **`coffee_quality_score`**, not a coffee price.
The Docker stack reuses the existing MLflow REST API and the training
utilities in `productionReadyCode/src`. The notebooks remain unchanged.

## Start with Docker Compose

Requirements: Docker Desktop or Rancher Desktop running **Linux containers**,
Docker Compose, and enough memory for three model candidates and monitoring.
The first build downloads Python dependencies and can take several minutes.

From this directory in PowerShell:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f api dashboard mlflow
```

| Service | Address |
| --- | --- |
| Streamlit dashboard | <http://localhost:8501> |
| REST API / Swagger | <http://localhost:8000/docs> |
| MLflow experiments and registry | <http://localhost:5000> |
| Airflow daily retraining | <http://localhost:8080> |

1. Open Streamlit. A new model store initially has **no champion**. Airflow can
   perform its first daily training automatically for the latest due slot.
2. For manual training, wait until the API is idle. On **Training**, select the
   test fraction and random seed, then select **Train models**.
   RandomForest, XGBoost and LightGBM are trained by the API.
   The best test R-squared selects the registered `champion`.
3. Review the model metadata, train/test metrics and candidate comparison.
   The last dashboard training comparison is kept in the browser session;
   the complete durable history is available in MLflow.
4. On **Inference and monitoring**, download the sample CSV or upload your
   own UTF-8 CSV containing the 31 model features. Select **Predict CSV**.
   Predictions and their MLflow run ID appear, with a CSV download.
5. Generate **Evidently data drift and quality** independently of inference.
   After prediction, a CSV with at least two known `coffee_quality_score`
   labels also supports the **Evidently regression report**.
6. Reports are embedded in Streamlit and downloadable as HTML or JSON.
   **Saved reports** lists previous reports, including after container restarts.

Manual training and report generation are explicit button actions: refreshing
the page does not repeat training, inference or report creation. Airflow also
retrains once per day as described below. Use **Refresh
status** to pick up externally completed training. Uploaded CSVs are limited
to 20 MB, matching the API. Training requests have a 30-minute client timeout;
if a timeout occurs, the API may still be training. Refresh status before retrying.
Changing the input, reference file or champion clears stale dashboard results.

## Persistent storage and compatibility

Compose creates two named volumes (prefixed by `coffee-quality`):

- `mlflow-data`: `/app/artifacts/mlflow.db` and `/app/artifacts/mlruns`;
  shared by the API and MLflow UI at the **same Linux absolute path**.
- `monitoring-reports`: `/app/reports`, containing Evidently HTML/JSON outputs.
- `airflow-state`: `/opt/airflow/state`, containing Airflow's SQLite metadata,
  scheduler state, task logs and generated local login credentials.

The host's existing [artifacts](artifacts/) are **not mounted, copied,
migrated or modified**. They contain Windows absolute artifact URIs that
are not portable to Linux. The container starts with a separate empty
registry; its first model is trained by Airflow, the dashboard or `POST /train`.
Host notebook training still writes to the original host store and does
not update the container's champion.

The image excludes notebooks, existing models, the host SQLite database
and unrelated files. It contains the API, reusable training code, dashboard,
benchmark CSV and inference sample. Services run as a non-root user.
Only the four HTTP ports are published, all bound to host loopback.

```powershell
docker compose stop
docker compose start
# Removes containers/network, but preserves models and reports:
docker compose down
```

Do **not** add `--volumes` to `docker compose down` unless you deliberately
want to delete all container training history, models, reports and Airflow
state. Back up all three named volumes while the services are stopped.

The SQLite deployment uses one API worker and one MLflow worker, suitable
for this local application. The existing in-process training lock prevents
concurrent training within that API. It is not a distributed lock: do not
scale API workers or train concurrently from separate processes against
the same store. Multi-host production deployment would require a shared
database, artifact service, authentication, TLS and distributed job control.
Do not expose these unauthenticated local services to a public network.

## Daily retraining with Airflow

`docker compose up --build -d` also starts an independent
**Apache Airflow 3.3.2** container. The official image runs `airflow standalone`
(API/UI, scheduler, DAG processor and local executor) with persistent SQLite
state. This is a local, single-node setup, not a production Airflow cluster;
Airflow's dependencies are deliberately isolated from the ML/Streamlit image.
Native Windows Airflow installation is not supported or required.

The `coffee_daily_retraining` DAG is enabled automatically on first creation:

- **Schedule:** `0 2 * * *`, daily at **02:00 Europe/Budapest**.
  The timetable uses the timezone rather than a fixed UTC offset and follows
  Airflow's daylight-saving rules. On the spring clock-change day, the
  nonexistent 02:00 slot moves forward; on the autumn change, the repeated
  clock hour does not intentionally create a second daily training interval.
- **No historical catchup:** `catchup=False`. The explicit trigger timetable
  runs only the latest due daily slot when starting/restarting after downtime,
  rather than replaying each missed day. A fresh stack can therefore train
  once immediately for the latest due slot; subsequent runs follow the daily
  schedule. Keep Docker and the host running for scheduled executions.
- **Concurrency:** one active DAG run/task. The API's existing lock also
  rejects concurrent manual training with HTTP 409.
- **Workflow:** check `GET /health`, call `POST /train` once with
  `{"test_size":0.2,"random_state":42}`, then verify the serving champion via
  `GET /model`. It reuses the API's configured benchmark, trains all three
  candidates, logs to the existing container MLflow volume and promotes the
  highest test R-squared model, just like manual API/dashboard training.
  This is a daily retraining **pipeline**, not three separate daily schedules.
- **Visibility:** the Airflow task log and small XCom identify the MLflow
  training run, model name and version. Full candidate metrics and history
  remain in MLflow. Select **Refresh status** in Streamlit after completion.
- **Failure safety:** connection errors, HTTP 409/5xx, invalid JSON and
  champion mismatches fail visibly. Automatic retries are disabled because
  `/train` is synchronous and not idempotent: a timed-out request may still
  complete on the API. The HTTP read timeout is 30 minutes, task timeout
  35 minutes, DAG-run timeout 40 minutes. A failed daily run is not silently
  skipped or repeatedly resubmitted.

Open <http://localhost:8080>. The local username is `admin`; the password is
generated at runtime, never committed. Retrieve it locally:

```powershell
docker compose exec airflow cat /opt/airflow/state/simple_auth_manager_passwords.json.generated
```

In the UI, locate `coffee_daily_retraining`, inspect its schedule and task
logs, and use **Trigger DAG** for an explicit manual run. Alternatively:

```powershell
docker compose exec airflow airflow dags list
docker compose exec airflow airflow dags list-import-errors
docker compose exec airflow airflow dags trigger coffee_daily_retraining
docker compose logs -f airflow
```

Manual triggers and clearing failed tasks can create extra training runs.
Before retrying a failed task, inspect API health and MLflow to confirm the
previous request did not already complete; exactly-once execution across
network failures is not guaranteed by the existing API. Avoid manual
dashboard training around the scheduled time. Pausing the DAG in the UI is
persisted across restarts; auto-enable applies only on initial DAG creation.
To change the schedule, edit
`airflow/dags/coffee_daily_retraining.py` and rebuild the Airflow service.
The host notebooks/artifacts are not mounted into Airflow.

To disable the scheduler temporarily without affecting the other services:

```powershell
docker compose stop airflow
docker compose start airflow
```

Airflow's standalone UI and scheduler are health-checked together. If the
Rancher Desktop UI is running but Windows `docker info` still fails, verify
the Docker/Moby backend and Windows Docker integration: a running desktop
application alone does not guarantee that the Windows named pipe is usable.

## What monitoring means

- **Reference:** the entire `data/coffee_value_benchmark.csv` bundled with
  the image; **current:** the uploaded CSV. Rebuild to change the bundled data.
- Data drift and quality use only training features. IDs, the target,
  `quality_grade` and `price_usd_per_kg` are excluded, matching training.
  Numeric/categorical types are explicitly mapped from the reference.
  Missing values remain visible to the quality report; they are not imputed
  by the dashboard. The model still uses its trained imputers for inference.
  Entirely empty columns in either dataset cannot be statistically tested:
  they remain in the quality report, are excluded from drift, and are listed
  in an explicit warning and the JSON `coffee_monitoring` metadata.
- Evidently uses its default per-feature drift tests and thresholds.
  The slider controls the **fraction of eligible drifted features** required to
  declare dataset drift (default 50%), not a per-feature p-value.
- At least two rows are required for drift reports. Small batches have
  low statistical reliability; the UI warns below 100 rows.
- Regression is evaluated on labeled **current** rows only, not against
  predictions on the training reference. Missing labels are excluded and
  their count is reported. Invalid numeric/infinite values are rejected.
  Without labels, drift is not evidence that prediction accuracy declined.
- The reference is not an independent test set. For unbiased performance
  measurement, upload held-out or production data with observed labels.
- Reports can contain data distributions and example values. Treat the
  persisted volume and downloaded reports as application data. No report
  is uploaded to Evidently Cloud. Telemetry is disabled in the Docker image.

The API also supports a single labeled inference row: MAE/RMSE are returned,
but R-squared is omitted because it is undefined for one observation.

## Run locally without Docker

The existing virtual environment can be used. From this directory:

```powershell
& C:\Users\bordasm\.venv\Scripts\python.exe -m pip install -r docker\requirements.txt
& C:\Users\bordasm\.venv\Scripts\python.exe api_mlflow\main.py
```

In another PowerShell terminal:

```powershell
$env:COFFEE_API_URL = "http://127.0.0.1:56"
& C:\Users\bordasm\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.maxUploadSize=20
```

Unlike Docker, this local API uses the existing host artifact store.
The existing API binds to port 56 locally; the container uses port 8000.
Start the local MLflow UI as described in [the API guide](api_mlflow/README.md).

| Environment variable | Purpose / default |
| --- | --- |
| `COFFEE_DATA_PATH` | API training and dashboard reference CSV; bundled benchmark |
| `COFFEE_ARTIFACTS_DIR` | API artifact directory; project `artifacts` |
| `MLFLOW_TRACKING_URI` | API tracking/registry URI; SQLite in the artifact directory |
| `COFFEE_API_URL` | Dashboard API address; `http://127.0.0.1:56` locally |
| `COFFEE_REPORTS_DIR` | Dashboard report output; `artifacts/reports` locally |
| `COFFEE_MLFLOW_UI_URL` | Browser-visible MLflow link; `http://localhost:5000` |

Set environment variables before starting the relevant process. The
dashboard intentionally calls the REST API rather than loading models or
opening the MLflow database itself.

## Validation

Tests use standard-library `unittest` and the installed application packages.
They generate actual Evidently reports and train a small model in a temporary
MLflow store; they do not train into or modify the original artifacts.

```powershell
& C:\Users\bordasm\.venv\Scripts\python.exe -m pip install -r tests\requirements.txt
& C:\Users\bordasm\.venv\Scripts\python.exe -m unittest discover -s tests -v
& C:\Users\bordasm\.venv\Scripts\python.exe -m pycodestyle dashboard tests api_mlflow\coffee_mlflow_api\config.py api_mlflow\coffee_mlflow_api\service.py
docker compose config --quiet
```

The standard Windows test suite also validates the Airflow HTTP training
helper (including failure/no-retry behavior) using a real local HTTP test
server, without importing Airflow. Actual DAG parsing, winter/summer timezone
scheduling, clock changes and no-historical-backfill behavior are checked
in the Linux Airflow image:

```powershell
docker compose build airflow
docker compose run --rm --no-deps airflow python -m unittest discover -s /opt/airflow/tests -v
```

After startup, verify the real container stack:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest http://localhost:8501/_stcore/health
Invoke-WebRequest http://localhost:5000/health
Invoke-RestMethod -Method Post http://localhost:8000/train -ContentType "application/json" -Body '{"test_size":0.2,"random_state":42}'
curl.exe -f -F "file=@data\coffee_inference_sample.csv" http://localhost:8000/predict
```

Stop/start the stack and repeat `GET /model` to verify the champion persists.
If Docker reports a Hyper-V socket/backend timeout, start or repair the
Linux container engine before building. If package downloads fail, check
the machine's pip/proxy settings rather than changing repository code.
