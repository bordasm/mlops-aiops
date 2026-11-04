# Coffee quality: Docker, Streamlit and Evidently

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

1. Open Streamlit. On first startup, the API is healthy but has **no champion**.
2. On **Training**, select the test fraction and random seed, then select
   **Train models**. RandomForest, XGBoost and LightGBM are trained by the API.
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

Training and report generation are explicit button actions: refreshing the
page does not repeat training, inference or report creation. Use **Refresh
status** to pick up externally completed training. Uploaded CSVs are limited
to 20 MB, matching the API. Training requests have a 30-minute client timeout;
if a timeout occurs, the API may still be training. Refresh status before retrying.
Changing the input, reference file or champion clears stale dashboard results.

## Persistent storage and compatibility

Compose creates two named volumes (prefixed by `coffee-quality`):

- `mlflow-data`: `/app/artifacts/mlflow.db` and `/app/artifacts/mlruns`;
  shared by the API and MLflow UI at the **same Linux absolute path**.
- `monitoring-reports`: `/app/reports`, containing Evidently HTML/JSON outputs.

The host's existing [artifacts](artifacts/) are **not mounted, copied,
migrated or modified**. They contain Windows absolute artifact URIs that
are not portable to Linux. The container starts with a separate empty
registry; train once through the dashboard or `POST /train`.
Host notebook training still writes to the original host store and does
not update the container's champion.

The image excludes notebooks, existing models, the host SQLite database
and unrelated files. It contains the API, reusable training code, dashboard,
benchmark CSV and inference sample. Services run as a non-root user.
Only the three HTTP ports are published, all bound to host loopback.

```powershell
docker compose stop
docker compose start
# Removes containers/network, but preserves models and reports:
docker compose down
```

Do **not** add `--volumes` to `docker compose down` unless you deliberately
want to delete all container training history, models and reports. Back up
both named volumes while the services are stopped.

The SQLite deployment uses one API worker and one MLflow worker, suitable
for this local application. The existing in-process training lock prevents
concurrent training within that API. It is not a distributed lock: do not
scale API workers or train concurrently from separate processes against
the same store. Multi-host production deployment would require a shared
database, artifact service, authentication, TLS and distributed job control.
Do not expose these unauthenticated local services to a public network.

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
