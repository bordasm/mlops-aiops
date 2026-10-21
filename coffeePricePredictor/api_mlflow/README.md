# Coffee quality predictor REST API with MLflow

FastAPI service that trains, registers and serves the coffee quality
benchmark model (target: `coffee_quality_score`) with MLflow tracking and
the MLflow Model Registry. Data preparation and modeling reuse the
production-ready code in `../productionReadyCode/src`.

The API shares its MLflow store with the notebooks
`../pipelines/coffee-quality-pricing-benchmark_training_pipeline_mlflow.ipynb`
and `../pipelines/coffee-quality-pricing-benchmark_inference_pipeline_mlflow.ipynb`.

## MLflow store

Everything is stored in `../artifacts`:

- `mlflow.db` – tracking and model registry (SQLite)
- `mlruns/coffee-quality-training/` – training runs and logged models
- `mlruns/coffee-quality-inference/` – inference runs with the predictions CSV

| Item                | Name                                       |
|---------------------|--------------------------------------------|
| Training experiment | `coffee-quality-training`                  |
| Inference experiment| `coffee-quality-inference`                 |
| Registered model    | `coffee-quality-model`                     |
| Serving alias       | `champion` (always the latest trained best model) |

The logged model is a scikit-learn pipeline (preprocessor + regressor),
saved in `cloudpickle` format because the default `skops` format does not
support LightGBM and XGBoost. Load it only from this trusted local store.

## Install

```powershell
c:\Users\bordasm\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From this directory:

```powershell
c:\Users\bordasm\.venv\Scripts\python.exe main.py
```

The API listens on <http://127.0.0.1:56> (the API without MLflow in
`../api` uses port 55).

- Swagger UI: <http://127.0.0.1:56/docs>
- ReDoc: <http://127.0.0.1:56/redoc>

## MLflow UI

From the `coffeePricePredictor` directory:

```powershell
c:\Users\bordasm\.venv\Scripts\mlflow.exe ui `
    --backend-store-uri sqlite:///artifacts/mlflow.db `
    --default-artifact-root ./artifacts/mlruns `
    --host 127.0.0.1 --port 5000
```

Then open <http://127.0.0.1:5000>.

## Endpoints

| Method | Path       | Description                                                        |
|--------|------------|--------------------------------------------------------------------|
| GET    | `/health`  | Service status, tracking URI, champion availability.               |
| GET    | `/model`   | Registry version, MLflow run ids, metrics and features of the champion. |
| POST   | `/train`   | Trains, logs and registers the models; moves `champion` to the best one. |
| POST   | `/predict` | Scores an uploaded CSV with the champion and logs an inference run. |

### POST /train

Optional JSON body (defaults shown):

```json
{"test_size": 0.2, "random_state": 42}
```

One parent MLflow run is created with a nested run per candidate model
(RandomForest, XGBoost, LightGBM). Each nested run contains the model
parameters, train/test metrics and the logged model. The parent run
contains the dataset reference, `model_comparison.json` and
`model_summary.json`. The best model (by test R²) is registered as a new
version of `coffee-quality-model` and gets the `champion` alias.
Training takes about 1–2 minutes.

### POST /predict

`multipart/form-data` upload with a `file` field containing a CSV file.
The CSV must contain the 31 feature columns (see `GET /model`). Extra
columns are ignored, `record_id` is echoed back, and if
`coffee_quality_score` is present, RMSE, MAE and R² are returned and
logged. Each call creates an MLflow run in `coffee-quality-inference`
with the predictions CSV as an artifact. Sample file:
`../data/coffee_inference_sample.csv`.

The champion alias is checked on every request, so a model registered
by the training notebook is used without restarting the API.

```powershell
curl.exe -F "file=@..\data\coffee_inference_sample.csv" http://127.0.0.1:56/predict
```

### Error codes

- `400` – invalid CSV, missing feature columns or non-numeric values
- `409` – a training run is already in progress
- `413` – uploaded file is larger than 20 MB
- `422` – invalid training parameters
- `503` – no champion model registered yet (call `POST /train` first)
