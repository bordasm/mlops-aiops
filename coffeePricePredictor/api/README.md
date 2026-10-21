# Coffee quality predictor REST API

FastAPI service that trains and serves the coffee quality benchmark model
(target: `coffee_quality_score`). Training reuses the production-ready code
in `../productionReadyCode/src`.

## Install

```powershell
c:\Users\bordasm\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From this directory:

```powershell
c:\Users\bordasm\.venv\Scripts\python.exe main.py
```

The API listens on <http://127.0.0.1:55>.

- Swagger UI: <http://127.0.0.1:55/docs>
- ReDoc: <http://127.0.0.1:55/redoc>
- OpenAPI schema: <http://127.0.0.1:55/openapi.json>

## Endpoints

| Method | Path       | Description                                                   |
|--------|------------|---------------------------------------------------------------|
| GET    | `/health`  | Service status and whether a model is loaded.                 |
| GET    | `/model`   | Name, feature columns and metrics of the loaded model.        |
| POST   | `/train`   | Trains RandomForest, XGBoost and LightGBM, keeps the best one. |
| POST   | `/predict` | Predicts `coffee_quality_score` for an uploaded CSV file.     |

### POST /train

Optional JSON body (defaults shown):

```json
{"test_size": 0.2, "random_state": 42}
```

Training takes about 30 seconds. The artifacts are saved to `artifacts/`
and loaded immediately; they are also loaded automatically on startup.

### POST /predict

`multipart/form-data` upload with a `file` field containing a CSV file.
The CSV must contain the 31 feature columns of the training data (see
`GET /model`). Extra columns (for example targets) are ignored, and
`record_id` is echoed back when present. Sample file:
`../data/coffee_inference_sample.csv`.

```powershell
curl.exe -F "file=@..\data\coffee_inference_sample.csv" http://127.0.0.1:55/predict
```

### Error codes

- `400` – invalid CSV, missing feature columns or non-numeric values
- `409` – a training run is already in progress
- `413` – uploaded file is larger than 20 MB
- `422` – invalid training parameters
- `503` – no trained model yet (call `POST /train` first)
