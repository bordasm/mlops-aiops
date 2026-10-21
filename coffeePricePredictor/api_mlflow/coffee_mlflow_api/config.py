"""Configuration values for the MLflow based coffee quality REST API."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "coffee_value_benchmark.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRACKING_URI = "sqlite:///" + (ARTIFACTS_DIR / "mlflow.db").as_posix()
MLRUNS_DIR = ARTIFACTS_DIR / "mlruns"

TRAINING_EXPERIMENT_NAME = "coffee-quality-training"
INFERENCE_EXPERIMENT_NAME = "coffee-quality-inference"
REGISTERED_MODEL_NAME = "coffee-quality-model"
MODEL_ALIAS = "champion"
MODEL_ARTIFACT_NAME = "model"
PREDICTIONS_FILE_NAME = "coffee_quality_predictions.csv"

TARGET_COLUMN = "coffee_quality_score"
ID_COLUMN = "record_id"
PREDICTION_COLUMN = "predicted_coffee_quality_score"
SELECTION_METRIC = "test_r2"

HOST = "127.0.0.1"
PORT = 56

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
