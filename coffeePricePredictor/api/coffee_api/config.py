"""Configuration values for the coffee quality REST API."""

from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = API_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "coffee_value_benchmark.csv"
ARTIFACTS_DIR = API_DIR / "artifacts"

MODEL_FILE_NAME = "coffee_quality_model.joblib"
PREPROCESSOR_FILE_NAME = "preprocessor.joblib"
SUMMARY_FILE_NAME = "model_summary.json"

TARGET_COLUMN = "coffee_quality_score"
ID_COLUMN = "record_id"

HOST = "127.0.0.1"
PORT = 55

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
