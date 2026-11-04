"""Independent inference API: no training code, job store or dataset."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.exceptions import MlflowException

from coffee_mlflow_api import config
from coffee_mlflow_api.schemas import ModelInfoResponse, PredictResponse
from coffee_mlflow_api.serving import (
    InvalidInputError,
    ModelNotLoadedError,
    ServingService,
)

service = ServingService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize only the inference experiment in remote MLflow."""
    service.setup()
    yield


app = FastAPI(title="Coffee Inference Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Report dependency readiness without coupling to the training service."""
    try:
        available = service.is_champion_available()
    except MlflowException as error:
        raise HTTPException(503, f"MLflow unavailable: {error}") from error
    return {
        "status": "ok",
        "tracking_uri": service.tracking_uri,
        "champion_model_available": available,
    }


@app.get("/model", response_model=ModelInfoResponse)
def model() -> dict:
    """Return champion metadata, reloading when its version changes."""
    try:
        return service.get_model_info()
    except (ModelNotLoadedError, MlflowException) as error:
        raise HTTPException(503, str(error)) from error


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> dict:
    """Score a CSV and persist its inference run through MLflow HTTP."""
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported.")
    content = file.file.read(config.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, "The uploaded file exceeds the 20 MB limit.")
    try:
        return service.predict_from_csv(content, file.filename)
    except InvalidInputError as error:
        raise HTTPException(400, str(error)) from error
    except (ModelNotLoadedError, MlflowException) as error:
        raise HTTPException(503, str(error)) from error
