"""FastAPI application serving the coffee quality model via MLflow."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse

from coffee_mlflow_api import __version__, config
from coffee_mlflow_api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from coffee_mlflow_api.service import (
    CoffeeQualityMlflowService,
    InvalidInputError,
    ModelNotLoadedError,
    TrainingInProgressError,
)

service = CoffeeQualityMlflowService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure MLflow when the API starts."""
    service.setup()
    yield


app = FastAPI(
    title="Coffee Quality Predictor API (MLflow)",
    description=(
        "REST API for training and serving the coffee quality benchmark "
        "regression model with **MLflow** tracking and Model Registry. "
        "The model predicts `coffee_quality_score` from the farm, "
        "environment, processing, storage, physical, commercial and "
        "sensory attributes of a coffee lot.\n\n"
        "* **POST /train** logs every candidate model in MLflow, registers "
        "the best one as a new version of `coffee-quality-model` and "
        "moves the `champion` alias to it.\n"
        "* **POST /predict** scores an uploaded CSV with the `champion` "
        "model and logs the predictions as an MLflow artifact.\n\n"
        "MLflow store: `coffeePricePredictor/artifacts` "
        "(`mlflow.db` + `mlruns`). A sample input file is available at "
        "`data/coffee_inference_sample.csv`."
    ),
    version=__version__,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Service", "description": "Health and model metadata."},
        {"name": "Training", "description": "Model training."},
        {"name": "Inference", "description": "Model predictions."},
    ],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root URL to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health() -> HealthResponse:
    """Return the health status of the service."""
    return HealthResponse(
        status="ok",
        tracking_uri=service.tracking_uri,
        champion_model_available=service.is_champion_available(),
        training_in_progress=service.is_training,
    )


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["Service"],
    responses={503: {"model": ErrorResponse}},
)
def model_info() -> ModelInfoResponse:
    """Return registry information and metrics of the champion model."""
    try:
        return ModelInfoResponse(**service.get_model_info())
    except ModelNotLoadedError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    responses={
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def train(request: TrainRequest | None = None) -> TrainResponse:
    """Train, log and register the coffee quality model with MLflow.

    RandomForest, XGBoost and LightGBM are trained in nested MLflow runs.
    The best model (by test R-squared) is registered as a new version of
    `coffee-quality-model` and gets the `champion` alias. The request
    body is optional. Training takes roughly 1-2 minutes.
    """
    request = request or TrainRequest()
    try:
        result = service.train(
            test_size=request.test_size,
            random_state=request.random_state,
        )
    except TrainingInProgressError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {error}",
        ) from error

    return TrainResponse(**result)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(
    file: UploadFile = File(
        ...,
        description=(
            "CSV file with the feature columns of the training data. "
            "Extra columns are ignored; `record_id` is echoed back and "
            "`coffee_quality_score`, if present, is used for evaluation."
        ),
    ),
) -> PredictResponse:
    """Predict the coffee quality score and log the run in MLflow."""
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported.",
        )

    content = file.file.read(config.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded file exceeds the 20 MB limit.",
        )

    try:
        result = service.predict_from_csv(content, file.filename)
    except ModelNotLoadedError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except InvalidInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return PredictResponse(**result)
