"""FastAPI application exposing the coffee quality model."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse

from coffee_api import __version__, config
from coffee_api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from coffee_api.service import (
    CoffeeQualityService,
    InvalidInputError,
    ModelNotLoadedError,
    TrainingInProgressError,
)

service = CoffeeQualityService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load previously trained artifacts when the API starts."""
    service.load_artifacts()
    yield


app = FastAPI(
    title="Coffee Quality Predictor API",
    description=(
        "REST API for training and serving the coffee quality benchmark "
        "regression model. The model predicts `coffee_quality_score` from "
        "the farm, environment, processing, storage, physical, commercial "
        "and sensory attributes of a coffee lot.\n\n"
        "Typical workflow: call **POST /train** to train the model, then "
        "upload a CSV file to **POST /predict**. A sample input file is "
        "available at `data/coffee_inference_sample.csv`."
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
    return HealthResponse(status="ok", model_loaded=service.is_model_loaded)


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["Service"],
    responses={503: {"model": ErrorResponse}},
)
def model_info() -> ModelInfoResponse:
    """Return metadata and metrics of the currently loaded model."""
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
    """Train RandomForest, XGBoost and LightGBM and keep the best model.

    The request body is optional. The best model is selected by test
    R-squared, saved to the `api/artifacts` folder and loaded for
    inference immediately. Training takes roughly 30 seconds.
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

    return TrainResponse(message="Training completed successfully.", **result)


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
            "Target and identifier columns are optional and ignored; "
            "`record_id` is echoed back when present."
        ),
    ),
) -> PredictResponse:
    """Predict the coffee quality score for each row of a CSV file."""
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
        return PredictResponse(**service.predict_from_csv(content))
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
