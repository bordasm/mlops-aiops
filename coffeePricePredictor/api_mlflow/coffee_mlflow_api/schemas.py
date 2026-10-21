"""Request and response schemas of the MLflow based REST API."""

from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Optional hyperparameters of the training workflow."""

    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of the data reserved for model evaluation.",
        examples=[0.2],
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed used by the train/test split.",
        examples=[42],
    )


class RegressionMetrics(BaseModel):
    """Regression metrics computed on one data split."""

    rmse: float
    mae: float
    r2: float


class ModelEvaluation(BaseModel):
    """Train and test metrics of a candidate model."""

    train: RegressionMetrics
    test: RegressionMetrics
    fit_time_seconds: float = Field(description="Model fitting time.")


class RegisteredModelInfo(BaseModel):
    """Reference to a model version in the MLflow Model Registry."""

    model_config = ConfigDict(protected_namespaces=())

    registered_model_name: str
    model_version: str
    model_alias: str
    model_uri: str


class TrainResponse(RegisteredModelInfo):
    """Result of a training run tracked in MLflow."""

    message: str
    best_model: str
    best_model_metrics: ModelEvaluation
    all_models: dict[str, ModelEvaluation]
    training_run_id: str = Field(description="Parent MLflow run id.")
    candidate_run_ids: dict[str, str] = Field(
        description="Nested MLflow run id of each candidate model."
    )
    training_time_seconds: float


class ModelInfoResponse(RegisteredModelInfo):
    """Metadata of the current champion model."""

    model_name: str
    model_run_id: str = Field(
        description="MLflow run id in which the model was logged."
    )
    training_run_id: str | None = Field(
        default=None,
        description="Parent MLflow run id of the training pipeline.",
    )
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float]


class PredictionItem(BaseModel):
    """Prediction for one row of the uploaded CSV file."""

    row_index: int = Field(description="Zero-based row index in the CSV.")
    record_id: str | None = Field(
        default=None,
        description="Value of the 'record_id' column, if present.",
    )
    predicted_coffee_quality_score: float


class PredictResponse(RegisteredModelInfo):
    """Predictions for an uploaded CSV file."""

    model_name: str
    inference_run_id: str = Field(
        description="MLflow run id in which the predictions were logged."
    )
    predictions_artifact_uri: str
    target_column: str
    record_count: int
    evaluation: dict[str, float] | None = Field(
        default=None,
        description=(
            "RMSE, MAE and R-squared; only if the CSV contains the "
            "target column."
        ),
    )
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Service health information."""

    status: str
    tracking_uri: str
    champion_model_available: bool
    training_in_progress: bool


class ErrorResponse(BaseModel):
    """Error payload returned by the API."""

    detail: str
