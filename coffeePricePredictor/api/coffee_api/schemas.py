"""Request and response schemas of the coffee quality REST API."""

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
    time_seconds: float = Field(description="Model fitting time.")


class TrainResponse(BaseModel):
    """Result of a training run."""

    message: str
    best_model: str
    best_model_metrics: ModelEvaluation
    all_models: dict[str, ModelEvaluation]
    training_time_seconds: float


class ModelInfoResponse(BaseModel):
    """Metadata of the currently loaded model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    target_column: str
    feature_columns: list[str]
    best_model_metrics: ModelEvaluation
    all_models: dict[str, ModelEvaluation]


class PredictionItem(BaseModel):
    """Prediction for one row of the uploaded CSV file."""

    row_index: int = Field(description="Zero-based row index in the CSV.")
    record_id: str | None = Field(
        default=None,
        description="Value of the 'record_id' column, if present.",
    )
    predicted_coffee_quality_score: float


class PredictResponse(BaseModel):
    """Predictions for an uploaded CSV file."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    target_column: str
    record_count: int
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Service health information."""

    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Error payload returned by the API."""

    detail: str
