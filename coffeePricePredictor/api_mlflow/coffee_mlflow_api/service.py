"""Backward-compatible local API combining the separated services."""

from coffee_mlflow_api.serving import (
    InvalidInputError,
    ModelNotLoadedError,
    ServingService,
)
from coffee_mlflow_api.training import TrainingInProgressError, TrainingService

__all__ = [
    "CoffeeQualityMlflowService", "InvalidInputError",
    "ModelNotLoadedError", "TrainingInProgressError",
]


class CoffeeQualityMlflowService(TrainingService, ServingService):
    """Keep the original local API contract for notebooks and users."""

    experiment_names = (
        *TrainingService.experiment_names, *ServingService.experiment_names,
    )
