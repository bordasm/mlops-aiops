"""Shared MLflow registry access; no training or serving logic."""

from pathlib import Path
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from coffee_mlflow_api import config


class ModelNotLoadedError(RuntimeError):
    """No champion has been registered in the configured store."""


class RegistryService:
    """Communicate through MLflow; retain local legacy store support."""

    experiment_names: tuple[str, ...] = ()

    def __init__(self, tracking_uri: str = config.TRACKING_URI,
                 data_path: Path = config.DATA_PATH) -> None:
        self._tracking_uri = tracking_uri
        self._data_path = Path(data_path)
        self._experiment_ids: dict[str, str] = {}
        self._client: MlflowClient | None = None

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri

    def setup(self) -> None:
        if not self._tracking_uri.startswith(("http://", "https://")):
            config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        for name in self.experiment_names:
            self._experiment_ids[name] = self._get_or_create_experiment(name)

    def _get_or_create_experiment(self, name: str) -> str:
        """Return the id of an experiment, creating it if needed."""
        client = self._require_client()
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            if self._tracking_uri.startswith(("http://", "https://")):
                return client.create_experiment(name)
            return client.create_experiment(
                name, artifact_location=(config.MLRUNS_DIR / name).as_uri(),
            )
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                raise
            return experiment.experiment_id

    def _require_client(self) -> MlflowClient:
        """Return the MLflow client or raise if setup was not called."""
        if self._client is None:
            raise RuntimeError("Call setup() before using the service.")
        return self._client

    @staticmethod
    def _registered_model_info(version: ModelVersion) -> dict[str, str]:
        """Return the registry reference of a model version."""
        return {
            "registered_model_name": config.REGISTERED_MODEL_NAME,
            "model_version": str(version.version),
            "model_alias": config.MODEL_ALIAS,
            "model_uri": (
                f"models:/{config.REGISTERED_MODEL_NAME}/{version.version}"
            ),
        }
