import mlflow
import pandas as pd

tracking_uri = "sqlite:///C:/D/Work/Ismeret/AI_ML_DL_TL/CUBIX_MLOPS_AIOPS/coffeePricePredictor/experimentLogs/mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

# experiment neve
experiment_name = "coffee-quality-benchmark-pycaret"

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment not found: {experiment_name}")

runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas"
)

print(runs_df.columns.tolist())
print(runs_df[["tags.mlflow.runName", "metrics.r2", "metrics.rmse"]])