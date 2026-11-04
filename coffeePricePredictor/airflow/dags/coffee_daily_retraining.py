"""One daily model refresh at 02:00 in the Europe/Budapest timezone."""

from datetime import timedelta
from hashlib import sha256

import pendulum
from airflow.sdk import DAG, get_current_context, task
from airflow.timetables.trigger import CronTriggerTimetable
from coffee_training import retrain_coffee_model

with DAG(
    dag_id="coffee_daily_retraining",
    description=(
        "Retrain candidates via the API and verify the MLflow champion."
    ),
    schedule=CronTriggerTimetable(
        "0 2 * * *",
        timezone="Europe/Budapest",
    ),
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Budapest"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    is_paused_upon_creation=False,
    dagrun_timeout=timedelta(minutes=40),
    default_args={"retries": 0},
    tags=["coffee", "mlflow", "daily-training"],
    doc_md=(
        "Daily at 02:00 Europe/Budapest. No historical catchup or automatic "
        "task retries. Submits an idempotent asynchronous training job, "
        "polls it, uses the configured benchmark, logs candidates "
        "to MLflow and updates the champion. Check API/MLflow before manually "
        "retrying a failed or timed-out task."
    ),
) as dag:

    @task(
        task_id="retrain_and_verify_champion",
        execution_timeout=timedelta(minutes=35),
        multiple_outputs=False,
    )
    def retrain_and_verify_champion() -> dict:
        """Return the verified training run reference as a small XCom."""
        context = get_current_context()
        run_id = context["run_id"]
        request_id = "airflow:" + sha256(run_id.encode()).hexdigest()
        return retrain_coffee_model(request_id)

    retrain_and_verify_champion()
