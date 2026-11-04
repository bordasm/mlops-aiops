"""Airflow-native DAG checks; execute inside the Linux Airflow image."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pendulum
from airflow.dag_processing.dagbag import DagBag
from airflow.timetables.base import DataInterval, TimeRestriction

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "dags"
sys.path.insert(0, str(DAGS_FOLDER))


class DailyDagTests(unittest.TestCase):
    """Validate the parsed DAG and its real Airflow timetable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bag = DagBag(dag_folder=str(DAGS_FOLDER))
        cls.dag = cls.bag.dags.get("coffee_daily_retraining")

    def test_imports_and_task_contract(self) -> None:
        self.assertEqual(self.bag.import_errors, {})
        self.assertIsNotNone(self.dag)
        self.assertEqual(
            self.dag.task_ids, ["retrain_and_verify_champion"],
        )
        task = self.dag.get_task("retrain_and_verify_champion")
        self.assertEqual(task.retries, 0)
        self.assertEqual(task.execution_timeout.total_seconds(), 35 * 60)
        self.assertFalse(self.dag.catchup)
        self.assertFalse(self.dag.is_paused_upon_creation)
        self.assertEqual(self.dag.max_active_runs, 1)

    def test_winter_and_summer_local_schedule(self) -> None:
        for day in ((2026, 1, 12), (2026, 7, 12)):
            now = pendulum.datetime(*day, 1, tz="Europe/Budapest")
            with self.subTest(day=day), patch(
                "airflow.timetables.trigger.utcnow",
                return_value=now.in_timezone("UTC"),
            ):
                next_run = self.dag.timetable.next_dagrun_info(
                    last_automated_data_interval=DataInterval.exact(
                        now.subtract(days=1).replace(hour=2),
                    ),
                    restriction=TimeRestriction(
                        earliest=self.dag.start_date,
                        latest=None,
                        catchup=False,
                    ),
                )
                local = next_run.run_after.in_timezone("Europe/Budapest")
                self.assertEqual(local.hour, 2)
                self.assertEqual(local.minute, 0)
                self.assertEqual(local.date(), now.date())
                self.assertGreater(next_run.run_after, now)

    def test_restart_runs_only_the_latest_slot_not_each_missed_day(self):
        now = pendulum.datetime(2026, 11, 4, 15, tz="Europe/Budapest")
        with patch(
            "airflow.timetables.trigger.utcnow",
            return_value=now.in_timezone("UTC"),
        ):
            next_run = self.dag.timetable.next_dagrun_info(
                last_automated_data_interval=DataInterval.exact(
                    pendulum.datetime(2026, 11, 1, 2, tz="Europe/Budapest"),
                ),
                restriction=TimeRestriction(
                    earliest=self.dag.start_date, latest=None, catchup=False,
                ),
            )
        self.assertEqual(
            next_run.run_after.in_timezone("Europe/Budapest"),
            pendulum.datetime(2026, 11, 4, 2, tz="Europe/Budapest"),
        )

    def test_clock_changes_keep_one_daily_slot(self) -> None:
        for month, day, expected_hour in ((3, 29, 3), (10, 25, 2)):
            now = pendulum.datetime(
                2026, month, day, 0, tz="Europe/Budapest",
            )
            previous = DataInterval.exact(
                now.subtract(days=1).replace(hour=2),
            )
            with self.subTest(month=month), patch(
                "airflow.timetables.trigger.utcnow",
                return_value=now.in_timezone("UTC"),
            ):
                next_run = self.dag.timetable.next_dagrun_info(
                    last_automated_data_interval=previous,
                    restriction=TimeRestriction(
                        earliest=self.dag.start_date,
                        latest=None,
                        catchup=False,
                    ),
                )
                local = next_run.run_after.in_timezone("Europe/Budapest")
                self.assertEqual(local.date(), now.date())
                self.assertEqual(local.hour, expected_hour)
                following = self.dag.timetable.next_dagrun_info(
                    last_automated_data_interval=next_run.data_interval,
                    restriction=TimeRestriction(
                        earliest=self.dag.start_date,
                        latest=None,
                        catchup=False,
                    ),
                )
                following_local = following.run_after.in_timezone(
                    "Europe/Budapest",
                )
                self.assertEqual(
                    following_local.date(), now.add(days=1).date(),
                )
                self.assertEqual(following_local.hour, 2)


if __name__ == "__main__":
    unittest.main()
