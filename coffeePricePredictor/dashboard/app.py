"""Streamlit UI for API training, inference and Evidently monitoring."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from client import ApiError, CoffeeApiClient
from monitoring import (
    TARGET,
    MonitoringResult,
    attach_predictions,
    create_regression_report,
    create_report,
    prepare_current,
    read_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get(
    "COFFEE_DATA_PATH",
    PROJECT_ROOT / "data" / "coffee_value_benchmark.csv",
))
REPORTS_DIR = Path(os.environ.get(
    "COFFEE_REPORTS_DIR", PROJECT_ROOT / "artifacts" / "reports",
))
API_URL = os.environ.get("COFFEE_API_URL", "http://127.0.0.1:56")
MLFLOW_UI_URL = os.environ.get(
    "COFFEE_MLFLOW_UI_URL", "http://localhost:5000",
)


@st.cache_data
def load_reference(path: str, modified_ns: int) -> pd.DataFrame:
    """Cache the benchmark, invalidating when its file changes."""
    return pd.read_csv(path)


def display_report(result: MonitoringResult, title: str) -> None:
    """Render a persisted report with portable HTML and JSON downloads."""
    with st.expander(title, expanded=True):
        if result.excluded_drift_columns:
            st.warning(
                "Drift cannot be computed for entirely missing columns: "
                + ", ".join(result.excluded_drift_columns)
                + ". They remain in the quality report. The drift share "
                "applies only to features with observed values."
            )
        html = result.html_path.read_text(encoding="utf-8")
        st.iframe(html, height=850)
        left, right = st.columns(2)
        left.download_button(
            "Download HTML",
            html,
            file_name=result.html_path.name,
            mime="text/html",
            key=f"{title}:{result.html_path.name}",
        )
        right.download_button(
            "Download JSON",
            result.json_path.read_bytes(),
            file_name=result.json_path.name,
            mime="application/json",
            key=f"{title}:{result.json_path.name}",
        )
        if result.evaluated_rows:
            st.caption(
                f"Evaluated labels: {result.evaluated_rows}; "
                f"excluded missing labels: {result.excluded_target_rows}."
            )


def show_training(client: CoffeeApiClient, is_training: bool) -> None:
    """Launch training explicitly and compare the returned candidates."""
    st.subheader("Train and compare candidate models")
    st.caption(
        "Training uses the full configured benchmark and updates the "
        "champion alias. It can take several minutes."
    )
    with st.form("training"):
        test_size = st.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)
        random_state = st.number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, value=42,
        )
        submitted = st.form_submit_button(
            "Train models", disabled=is_training,
        )
    if submitted:
        try:
            with st.spinner("Training models via the REST API..."):
                result = client.request(
                    "POST", "/train", timeout=1800,
                    json={
                        "test_size": test_size,
                        "random_state": random_state,
                    },
                )
            st.session_state["training_result"] = result
            for key in (
                "prediction_result", "data_report", "regression_report",
            ):
                st.session_state.pop(key, None)
            st.success(result["message"])
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    result = st.session_state.get("training_result")
    if result:
        st.caption(
            f"Last dashboard training run: {result['training_run_id']}. "
            f"Selected: {result['best_model']}."
        )
        comparison = pd.DataFrame([
            {
                "model": name,
                **{f"test_{key}": value
                   for key, value in metrics["test"].items()},
                "fit_time_seconds": metrics["fit_time_seconds"],
            }
            for name, metrics in result["all_models"].items()
        ]).sort_values("test_r2", ascending=False)
        st.dataframe(comparison, hide_index=True)


def show_inference(
    client: CoffeeApiClient,
    reference: pd.DataFrame,
    model: dict | None,
) -> None:
    """Predict uploads and generate monitoring independently of training."""
    st.subheader("CSV inference and monitoring")
    st.caption(
        "The model predicts coffee_quality_score, not price. "
        "Feature drift excludes IDs, the target and leakage columns. "
        "The full training benchmark is the reference, not a held-out "
        "performance baseline. Small uploads give unreliable drift estimates."
    )
    sample_path = PROJECT_ROOT / "data" / "coffee_inference_sample.csv"
    st.download_button(
        "Download sample CSV", sample_path.read_bytes(),
        file_name=sample_path.name, mime="text/csv",
    )
    uploaded = st.file_uploader(
        "Current data (UTF-8 CSV, max. 20 MB)", type="csv",
    )
    if uploaded is None:
        return
    content = uploaded.getvalue()
    model_identity = (
        (model["model_version"], model["model_run_id"]) if model else None
    )
    input_key = (
        hashlib.sha256(content).hexdigest(),
        model_identity,
        DATA_PATH.stat().st_mtime_ns,
    )
    if st.session_state.get("input_key") != input_key:
        for key in ("prediction_result", "data_report", "regression_report"):
            st.session_state.pop(key, None)
        st.session_state["input_key"] = input_key
    try:
        current = prepare_current(reference, read_csv(content))
    except ValueError as error:
        st.error(str(error))
        return
    st.write(f"Current rows: {len(current)}; reference rows: {len(reference)}")
    st.dataframe(current.head(20), hide_index=True)
    if len(current) < 100:
        st.warning(
            "Fewer than 100 current rows: interpret drift tests with caution."
        )
    if st.button("Predict CSV", disabled=model is None):
        try:
            with st.spinner("Scoring through the API..."):
                response = client.predict(content, uploaded.name)
                scored = attach_predictions(current, response)
            st.session_state["prediction_result"] = (response, scored)
            st.session_state.pop("regression_report", None)
        except (ApiError, ValueError) as error:
            st.error(str(error))
    prediction_result = st.session_state.get("prediction_result")
    if prediction_result:
        response, scored = prediction_result
        st.caption(
            f"Model version: {response['model_version']}; "
            f"inference run: {response['inference_run_id']}"
        )
        if response["evaluation"]:
            st.json(response["evaluation"])
        st.dataframe(scored.head(100), hide_index=True)
        st.download_button(
            "Download predictions CSV",
            scored.to_csv(index=False).encode("utf-8"),
            file_name="coffee_quality_predictions.csv",
            mime="text/csv",
        )
    drift_share = st.slider(
        "Drifted feature share required for dataset drift",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Uses Evidently's default per-feature statistical tests.",
    )
    if st.session_state.get("drift_share") != drift_share:
        st.session_state.pop("data_report", None)
        st.session_state["drift_share"] = drift_share
    if st.button("Generate Evidently data drift and quality report"):
        try:
            with st.spinner("Computing Evidently data report..."):
                st.session_state["data_report"] = create_report(
                    reference, current, REPORTS_DIR, drift_share=drift_share,
                )
        except (ValueError, OSError) as error:
            st.error(f"Cannot generate data report: {error}")
    if "data_report" in st.session_state:
        display_report(
            st.session_state["data_report"], "Data drift and quality",
        )
    if prediction_result and TARGET in current:
        _, scored = prediction_result
        if st.button("Generate Evidently regression report"):
            try:
                with st.spinner("Computing regression quality..."):
                    result = create_regression_report(scored, REPORTS_DIR)
                    st.session_state["regression_report"] = result
            except (ValueError, OSError) as error:
                st.error(f"Cannot generate regression report: {error}")
    else:
        st.info(
            f"Regression monitoring needs predictions and a '{TARGET}' "
            "column containing at least two known target values."
        )
    if "regression_report" in st.session_state:
        display_report(
            st.session_state["regression_report"],
            "Regression performance",
        )


def show_saved_reports() -> None:
    """Allow durable report downloads after browser or container restarts."""
    st.subheader("Saved Evidently reports")
    paths = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
    if not paths:
        st.info("No saved reports yet.")
        return
    selected = st.selectbox(
        "Saved HTML report", paths, format_func=lambda path: path.name,
    )
    if selected:
        json_path = selected.with_suffix(".json")
        if not json_path.exists():
            st.error(f"Missing JSON companion for {selected.name}.")
            return
        try:
            metadata = json.loads(
                json_path.read_text(encoding="utf-8"),
            )["coffee_monitoring"]
            result = MonitoringResult(
                selected, json_path,
                metadata["evaluated_rows"],
                metadata["excluded_target_rows"],
                tuple(metadata["excluded_drift_columns"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"Cannot read saved report metadata: {error}")
            return
        display_report(result, "Saved report")


def main() -> None:
    """Render API state, training, inference and saved monitoring reports."""
    st.set_page_config(page_title="Coffee Quality Monitor", layout="wide")
    st.title("Coffee Quality Monitor")
    st.sidebar.caption(f"API: {API_URL}")
    st.sidebar.link_button("MLflow experiments and registry", MLFLOW_UI_URL)
    st.sidebar.button("Refresh status")
    client = CoffeeApiClient(API_URL)
    try:
        health = client.request("GET", "/health")
    except ApiError as error:
        st.error(str(error))
        show_saved_reports()
        return
    left, middle, right = st.columns(3)
    left.metric("API", health["status"])
    middle.metric(
        "Champion",
        "Available" if health["champion_model_available"] else "None",
    )
    right.metric(
        "Training", "Running" if health["training_in_progress"] else "Idle",
    )
    model = None
    if health["champion_model_available"]:
        try:
            model = client.request("GET", "/model")
            st.subheader(
                f"Champion: {model['model_name']} "
                f"(version {model['model_version']})"
            )
            st.dataframe(
                pd.DataFrame(
                    model["metrics"].items(), columns=["metric", "value"],
                ),
                hide_index=True,
            )
            with st.expander("Model metadata and required features"):
                st.json(model)
        except ApiError as error:
            st.error(str(error))
    else:
        st.info("New model store: train models below before predicting.")
    training, inference, saved = st.tabs(
        ["Training", "Inference and monitoring", "Saved reports"],
    )
    with training:
        show_training(client, health["training_in_progress"])
    with inference:
        try:
            reference = load_reference(
                str(DATA_PATH), DATA_PATH.stat().st_mtime_ns,
            )
            show_inference(client, reference, model)
        except (OSError, pd.errors.ParserError, ValueError) as error:
            st.error(f"Cannot load monitoring data: {error}")
    with saved:
        show_saved_reports()


if __name__ == "__main__":
    main()
