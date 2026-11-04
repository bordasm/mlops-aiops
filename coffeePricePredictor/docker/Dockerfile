FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/api_mlflow:/app/productionReadyCode/src \
    COFFEE_ARTIFACTS_DIR=/app/artifacts \
    MLFLOW_TRACKING_URI=sqlite:////app/artifacts/mlflow.db \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 coffee \
    && useradd --uid 10001 --gid coffee --create-home coffee

WORKDIR /app
COPY api_mlflow/requirements.txt api_mlflow/requirements.txt
COPY dashboard/requirements.txt dashboard/requirements.txt
COPY docker/requirements.txt docker/requirements.txt
RUN python -m pip install -r docker/requirements.txt

COPY --chown=coffee:coffee api_mlflow api_mlflow
COPY --chown=coffee:coffee productionReadyCode/src productionReadyCode/src
COPY --chown=coffee:coffee dashboard dashboard
COPY --chown=coffee:coffee data/coffee_value_benchmark.csv data/
COPY --chown=coffee:coffee data/coffee_inference_sample.csv data/
RUN mkdir -p /app/artifacts /app/reports \
    && chown coffee:coffee /app/artifacts /app/reports

USER coffee
EXPOSE 8000 8501 5000
CMD ["python", "-m", "uvicorn", "coffee_mlflow_api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
