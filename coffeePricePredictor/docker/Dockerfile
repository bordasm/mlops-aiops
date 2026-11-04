FROM python:3.11-slim-bookworm AS python-service
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false \
    DO_NOT_TRACK=1
RUN groupadd --gid 10001 coffee \
    && useradd --uid 10001 --gid coffee --create-home coffee
WORKDIR /app

FROM python-service AS ml-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/ml-requirements.txt docker/ml-requirements.txt
RUN python -m pip install -r docker/ml-requirements.txt
ENV PYTHONPATH=/app/api_mlflow:/app/productionReadyCode/src
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/__init__.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/config.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/registry.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/schemas.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee productionReadyCode/src/coffee_price_predictor/__init__.py productionReadyCode/src/coffee_price_predictor/
COPY --chown=coffee:coffee productionReadyCode/src/coffee_price_predictor/metrics.py productionReadyCode/src/coffee_price_predictor/

FROM ml-runtime AS inference
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/serving.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/inference_app.py api_mlflow/coffee_mlflow_api/
USER coffee
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "coffee_mlflow_api.inference_app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM ml-runtime AS training
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/training.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/jobs.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/training_app.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/worker.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee api_mlflow/coffee_mlflow_api/job_runner.py api_mlflow/coffee_mlflow_api/
COPY --chown=coffee:coffee productionReadyCode/src/coffee_price_predictor/data_pipeline.py productionReadyCode/src/coffee_price_predictor/
COPY --chown=coffee:coffee productionReadyCode/src/coffee_price_predictor/modeling.py productionReadyCode/src/coffee_price_predictor/
COPY --chown=coffee:coffee data/coffee_value_benchmark.csv data/
RUN mkdir -p /app/jobs && chown coffee:coffee /app/jobs
ENV COFFEE_JOBS_PATH=/app/jobs/training.db
USER coffee
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "coffee_mlflow_api.training_app:app", "--host", "0.0.0.0", "--port", "8001"]

FROM python-service AS dashboard
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN python -m pip install -r dashboard/requirements.txt
COPY --chown=coffee:coffee dashboard dashboard
COPY --chown=coffee:coffee data/coffee_value_benchmark.csv data/
COPY --chown=coffee:coffee data/coffee_inference_sample.csv data/
RUN mkdir -p /app/reports && chown coffee:coffee /app/reports
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    COFFEE_REPORTS_DIR=/app/reports
USER coffee
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.maxUploadSize=20", "--server.headless=true"]

FROM python-service AS mlflow
COPY docker/mlflow-requirements.txt docker/mlflow-requirements.txt
RUN python -m pip install -r docker/mlflow-requirements.txt \
    && mkdir -p /app/mlflow-artifacts \
    && chown coffee:coffee /app/mlflow-artifacts
USER coffee
EXPOSE 5000
