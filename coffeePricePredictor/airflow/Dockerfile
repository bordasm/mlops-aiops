FROM apache/airflow:3.3.2-python3.11

USER root
RUN mkdir -p /opt/airflow/state \
    && chown airflow:root /opt/airflow/state \
    && chmod 0770 /opt/airflow/state

COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root tests/ /opt/airflow/tests/
USER airflow
