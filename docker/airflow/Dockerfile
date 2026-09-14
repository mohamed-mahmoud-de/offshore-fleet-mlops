# Custom Airflow image = official Airflow + our pipeline dependencies,
# so DAG tasks can run `python -m src...` directly.
FROM apache/airflow:2.10.5

# libgomp1 is required by LightGBM at runtime (installed as root, then drop back).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
# Install WITH Airflow's constraints so our deps can't upgrade Airflow's pinned core.
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
