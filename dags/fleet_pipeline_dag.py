"""
Milestone 7 — Airflow DAG that orchestrates the whole fleet pipeline.

    ingest_raw -> build_features -> build_scorecard -> train_model   (daily)

Each task runs one of our `python -m src...` modules inside the Airflow worker.
The worker image carries our pipeline deps, and the tasks reach the project's
Postgres (maridive_pg) on the host via host.docker.internal (set in the compose).
Training runs with MLFLOW_ENABLED=0 here, so it trains + saves the model without
MLflow (whose deps conflict with Airflow); local runs still track to MLflow.
"""
from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="fleet_pipeline",
    description="Ingest -> features -> scorecard -> train, on a schedule",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["fleet", "mlops"],
) as dag:

    def step(task_id: str, module: str) -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f"cd {PROJECT} && python -m {module}",
        )

    ingest = step("ingest_raw", "src.ingestion.load_raw")
    features = step("build_features", "src.features.build_features")
    scorecard = step("build_scorecard", "src.scorecard.build_scorecard")
    train = step("train_model", "src.ml.train_model")

    ingest >> features >> scorecard >> train
