from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "data-eng",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bronze_ingestion_daily",
    default_args=DEFAULT_ARGS,
    description="Daily Bronze ingestion for mobility trips",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mobility", "bronze"],
) as dag:

    run_bronze_loader = BashOperator(
        task_id="run_bronze_loader",
        bash_command="python -m src.ingestion.bronze_loader",
    )
    run_bronze_loader
    