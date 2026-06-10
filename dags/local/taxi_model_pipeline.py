"""
Taxi model training DAG.

Runs only when manually triggered.

Dependencies: taxi_data_pipeline (provides merged and engineered features)

Pipeline:
    train_model
        ->
    forecast
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

AIRFLOW_ROOT = "/opt/airflow"

BASH_PREFIX = (
    f"cd {AIRFLOW_ROOT} && "
    "export PYTHONPATH=/opt/airflow:${PYTHONPATH:-} && "
)

with DAG(
    dag_id="taxi_model_training",
    description="Merge, feature engineering, training and forecasting",
    schedule=None,  # manual only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["taxi", "training"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{BASH_PREFIX} python models/local/train.py",
    )

    forecast = BashOperator(
        task_id="forecast",
        bash_command=f"{BASH_PREFIX} python models/local/forecast.py",
    )

    train_model >> forecast