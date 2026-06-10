"""
Taxi model training DAG — EC2 self-stop variant.

Runs only when manually triggered (or chained from the ingestion DAG).

Pipeline:
    train_model -> forecast -> stop_instance
"""

from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

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

IMDS_BASE = "http://169.254.169.254/latest"


def _imds_token() -> str:
    resp = requests.put(
        f"{IMDS_BASE}/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.text


def _current_instance_id(token: str) -> str:
    resp = requests.get(
        f"{IMDS_BASE}/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.text


def stop_self(**context):
    """Stop the EC2 instance this Airflow worker is running on."""
    import boto3

    token = _imds_token()
    instance_id = _current_instance_id(token)

    print(f"Pipeline finished — stopping EC2 instance {instance_id}")
    ec2 = boto3.client("ec2")
    ec2.stop_instances(InstanceIds=[instance_id])


with DAG(
    dag_id="taxi_model_training_ec2_self_stop",
    description="Train models, generate forecast, then stop the EC2 host",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["taxi", "training", "ec2"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{BASH_PREFIX} python models/ec2/train.py",
        execution_timeout=timedelta(hours=4),
    )

    forecast = BashOperator(
        task_id="forecast",
        bash_command=f"{BASH_PREFIX} python models/ec2/forecast.py",
        execution_timeout=timedelta(hours=1),
    )

    stop_instance = PythonOperator(
        task_id="stop_instance",
        python_callable=stop_self,
        trigger_rule="all_success",
    )

    train_model >> forecast >> stop_instance
