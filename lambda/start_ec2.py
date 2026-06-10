# =============================================================================
# Lambda: start the Airflow EC2 instance ahead of the monthly DAG run
# =============================================================================
# Trigger: EventBridge scheduled rule, e.g.
#   cron(0 6 1 * ? *)   -> 06:00 UTC on the 1st of every month
#
# Lambda execution role needs an inline policy like:
#   {
#     "Effect": "Allow",
#     "Action": "ec2:StartInstances",
#     "Resource": "arn:aws:ec2:<region>:<account-id>:instance/<instance-id>"
#   }
#
# Environment variable:
#   INSTANCE_ID = i-0123456789abcdef0
# =============================================================================

import os

import boto3

ec2 = boto3.client("ec2")


def handler(event, context):
    instance_id = os.environ["INSTANCE_ID"]

    response = ec2.start_instances(InstanceIds=[instance_id])
    state = response["StartingInstances"][0]["CurrentState"]["Name"]

    print(f"Requested start for {instance_id} -> current state: {state}")
    return {"instance_id": instance_id, "state": state}
