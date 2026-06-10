# =============================================================================
# Lambda: backup force-stop for the Airflow EC2 instance
# =============================================================================
# Trigger: EventBridge rule fired 8 hours after the start rule, e.g.
#   Start rule:  cron(0 6 1 * ? *)   -> 06:00 UTC on the 1st
#   Stop  rule:  cron(0 14 1 * ? *)  -> 14:00 UTC on the 1st
#
# This is a safety net: the instance normally stops itself via the
# stop_self task in dags/training.py. This Lambda fires regardless
# of what happened inside Airflow (crash, hang, failed task).
#
# Lambda execution role needs:
#   {"Effect": "Allow", "Action": ["ec2:StopInstances", "ec2:DescribeInstances"],
#    "Resource": "arn:aws:ec2:<region>:<account-id>:instance/<instance-id>"}
#
# Environment variable:
#   INSTANCE_ID = i-0123456789abcdef0
# =============================================================================

import os

import boto3

ec2 = boto3.client("ec2")


def handler(event, context):
    instance_id = os.environ["INSTANCE_ID"]

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]

    if state in ("stopped", "stopping"):
        print(f"{instance_id} is already {state} — nothing to do")
        return {"instance_id": instance_id, "state": state, "action": "skipped"}

    response = ec2.stop_instances(InstanceIds=[instance_id])
    new_state = response["StoppingInstances"][0]["CurrentState"]["Name"]
    print(f"Backup stop triggered for {instance_id} -> {new_state}")
    return {"instance_id": instance_id, "state": new_state, "action": "stopped"}
