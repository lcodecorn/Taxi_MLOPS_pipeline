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
