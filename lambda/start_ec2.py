import os

import boto3

ec2 = boto3.client("ec2")


def handler(event, context):
    instance_id = os.environ["INSTANCE_ID"]

    response = ec2.start_instances(InstanceIds=[instance_id])
    state = response["StartingInstances"][0]["CurrentState"]["Name"]

    print(f"Requested start for {instance_id} -> current state: {state}")
    return {"instance_id": instance_id, "state": state}
