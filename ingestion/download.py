"""Download NYC taxi trip parquet files to S3."""

import argparse
import os
from datetime import datetime
from io import BytesIO
import re
from typing import Optional, List

import boto3
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
DATASET_TYPE = os.getenv("DATASET_TYPE", "yellow")
TIMEOUT = 60


class TripDataNotAvailableError(ValueError):
    """Raised when TLC has not published data for the requested month."""


def generate_month_list(start: str, end: str) -> list[str]:
    start_date = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")

    months = []
    current = start_date

    while current <= end_date:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


def trip_filename(dataset_type: str, month: str) -> str:
    return f"{dataset_type}_tripdata_{month}.parquet"


def trip_s3_key(dataset_type: str, month: str) -> str:
    return f"raw/{trip_filename(dataset_type, month)}"


def trip_url(dataset_type: str, month: str) -> str:
    return f"{BASE_URL}{trip_filename(dataset_type, month)}"


def assert_trip_data_available(month: str, dataset_type: str = DATASET_TYPE) -> None:
    """Fail fast with a helpful message if TLC has not published this month yet."""
    url = trip_url(dataset_type, month)
    response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
    if response.status_code == 200:
        return
    if response.status_code in (403, 404):
        raise TripDataNotAvailableError(
            f"No TLC trip file for {month} ({url} returned {response.status_code}). "
            "Data is usually published 1–2 months after the trip month. "
            "For a manual Airflow run: open Trigger DAG → Advanced options → "
            'set Logical date to the 1st of the month *after* your data '
            '(e.g. 2024-02-01 for January 2024), or pass trigger config '
            '{"month": "2024-01"}.'
        )
    response.raise_for_status()


def download_month_to_s3(
    month: str,
    bucket: str | None = None,
    dataset_type: str = DATASET_TYPE,
    skip_if_exists: bool = True,
) -> str:
    """Download one month of trip data and upload to S3. Returns the S3 object key."""
    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set")

    s3_key = trip_s3_key(dataset_type, month)
    s3 = boto3.client("s3")

    if skip_if_exists:
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            print(f"Already on S3, skipping download: s3://{bucket}/{s3_key}")
            return s3_key
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "403", "NoSuchKey"):
                raise

    assert_trip_data_available(month, dataset_type)

    url = trip_url(dataset_type, month)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                buffer.write(chunk)
        buffer.seek(0)

    try:
        s3.upload_fileobj(buffer, bucket, s3_key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("AccessDenied", "403"):
            raise PermissionError(
                f"Cannot upload to s3://{bucket}/{s3_key}: IAM user lacks s3:PutObject. "
                "Attach a policy that allows PutObject (and GetObject/ListBucket) on "
                f"arn:aws:s3:::{bucket}/* — see iam/s3-pipeline-policy.json in this repo."
            ) from exc
        raise

    print(f"Uploaded s3://{bucket}/{s3_key}")
    return s3_key


def download_range_to_s3(
    start: str,
    end: str,
    bucket: str | None = None,
    dataset_type: str = DATASET_TYPE,
) -> list[str]:
    """Download a range of months to S3. Returns uploaded S3 keys."""
    keys = []
    for month in generate_month_list(start, end):
        try:
            keys.append(download_month_to_s3(month, bucket, dataset_type))
        except TripDataNotAvailableError as exc:
            print(f"Skipping {month}: {exc}")
        except requests.HTTPError as exc:
            print(f"Skipping {month}: {exc}")
    return keys


def prune_s3_months(
    latest_month: str,
    keep_months: int = 12,
    bucket: Optional[str] = None,
    dataset_type: str = DATASET_TYPE,
) -> List[str]:
    """Delete raw trip parquet files on S3 older than the rolling window."""
    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set")

    s3 = boto3.client("s3")
    prefix = "raw/"

    latest_dt = datetime.strptime(latest_month, "%Y-%m")
    keep_set = set()
    cur = latest_dt
    for _ in range(keep_months):
        keep_set.add(cur.strftime("%Y-%m"))
        if cur.month == 1:
            cur = cur.replace(year=cur.year - 1, month=12)
        else:
            cur = cur.replace(month=cur.month - 1)

    deleted = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = re.search(rf"{re.escape(dataset_type)}_tripdata_(\d{{4}}-\d{{2}})\.parquet$", key)
            if not m:
                continue
            month = m.group(1)
            if month not in keep_set:
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                    deleted.append(key)
                    print(f"Deleted s3://{bucket}/{key}")
                except ClientError as exc:
                    print(f"Failed to delete s3://{bucket}/{key}: {exc}")

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYC taxi data to S3")
    parser.add_argument("--month", help="Single month YYYY-MM (used by Airflow)")
    parser.add_argument("--start", default=os.getenv("START_MONTH", "2024-01"))
    parser.add_argument("--end", default=os.getenv("END_MONTH", "2024-03"))
    args = parser.parse_args()

    if args.month:
        download_month_to_s3(args.month)
    else:
        download_range_to_s3(args.start, args.end)


if __name__ == "__main__":
    main()
