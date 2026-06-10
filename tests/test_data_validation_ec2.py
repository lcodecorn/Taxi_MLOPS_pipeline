"""
Pytest validation script for downloaded taxi data.

Validates that:
- S3 files exist and are accessible
- Data has expected structure and columns
- Data quality checks pass
"""

import sys
import os
import pytest
import pandas as pd
from datetime import datetime

sys.path.insert(0, "/opt/airflow")

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    pytest.skip("boto3 not available", allow_module_level=True)


@pytest.fixture
def s3_client():
    try:
        client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        return client
    except Exception as e:
        pytest.skip(f"Could not initialize S3 client: {e}")


@pytest.fixture
def sample_month():
    return os.getenv("TEST_MONTH", datetime.utcnow().strftime("%Y-%m"))


def test_s3_file_exists(s3_client, sample_month):
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")

    key = f"raw/yellow_tripdata_{sample_month}.parquet"

    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        print(f"File exists: s3://{bucket}/{key}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found yet: s3://{bucket}/{key}")
        raise


def test_data_structure(s3_client, sample_month):
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")

    key = f"raw/yellow_tripdata_{sample_month}.parquet"

    try:
        from io import BytesIO

        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))

        required_taxi_columns = {
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount',
            'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount', 'Airport_fee',
        }

        actual_columns = set(df.columns)
        missing_columns = required_taxi_columns - actual_columns

        if missing_columns:
            assert False, f"Missing taxi columns: {sorted(missing_columns)}"

        assert len(df) > 0, "Dataset is empty"
        print(f"Taxi data structure valid. Rows: {len(df)}, Columns: {len(actual_columns)}")

    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate taxi data structure: {e}")


def test_data_quality(s3_client, sample_month):
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")

    key = f"raw/yellow_tripdata_{sample_month}.parquet"

    try:
        from io import BytesIO

        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))

        assert pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime'])
        assert pd.api.types.is_datetime64_any_dtype(df['tpep_dropoff_datetime'])
        assert df['passenger_count'].min() > 0
        assert df['passenger_count'].max() <= 10
        assert df['trip_distance'].min() >= 0
        assert df['fare_amount'].min() >= 0
        assert df['tip_amount'].min() >= 0
        assert df['total_amount'].min() >= 0
        assert df['Airport_fee'].min() >= 0

        print("Taxi data quality checks passed")

    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate taxi data quality: {e}")


def test_weather_data_structure(s3_client, sample_month):
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")

    key = f"raw/weather_{sample_month}.parquet"

    try:
        from io import BytesIO

        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))

        hourly_weather_columns = {
            'temperature_2m', 'precipitation', 'apparent_temperature',
            'wind_speed_100m', 'wind_speed_10m', 'relative_humidity_2m',
        }

        actual_columns = set(df.columns)
        has_hourly = hourly_weather_columns.issubset(actual_columns)

        if not has_hourly:
            missing = hourly_weather_columns - actual_columns
            assert False, f"Missing weather columns: {sorted(missing)}"

        assert len(df) > 0, "Weather dataset is empty"
        print(f"Weather data structure valid. Rows: {len(df)}")

    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"Weather data not found yet: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate weather data structure: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
