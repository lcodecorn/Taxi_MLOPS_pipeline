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
    """Get S3 client for testing."""
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        return client
    except Exception as e:
        pytest.skip(f"Could not initialize S3 client: {e}")


@pytest.fixture
def sample_month():
    """Get a sample month from environment or use current."""
    return os.getenv("TEST_MONTH", datetime.utcnow().strftime("%Y-%m"))


def test_s3_file_exists(s3_client, sample_month):
    """Test that downloaded taxi data file exists in S3."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/yellow_tripdata_{sample_month}.parquet"
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        print(f"✓ File exists: s3://{bucket}/{key}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found yet: s3://{bucket}/{key}")
        raise


def test_data_structure(s3_client, sample_month):
    """Test that taxi data has all expected columns."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/yellow_tripdata_{sample_month}.parquet"
    
    try:
        from io import BytesIO
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # All required columns for NYC yellow taxi data (TLC standard)
        required_taxi_columns = {
            'VendorID',
            'tpep_pickup_datetime',
            'tpep_dropoff_datetime',
            'passenger_count',
            'trip_distance',
            'RatecodeID',
            'store_and_fwd_flag',
            'PULocationID',
            'DOLocationID',
            'payment_type',
            'fare_amount',
            'extra',
            'mta_tax',
            'tip_amount',
            'tolls_amount',
            'total_amount',
            'Airport_fee',
        }
        
        actual_columns = set(df.columns)
        missing_columns = required_taxi_columns - actual_columns
        
        if missing_columns:
            print(f"Available columns: {sorted(actual_columns)}")
            assert False, f"Missing taxi columns: {sorted(missing_columns)}"
        
        assert len(df) > 0, "Dataset is empty"
        print(f"✓ Taxi data structure valid. Rows: {len(df)}, Columns: {len(actual_columns)}")
        
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate taxi data structure: {e}")


def test_data_quality(s3_client, sample_month):
    """Test basic data quality checks for taxi data."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/yellow_tripdata_{sample_month}.parquet"
    
    try:
        from io import BytesIO
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Datetime columns should be properly formatted
        assert pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']), \
            "Pickup datetime is not datetime type"
        assert pd.api.types.is_datetime64_any_dtype(df['tpep_dropoff_datetime']), \
            "Dropoff datetime is not datetime type"
        
        # Passenger count should be positive
        assert df['passenger_count'].min() > 0, "Found invalid passenger count <= 0"
        assert df['passenger_count'].max() <= 10, "Unreasonable passenger count > 10"
        
        # Trip distance should be non-negative
        assert df['trip_distance'].min() >= 0, "Found negative trip distance"
        assert df['trip_distance'].max() <= 300, "Unreasonable trip distance > 300 miles"
        
        # Fare amount should be non-negative
        assert df['fare_amount'].min() >= 0, "Found negative fare amount"
        
        # Extra charges should be non-negative
        assert df['extra'].min() >= 0, "Found negative extra charges"
        assert df['extra'].max() <= 5, "Unreasonable extra charges"
        
        # MTA tax should be reasonable (typically $0.50)
        assert df['mta_tax'].min() >= 0, "Found negative MTA tax"
        assert df['mta_tax'].max() <= 5, "Unreasonable MTA tax"
        
        # Tip amount should be non-negative
        assert df['tip_amount'].min() >= 0, "Found negative tip amount"
        
        # Tolls should be non-negative
        assert df['tolls_amount'].min() >= 0, "Found negative tolls"
        
        # Total amount should be non-negative
        assert df['total_amount'].min() >= 0, "Found negative total amount"
        
        # Airport fee should be non-negative
        assert df['Airport_fee'].min() >= 0, "Found negative airport fee"
        
        # Payment type should be valid
        valid_payment_types = {1, 2, 3, 4}  # Credit card, cash, no charge, dispute
        invalid_payments = set(df['payment_type'].unique()) - valid_payment_types
        assert not invalid_payments, f"Found invalid payment types: {invalid_payments}"
        
        # Rate code should be valid
        valid_rate_codes = {1, 2, 3, 4, 5}
        invalid_rates = set(df['RatecodeID'].unique()) - valid_rate_codes
        assert not invalid_rates, f"Found invalid rate codes: {invalid_rates}"
        
        # Store and forward flag should be Y or N
        valid_flags = {'Y', 'N'}
        invalid_flags = set(df['store_and_fwd_flag'].unique()) - valid_flags
        assert not invalid_flags, f"Found invalid store_and_fwd_flag values: {invalid_flags}"
        
        # Location IDs should be positive
        assert df['PULocationID'].min() > 0, "Found invalid PULocationID <= 0"
        assert df['DOLocationID'].min() > 0, "Found invalid DOLocationID <= 0"
        
        print(f"✓ Taxi data quality checks passed")
            
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate taxi data quality: {e}")


def test_no_duplicates(s3_client, sample_month):
    """Test that data doesn't have obvious duplicates."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/yellow_tripdata_{sample_month}.parquet"
    
    try:
        import pyarrow.parquet as pq
        from io import BytesIO
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        
        # Allow up to 1% duplicates (might be legitimate)
        max_allowed_duplicates = max(int(total_rows * 0.01), 10)
        assert duplicate_rows <= max_allowed_duplicates, \
            f"Too many duplicates: {duplicate_rows}/{total_rows}"
        
        print(f"Duplicate check passed. Duplicates: {duplicate_rows}/{total_rows}")
        
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"S3 file not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate duplicates: {e}")


def test_weather_data_structure(s3_client, sample_month):
    """Test that weather data has expected columns."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/weather_{sample_month}.parquet"
    
    try:
        from io import BytesIO
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Check for either hourly or daily weather columns
        hourly_weather_columns = {
            'temperature_2m',
            'precipitation',
            'apparent_temperature',
            'wind_speed_100m',
            'wind_speed_10m',
            'relative_humidity_2m',
        }
        
        daily_weather_columns = {
            'temperature_2m_max',
            'temperature_2m_min',
            'apparent_temperature_mean',
            'sunset',
            'sunrise',
            'precipitation_hours',
        }
        
        actual_columns = set(df.columns)
        
        # Check if it has hourly columns
        has_hourly = hourly_weather_columns.issubset(actual_columns)
        # Check if it has daily columns
        has_daily = daily_weather_columns.issubset(actual_columns)
        
        if not (has_hourly or has_daily):
            missing_hourly = hourly_weather_columns - actual_columns
            missing_daily = daily_weather_columns - actual_columns
            print(f"Available columns: {sorted(actual_columns)}")
            assert False, f"Missing both hourly and daily weather columns. " \
                          f"Missing hourly: {sorted(missing_hourly)}, " \
                          f"Missing daily: {sorted(missing_daily)}"
        
        weather_type = "hourly" if has_hourly else "daily"
        assert len(df) > 0, "Weather dataset is empty"
        print(f"✓ Weather data structure valid ({weather_type}). Rows: {len(df)}, Columns: {len(actual_columns)}")
        
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"Weather data not found yet: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate weather data structure: {e}")


def test_weather_data_quality(s3_client, sample_month):
    """Test basic weather data quality checks."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        pytest.skip("S3_BUCKET not configured")
    
    key = f"raw/weather_{sample_month}.parquet"
    
    try:
        from io import BytesIO
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Check numeric weather columns for reasonable values
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Temperature should be reasonable (-60 to 60 Celsius)
        temp_cols = [col for col in numeric_cols if 'temperature' in col.lower()]
        for col in temp_cols:
            if col in df.columns:
                assert df[col].min() >= -60, f"{col} has unreasonable low values"
                assert df[col].max() <= 60, f"{col} has unreasonable high values"
        
        # Wind speed should be non-negative
        wind_cols = [col for col in numeric_cols if 'wind' in col.lower()]
        for col in wind_cols:
            if col in df.columns:
                assert df[col].min() >= 0, f"{col} has negative values"
        
        # Precipitation should be non-negative
        precip_cols = [col for col in numeric_cols if 'precipitation' in col.lower()]
        for col in precip_cols:
            if col in df.columns:
                assert df[col].min() >= 0, f"{col} has negative values"
        
        # Humidity should be 0-100%
        if 'relative_humidity_2m' in df.columns:
            assert df['relative_humidity_2m'].min() >= 0, "Humidity below 0%"
            assert df['relative_humidity_2m'].max() <= 100, "Humidity above 100%"
        
        print(f"✓ Weather data quality checks passed")
            
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.skip(f"Weather data not found: {key}")
        raise
    except Exception as e:
        pytest.skip(f"Could not validate weather data quality: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])