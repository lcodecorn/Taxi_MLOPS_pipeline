"""Fetch NYC weather from Open-Meteo and upload to S3 for preprocess/merge.py."""

import os
from calendar import monthrange
from io import BytesIO

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 120


def month_date_bounds(month: str) -> tuple[str, str]:
    """Return inclusive ISO start/end dates for YYYY-MM."""
    year, mon = map(int, month.split("-"))
    last_day = monthrange(year, mon)[1]
    return f"{year:04d}-{mon:02d}-01", f"{year:04d}-{mon:02d}-{last_day:02d}"


def weather_s3_key_for_month(month: str) -> str:
    return f"raw/weather_{month}.parquet"


def fetch_weather_dataframe(start_date: str, end_date: str) -> pd.DataFrame:
    """Download hourly + daily weather for NYC (inclusive date range)."""
    params = {
        "latitude": 40.7143,
        "longitude": -74.006,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "sunset",
            "sunrise",
            "precipitation_hours",
        ],
        "hourly": [
            "temperature_2m",
            "precipitation",
            "apparent_temperature",
            "wind_speed_100m",
            "wind_speed_10m",
            "relative_humidity_2m",
        ],
        "timezone": "America/New_York",
    }
    print(f"Fetching Open-Meteo archive {start_date} → {end_date}")
    response = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    hourly_df = pd.DataFrame(
        {
            "date": pd.to_datetime(hourly["time"]),
            "temperature_2m": hourly["temperature_2m"],
            "precipitation": hourly["precipitation"],
            "apparent_temperature": hourly["apparent_temperature"],
            "wind_speed_100m": hourly["wind_speed_100m"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "type": "hourly",
        }
    )

    daily = payload["daily"]
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_2m_max": daily["temperature_2m_max"],
            "temperature_2m_min": daily["temperature_2m_min"],
            "apparent_temperature_mean": daily["apparent_temperature_mean"],
            "sunset": daily["sunset"],
            "sunrise": daily["sunrise"],
            "precipitation_hours": daily["precipitation_hours"],
            "type": "daily",
        }
    )

    combined = pd.concat([hourly_df, daily_df], ignore_index=True)
    print(f"Fetched {len(hourly_df):,} hourly + {len(daily_df):,} daily rows")
    return combined


def upload_weather_to_s3(
    df: pd.DataFrame,
    bucket: str | None = None,
    key: str | None = None,
) -> str:
    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set")
    if not key:
        raise ValueError("S3 key is required")

    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    try:
        boto3.client("s3").upload_fileobj(buffer, bucket, key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("AccessDenied", "403"):
            raise PermissionError(
                f"Cannot upload to s3://{bucket}/{key}: IAM user lacks s3:PutObject."
            ) from exc
        raise

    print(f"Uploaded weather to s3://{bucket}/{key} ({len(df):,} rows)")
    return key


def sync_weather_month_to_s3(
    month: str,
    bucket: str | None = None,
    skip_if_exists: bool = True,
) -> str:
    """
    Fetch weather for the same calendar month as the taxi trip file (YYYY-MM)
    and upload to s3://{bucket}/raw/weather_{month}.parquet.
    """
    start_date, end_date = month_date_bounds(month)
    bucket = bucket or os.getenv("S3_BUCKET")
    key = weather_s3_key_for_month(month)

    if skip_if_exists and bucket:
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            print(f"Weather already on S3, skipping: s3://{bucket}/{key}")
            return key
        except ClientError:
            pass

    df = fetch_weather_dataframe(start_date, end_date)
    df["trip_month"] = month
    return upload_weather_to_s3(df, bucket, key)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sync NYC weather for one month to S3")
    parser.add_argument(
        "--month",
        required=True,
        help="Trip month YYYY-MM (must match downloaded yellow_tripdata file)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if s3://$S3_BUCKET/raw/weather_{month}.parquet exists",
    )
    args = parser.parse_args()
    sync_weather_month_to_s3(args.month, skip_if_exists=not args.force)


if __name__ == "__main__":
    main()
