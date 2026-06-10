"""
Merge all raw trip parquet files from S3 with taxi zones and weather.
Writes final/demand_dataset_enriched.parquet back to S3.

Weather inputs: all raw/weather_YYYY-MM.parquet files (one per pipeline month).
"""

import os
import tempfile
from pathlib import Path

import boto3
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
RAW_PREFIX = os.getenv("RAW_S3_PREFIX", "raw/")

if not S3_BUCKET:
    raise ValueError("S3_BUCKET environment variable is not set")

s3 = boto3.client("s3")


def list_s3_keys(bucket: str, prefix: str, predicate) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if predicate(key):
                keys.append(key)
    return sorted(keys)


def list_trip_s3_keys(bucket: str, prefix: str) -> list[str]:
    keys = list_s3_keys(
        bucket,
        prefix,
        lambda k: k.endswith(".parquet") and "tripdata" in k,
    )
    if not keys:
        raise FileNotFoundError(
            f"No trip parquet files under s3://{bucket}/{prefix} "
            "(expected keys like raw/yellow_tripdata_YYYY-MM.parquet)"
        )
    return keys


def list_weather_s3_keys(bucket: str, prefix: str) -> list[str]:
    keys = list_s3_keys(
        bucket,
        prefix,
        lambda k: k.endswith(".parquet") and "weather_" in k and "tripdata" not in k,
    )
    if not keys:
        # Legacy single-file layout
        legacy = f"{prefix.rstrip('/')}/weather.parquet"
        try:
            s3.head_object(Bucket=bucket, Key=legacy)
            return [legacy]
        except Exception:
            pass
        raise FileNotFoundError(
            f"No weather parquet under s3://{bucket}/{prefix} "
            "(expected raw/weather_YYYY-MM.parquet from fetch_weather task)"
        )
    return keys


def download_s3_object(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        s3.download_fileobj(S3_BUCKET, key, f)


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    trips_dir = tmp_path / "trips"
    weather_dir = tmp_path / "weather"
    trips_dir.mkdir()
    weather_dir.mkdir()

    trip_keys = list_trip_s3_keys(S3_BUCKET, RAW_PREFIX)
    print(f"Merging {len(trip_keys)} trip file(s) from S3:")
    for key in trip_keys:
        print(f"  - {key}")
        download_s3_object(key, trips_dir / Path(key).name)

    weather_keys = list_weather_s3_keys(S3_BUCKET, RAW_PREFIX)
    print(f"Using {len(weather_keys)} weather file(s) from S3:")
    for key in weather_keys:
        print(f"  - {key}")
        download_s3_object(key, weather_dir / Path(key).name)

    zones = pd.read_csv(
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    )

    con = duckdb.connect()
    con.register("zones_df", zones)

    trips_glob = (trips_dir / "*.parquet").as_posix()
    weather_glob = (weather_dir / "*.parquet").as_posix()
    out_path = tmp_path / "demand_dataset_enriched.parquet"

    con.execute(f"""
        COPY (
            SELECT
                t.*,
                pu.Borough  AS PUBorough,
                pu.Zone     AS PUZone,
                do_.Borough AS DOBorough,
                do_.Zone    AS DOZone,
                hw.temperature_2m,
                hw.precipitation,
                hw.apparent_temperature,
                hw.wind_speed_10m,
                hw.wind_speed_100m,
                hw.relative_humidity_2m,
                dw.apparent_temperature_mean,
                dw.precipitation_hours,
                dw.sunrise,
                dw.sunset
            FROM read_parquet('{trips_glob}') t
            LEFT JOIN zones_df pu ON t.PULocationID = pu.LocationID
            LEFT JOIN zones_df do_ ON t.DOLocationID = do_.LocationID
            LEFT JOIN read_parquet('{weather_glob}') hw
                ON hw.type = 'hourly'
                AND date_trunc('hour', t.tpep_pickup_datetime) = hw.date
            LEFT JOIN read_parquet('{weather_glob}') dw
                ON dw.type = 'daily'
                AND t.tpep_pickup_datetime::DATE = dw.date::DATE
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)

    enriched_key = "final/demand_dataset_enriched.parquet"
    with open(out_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, enriched_key)

print(f"Uploaded demand_dataset_enriched.parquet to s3://{S3_BUCKET}/{enriched_key}")
print("Done!")
