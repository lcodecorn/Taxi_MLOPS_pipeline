"""
features.py — Advanced feature engineering pipeline for NYC taxi
driver recommendation & forecasting system.

This pipeline builds a production-grade feature store for:

1. Demand forecasting
2. Revenue / profitability forecasting
3. Supply & competition estimation
4. Recommendation ranking systems

All computation runs entirely inside DuckDB.

───────────────────────────────────────────────────────────────────────────────

INPUT
    data/final/demand_dataset_enriched.parquet

OUTPUTS
    data/final/features.parquet
        Trip-level enriched features

    data/final/hourly_zone_timeseries.parquet
        Forecast-ready zone-hour feature store

───────────────────────────────────────────────────────────────────────────────

FEATURE GROUPS

✓ Revenue features
✓ Demand features
✓ Competition features
✓ Traffic / congestion features
✓ Weather features
✓ Airport features
✓ Lag features
✓ Rolling statistics
✓ Volatility indicators
✓ Momentum indicators
✓ Cyclical time encodings

───────────────────────────────────────────────────────────────────────────────

TARGET USE CASES

✓ XGBoost
✓ LightGBM
✓ CatBoost
✓ SARIMAX
✓ Prophet
✓ LSTM
✓ TFT
✓ Recommendation ranking systems
"""


from pathlib import Path
import duckdb
import pandas as pd
import boto3
import io
from dotenv import load_dotenv
import os

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client("s3")

# S3 paths
INPUT_S3_KEY = "final/demand_dataset_enriched.parquet"
OUT_TRIPS_S3_KEY = "final/features.parquet"
OUT_TS_S3_KEY = "final/hourly_zone_timeseries.parquet"

# Local temp paths (DuckDB reads/writes these; then upload to S3)
TMP_INPUT = "/tmp/input.parquet"
TMP_FEATURES = "/tmp/features.parquet"
TMP_TIMESERIES = "/tmp/hourly_zone_timeseries.parquet"


obj = s3.get_object(Bucket=S3_BUCKET, Key=INPUT_S3_KEY)
with open(TMP_INPUT, "wb") as f:
    f.write(obj["Body"].read())

# DuckDB setup
con = duckdb.connect()
con.execute("PRAGMA threads=8")


# Input row count
n_total = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{TMP_INPUT}')
""").fetchone()[0]
print(f"\nInput rows: {n_total:,}")

# 1. TRIP-LEVEL FEATURE ENGINEERING
print("\nBuilding trip-level features...")
con.execute(f"""
COPY (
    WITH base AS (
        SELECT *,
            epoch(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0 AS trip_duration_min,
            GREATEST(fare_amount + extra + tip_amount + tolls_amount + Airport_fee, 0) AS revenue
        FROM read_parquet('{TMP_INPUT}')
    ),
    features AS (
        SELECT *,
            CASE WHEN trip_duration_min > 1 THEN revenue / (trip_duration_min / 60.0) ELSE NULL END AS revenue_per_hour,
            date_trunc('hour', tpep_pickup_datetime) AS ts_hour,
            CAST(date_trunc('day', tpep_pickup_datetime) AS DATE) AS service_date,
            hour(tpep_pickup_datetime) AS hour,
            dayofweek(tpep_pickup_datetime) AS day_of_week,
            weekofyear(tpep_pickup_datetime) AS week_of_year,
            month(tpep_pickup_datetime) AS month,
            quarter(tpep_pickup_datetime) AS quarter,
            year(tpep_pickup_datetime) AS year,
            dayofyear(tpep_pickup_datetime) AS day_of_year,
            CASE WHEN dayofweek(tpep_pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 7 AND 9 THEN 1 ELSE 0 END AS is_am_rush,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END AS is_pm_rush,
            CASE WHEN hour(tpep_pickup_datetime) BETWEEN 7 AND 9 OR hour(tpep_pickup_datetime) BETWEEN 17 AND 19 THEN 1 ELSE 0 END AS is_rush,
            CASE WHEN hour(tpep_pickup_datetime) NOT BETWEEN 6 AND 21 THEN 1 ELSE 0 END AS is_night,
            SIN(2 * PI() * hour(tpep_pickup_datetime) / 24.0) AS hour_sin,
            COS(2 * PI() * hour(tpep_pickup_datetime) / 24.0) AS hour_cos,
            SIN(2 * PI() * dayofweek(tpep_pickup_datetime) / 7.0) AS dow_sin,
            COS(2 * PI() * dayofweek(tpep_pickup_datetime) / 7.0) AS dow_cos,
            CASE WHEN precipitation > 0 THEN 1 ELSE 0 END AS is_raining,
            CASE WHEN apparent_temperature < 5 THEN 1 ELSE 0 END AS is_cold,
            CASE WHEN apparent_temperature > 30 THEN 1 ELSE 0 END AS is_hot,
            CASE WHEN PUZone IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') THEN 1 ELSE 0 END AS is_airport_pickup,
            CASE WHEN DOZone IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') THEN 1 ELSE 0 END AS is_airport_dropoff
        FROM base
    )
    SELECT * FROM features
    WHERE trip_duration_min BETWEEN 2 AND 180
      AND trip_distance BETWEEN 0.1 AND 100
      AND revenue BETWEEN 3 AND 500
      AND revenue_per_hour BETWEEN 5 AND 500
) TO '{TMP_FEATURES}' (FORMAT PARQUET)
""")

with open(TMP_FEATURES, "rb") as f:
    s3.upload_fileobj(f, S3_BUCKET, OUT_TRIPS_S3_KEY)
print(f"Uploaded features.parquet to s3://{S3_BUCKET}/{OUT_TRIPS_S3_KEY}")

# 2. HOURLY ZONE FEATURE STORE
print("\nBuilding hourly forecasting feature store...")

con.execute(f"""
COPY (

    --------------------------------------------------------------------------
    -- Pickup aggregation
    --------------------------------------------------------------------------

    WITH pickups AS (

        SELECT

            ------------------------------------------------------------------
            -- Keys
            ------------------------------------------------------------------

            PUZone,
            PUBorough,
            ts_hour,
            service_date,

            ------------------------------------------------------------------
            -- Calendar
            ------------------------------------------------------------------

            hour,
            day_of_week,
            week_of_year,
            month,
            quarter,
            year,
            day_of_year,

            is_weekend,
            is_rush,
            is_night,

            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,

            ------------------------------------------------------------------
            -- Demand
            ------------------------------------------------------------------

            COUNT(*) AS trip_count,

            ------------------------------------------------------------------
            -- Revenue metrics
            ------------------------------------------------------------------

            AVG(revenue_per_hour)
                AS target_rph,

            MEDIAN(revenue_per_hour)
                AS median_rph,

            QUANTILE_CONT(revenue_per_hour, 0.75)
                AS p75_rph,

            AVG(revenue)
                AS avg_revenue,

            AVG(tip_amount)
                AS avg_tip,

            AVG(
                CASE
                    WHEN tip_amount > 0
                    THEN 1.0 ELSE 0.0
                END
            ) AS tip_rate,

            ------------------------------------------------------------------
            -- Trip metrics
            ------------------------------------------------------------------

            AVG(trip_distance)
                AS avg_distance_mi,

            AVG(trip_duration_min)
                AS avg_duration_min,

            ------------------------------------------------------------------
            -- Weather
            ------------------------------------------------------------------

            AVG(precipitation)
                AS precipitation,

            AVG(apparent_temperature)
                AS temperature,

            AVG(wind_speed_10m)
                AS wind_speed,

            AVG(is_raining::DOUBLE)
                AS frac_raining,

            ------------------------------------------------------------------
            -- Airport activity
            ------------------------------------------------------------------

            AVG(is_airport_pickup::DOUBLE)
                AS frac_airport_pickup,

            AVG(is_airport_dropoff::DOUBLE)
                AS frac_airport_dropoff

        FROM read_parquet('{TMP_FEATURES}')

        GROUP BY
            PUZone,
            PUBorough,
            ts_hour,
            service_date,
            hour,
            day_of_week,
            week_of_year,
            month,
            quarter,
            year,
            day_of_year,
            is_weekend,
            is_rush,
            is_night,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos
    ),

    --------------------------------------------------------------------------
    -- Dropoff aggregation
    --------------------------------------------------------------------------

    dropoffs AS (

        SELECT
            DOZone AS zone,
            ts_hour,
            COUNT(*) AS dropoff_count

        FROM read_parquet('{TMP_FEATURES}')

        GROUP BY
            DOZone,
            ts_hour
    ),

    --------------------------------------------------------------------------
    -- Merge pickups + dropoffs
    --------------------------------------------------------------------------

    merged AS (

        SELECT

            p.*,

            COALESCE(d.dropoff_count, 0)
                AS dropoff_count,

            ------------------------------------------------------------------
            -- Supply / competition features
            ------------------------------------------------------------------

            p.trip_count
            - COALESCE(d.dropoff_count, 0)
                AS net_pickup_flow,

            CASE
                WHEN COALESCE(d.dropoff_count, 0) > 0
                THEN p.trip_count::DOUBLE
                     / d.dropoff_count
                ELSE NULL
            END AS pickup_dropoff_ratio

        FROM pickups p

        LEFT JOIN dropoffs d
            ON p.PUZone = d.zone
           AND p.ts_hour = d.ts_hour
    ),

    --------------------------------------------------------------------------
    -- Lag features
    --------------------------------------------------------------------------

    lagged AS (

        SELECT
            *,

            ------------------------------------------------------------------
            -- Revenue lags
            ------------------------------------------------------------------

            LAG(target_rph, 1)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_1h_rph,

            LAG(target_rph, 2)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_2h_rph,

            LAG(target_rph, 24)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_24h_rph,

            LAG(target_rph, 168)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_168h_rph,

            ------------------------------------------------------------------
            -- Demand lags
            ------------------------------------------------------------------

            LAG(trip_count, 1)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_1h_trip_count,

            LAG(trip_count, 2)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_2h_trip_count,

            LAG(trip_count, 24)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_24h_trip_count,

            LAG(dropoff_count, 1)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS lag_1h_dropoff_count,

            ------------------------------------------------------------------
            -- Rolling demand
            ------------------------------------------------------------------

            AVG(trip_count)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS rolling_24h_trip_mean,

            AVG(trip_count)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS rolling_168h_trip_mean,

            ------------------------------------------------------------------
            -- Rolling revenue
            ------------------------------------------------------------------

            AVG(target_rph)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS rolling_24h_rph_mean,

            AVG(target_rph)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
                ) AS rolling_7d_rph_mean,

            ------------------------------------------------------------------
            -- Rolling volatility
            ------------------------------------------------------------------

            STDDEV_SAMP(target_rph)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS rolling_24h_rph_std,

            ------------------------------------------------------------------
            -- Rolling airport activity
            ------------------------------------------------------------------

            AVG(frac_airport_pickup)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS rolling_airport_activity

        FROM merged
    ),

    --------------------------------------------------------------------------
    -- Advanced derived features
    --------------------------------------------------------------------------

    final_features AS (

        SELECT
            *,

            ------------------------------------------------------------------
            -- Demand momentum
            ------------------------------------------------------------------

            lag_1h_trip_count - lag_2h_trip_count
                AS trip_count_delta_1h,

            CASE
                WHEN rolling_24h_trip_mean > 0
                THEN trip_count
                     / rolling_24h_trip_mean
                ELSE NULL
            END AS demand_vs_baseline,

            ------------------------------------------------------------------
            -- Revenue momentum
            ------------------------------------------------------------------

            target_rph - lag_1h_rph
                AS rph_delta_1h,

            ------------------------------------------------------------------
            -- Revenue surge z-score
            ------------------------------------------------------------------

            CASE
                WHEN rolling_24h_rph_std > 0
                THEN (
                    target_rph
                    - rolling_24h_rph_mean
                ) / rolling_24h_rph_std
                ELSE NULL
            END AS rph_zscore,

            ------------------------------------------------------------------
            -- Traffic speed estimate
            ------------------------------------------------------------------

            CASE
                WHEN avg_duration_min > 0
                THEN avg_distance_mi
                     / (avg_duration_min / 60.0)
                ELSE NULL
            END AS avg_speed_mph,

            ------------------------------------------------------------------
            -- Congestion score
            ------------------------------------------------------------------

            CASE
                WHEN avg_duration_min > 0
                     AND avg_distance_mi > 0
                THEN 1.0 /
                     (
                        avg_distance_mi
                        / (avg_duration_min / 60.0)
                     )
                ELSE NULL
            END AS congestion_score,

            ------------------------------------------------------------------
            -- Weather momentum
            ------------------------------------------------------------------

            precipitation
            - LAG(precipitation, 1)
                OVER (
                    PARTITION BY PUZone
                    ORDER BY ts_hour
                ) AS precipitation_delta,

            ------------------------------------------------------------------
            -- Revenue stability
            ------------------------------------------------------------------

            CASE
                WHEN rolling_24h_rph_mean > 0
                THEN rolling_24h_rph_std
                     / rolling_24h_rph_mean
                ELSE NULL
            END AS rph_cv,

            ------------------------------------------------------------------
            -- Relative zone activity
            ------------------------------------------------------------------

            CASE
                WHEN rolling_168h_trip_mean > 0
                THEN lag_1h_trip_count / rolling_168h_trip_mean
                ELSE NULL
            END AS relative_zone_activity

        FROM lagged
    ),

    --------------------------------------------------------------------------
    -- Shift same-hour aggregates back by 1h
    --
    -- avg_revenue, avg_tip, tip_rate, avg_distance_mi/duration/speed,
    -- congestion_score, dropoff_count, net_pickup_flow, frac_airport_*,
    -- precipitation/temperature/wind/frac_raining and demand_vs_baseline are
    -- all computed from trips that occur DURING ts_hour — the same hour as
    -- the trip_count / target_rph targets. A forecast made before ts_hour
    -- cannot know these values, so we expose lag1h_* versions (the prior
    -- hour's realized value) for use as model features.
    --------------------------------------------------------------------------

    shifted AS (
        SELECT
            *,
            LAG(frac_airport_pickup, 1)  OVER w AS lag1h_frac_airport_pickup,
            LAG(frac_airport_dropoff, 1) OVER w AS lag1h_frac_airport_dropoff,
            LAG(precipitation, 1)        OVER w AS lag1h_precipitation,
            LAG(temperature, 1)          OVER w AS lag1h_temperature,
            LAG(wind_speed, 1)           OVER w AS lag1h_wind_speed,
            LAG(frac_raining, 1)         OVER w AS lag1h_frac_raining,
            LAG(demand_vs_baseline, 1)   OVER w AS lag1h_demand_vs_baseline,
            LAG(dropoff_count, 1)        OVER w AS lag1h_dropoff_count,
            LAG(net_pickup_flow, 1)      OVER w AS lag1h_net_pickup_flow,
            LAG(avg_revenue, 1)          OVER w AS lag1h_avg_revenue,
            LAG(avg_tip, 1)              OVER w AS lag1h_avg_tip,
            LAG(tip_rate, 1)             OVER w AS lag1h_tip_rate,
            LAG(avg_distance_mi, 1)      OVER w AS lag1h_avg_distance_mi,
            LAG(avg_duration_min, 1)     OVER w AS lag1h_avg_duration_min,
            LAG(avg_speed_mph, 1)        OVER w AS lag1h_avg_speed_mph,
            LAG(congestion_score, 1)     OVER w AS lag1h_congestion_score
        FROM final_features
        WINDOW w AS (PARTITION BY PUZone ORDER BY ts_hour)
    )

    --------------------------------------------------------------------------
    -- Final dataset
    --------------------------------------------------------------------------

    SELECT *
    FROM shifted

    WHERE trip_count >= 5

    ORDER BY
        PUZone,
        ts_hour

) TO '{TMP_TIMESERIES}' (FORMAT PARQUET)
""")

with open(TMP_TIMESERIES, "rb") as f:
    s3.upload_fileobj(f, S3_BUCKET, OUT_TS_S3_KEY)
print(f"\nForecast-ready feature store uploaded to s3://{S3_BUCKET}/{OUT_TS_S3_KEY}")

# 3. DATASET SUMMARY
summary = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT PUZone) AS zones,
        MIN(ts_hour) AS min_ts,
        MAX(ts_hour) AS max_ts,
        AVG(target_rph) AS avg_rph,
        AVG(trip_count) AS avg_trip_count
    FROM read_parquet('{TMP_TIMESERIES}')
""").fetchdf()

print("\nDataset summary:")
print(summary.to_string(index=False))

# 4. PREVIEW
preview = con.execute(f"""
    SELECT

        PUZone,
        ts_hour,

        ----------------------------------------------------------------------
        -- Demand
        ----------------------------------------------------------------------

        trip_count,
        dropoff_count,
        net_pickup_flow,

        ----------------------------------------------------------------------
        -- Revenue
        ----------------------------------------------------------------------

        target_rph,
        lag_1h_rph,
        rolling_24h_rph_mean,

        ----------------------------------------------------------------------
        -- Competition
        ----------------------------------------------------------------------

        pickup_dropoff_ratio,

        ----------------------------------------------------------------------
        -- Traffic
        ----------------------------------------------------------------------

        avg_speed_mph,
        congestion_score,

        ----------------------------------------------------------------------
        -- Volatility
        ----------------------------------------------------------------------

        rph_zscore,
        rph_cv,

        ----------------------------------------------------------------------
        -- Weather
        ----------------------------------------------------------------------

        precipitation,
        temperature

    FROM read_parquet('{TMP_TIMESERIES}')

    ORDER BY ts_hour

    LIMIT 20
""").fetchdf()

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

print("\nSample rows:\n")

print(
    preview.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nDone.")

print("\nRecommended next steps:")
print("1. Train demand forecasting model")
print("2. Train revenue forecasting model")
print("3. Train competition/saturation model")
print("4. Build recommendation ranking engine")
print("5. Add complete hourly time grid")
print("6. Add neighboring zone spatial features")
