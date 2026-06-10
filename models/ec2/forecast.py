"""""
NYC TAXI INTELLIGENCE ENGINE — FORECAST PIPELINE
Loads the 4 trained models and generates ranked zone-hour opportunities
for the next N hours.
Usage:
  python models/forecast.py                       # next 24 hours, all zones
  python models/forecast.py --hours 48            # next 48 hours
  python models/forecast.py --hours 12 --top 20   # top 20 zones only
  python models/forecast.py --start "2024-06-01 08:00"
"""""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import joblib
import boto3
import io
import sys
from dotenv import load_dotenv
import os
import mlflow
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("taxi_forecasting")

warnings.filterwarnings("ignore")

MODEL_S3_PREFIX = "models/"
DATA_S3_KEY = "final/hourly_zone_timeseries.parquet"
TMP_TIMESERIES = "/tmp/hourly_zone_timeseries.parquet"
OUTPUT_S3_PREFIX = "forecasts/"
OUTPUT_DIR = Path("forecasts")
OUTPUT_DIR.mkdir(exist_ok=True)
s3 = boto3.client("s3")

SCORE_WEIGHTS = {
    "demand":      0.35,
    "profit":      0.35,
    "competition": 0.20,
    "tip":         0.10,
}

# Same-hour aggregates that training only sees as lag1h_* (see preprocess/features.py).
# At forecast time these are reconstructed by shifting the raw historical
# columns forward by 1h, same as lag_1h_rph / lag_1h_trip_count.
RAW_LAG1H_COLS = [
    "avg_revenue", "avg_tip", "tip_rate",
    "avg_distance_mi", "avg_duration_min", "avg_speed_mph",
    "congestion_score", "dropoff_count", "net_pickup_flow",
    "frac_airport_pickup", "frac_airport_dropoff", "demand_vs_baseline",
    "precipitation", "temperature", "wind_speed", "frac_raining",
]


def download_timeseries_from_s3() -> None:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=DATA_S3_KEY)
    with open(TMP_TIMESERIES, "wb") as f:
        f.write(obj["Body"].read())


def load_models():
    print("Loading models from S3...")
    def load_s3_joblib(key):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_S3_PREFIX + key)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    models = {
        "demand":      load_s3_joblib("demand_model.pkl"),
        "profit":      load_s3_joblib("profitability_model.pkl"),
        "competition": load_s3_joblib("competition_model.pkl"),
        "tip":         load_s3_joblib("tip_model.pkl"),
        "tip_imputer": load_s3_joblib("tip_imputer.pkl"),
        "zone_enc":    load_s3_joblib("zone_encoder.pkl"),
        "borough_enc": load_s3_joblib("borough_encoder.pkl"),
    }
    features = load_s3_joblib("feature_list.pkl")
    metadata = load_s3_joblib("metadata.pkl")
    print(f"  {len(features)} features | trained on {metadata['train_size']:,} rows")
    return models, features


def load_history(zone_enc, borough_enc):
    print("Loading historical context from S3...")
    download_timeseries_from_s3()
    df = pd.read_parquet(TMP_TIMESERIES).sort_values("ts_hour")
    df["PUZone_enc"] = zone_enc.transform(df["PUZone"])
    df["PUBorough_enc"] = borough_enc.transform(df["PUBorough"])
    return df


def resolve_forecast_start(df_hist: pd.DataFrame, start_override: str | None) -> datetime:
    min_ts = pd.Timestamp(df_hist["ts_hour"].min())
    max_ts = pd.Timestamp(df_hist["ts_hour"].max())
    print(f"  Feature store range : {min_ts} → {max_ts}  ({len(df_hist):,} rows)")

    if start_override:
        start_dt = datetime.strptime(start_override, "%Y-%m-%d %H:%M")
        print(f"  Forecast start      : {start_dt}  (manual --start)")
        return start_dt

    start_dt = (max_ts + pd.Timedelta(hours=1)).to_pydatetime()
    print(f"  Forecast start      : {start_dt}  (auto: last hour + 1)")
    return start_dt


def build_future_frame(df_hist, start_dt, n_hours, zone_enc):
    future_hours = [start_dt + timedelta(hours=h) for h in range(n_hours)]
    zones = df_hist[["PUZone", "PUBorough", "PUZone_enc", "PUBorough_enc"]].drop_duplicates()

    rows = []
    for ts in future_hours:
        tmp = zones.copy()
        tmp["ts_hour"] = ts
        rows.append(tmp)

    future = pd.concat(rows, ignore_index=True)

    future["hour"]         = future["ts_hour"].dt.hour
    future["day_of_week"]  = future["ts_hour"].dt.dayofweek
    future["week_of_year"] = future["ts_hour"].dt.isocalendar().week.astype(int)
    future["month"]        = future["ts_hour"].dt.month
    future["quarter"]      = future["ts_hour"].dt.quarter
    future["day_of_year"]  = future["ts_hour"].dt.dayofyear
    future["is_weekend"] = (future["day_of_week"] >= 5).astype(int)
    future["is_rush"]    = future["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    future["is_night"]   = (future["hour"].isin(range(22, 24)) | future["hour"].isin(range(0, 6))).astype(int)
    future["hour_sin"] = np.sin(2 * np.pi * future["hour"] / 24)
    future["hour_cos"] = np.cos(2 * np.pi * future["hour"] / 24)
    future["dow_sin"]  = np.sin(2 * np.pi * future["day_of_week"] / 7)
    future["dow_cos"]  = np.cos(2 * np.pi * future["day_of_week"] / 7)

    return future


def attach_lag_features(future, df_hist):
    raw_lag1h_cols = [c for c in RAW_LAG1H_COLS if c in df_hist.columns]
    hist = df_hist[["PUZone", "ts_hour", "trip_count", "target_rph"] + raw_lag1h_cols].copy()
    hist = hist.sort_values(["PUZone", "ts_hour"])

    grp = hist.groupby("PUZone", sort=False)
    hist["rolling_24h_trip_mean"] = grp["trip_count"].transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    hist["rolling_168h_trip_mean"] = grp["trip_count"].transform(lambda s: s.shift(1).rolling(168, min_periods=1).mean())
    hist["rolling_24h_rph_mean"] = grp["target_rph"].transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    hist["rolling_7d_rph_mean"] = grp["target_rph"].transform(lambda s: s.shift(1).rolling(168, min_periods=1).mean())
    hist["rolling_24h_rph_std"] = grp["target_rph"].transform(lambda s: s.shift(1).rolling(24, min_periods=2).std())

    def merge_lag(df, offset_hours, cols, suffixes):
        shifted = hist[["PUZone", "ts_hour"] + cols].copy()
        shifted["ts_hour"] = shifted["ts_hour"] + pd.Timedelta(hours=offset_hours)
        shifted = shifted.rename(columns=dict(zip(cols, suffixes)))
        return df.merge(shifted, on=["PUZone", "ts_hour"], how="left")

    future = merge_lag(future, 1,   ["trip_count"], ["lag_1h_trip_count"])
    future = merge_lag(future, 2,   ["trip_count"], ["lag_2h_trip_count"])
    future = merge_lag(future, 24,  ["trip_count"], ["lag_24h_trip_count"])
    future = merge_lag(future, 24,  ["rolling_24h_trip_mean"], ["rolling_24h_trip_mean"])
    future = merge_lag(future, 24,  ["rolling_168h_trip_mean"], ["rolling_168h_trip_mean"])
    future = merge_lag(future, 1, ["target_rph", "rolling_24h_rph_mean", "rolling_7d_rph_mean", "rolling_24h_rph_std"], ["lag_1h_rph", "rolling_24h_rph_mean", "rolling_7d_rph_mean", "rolling_24h_rph_std"])
    future = merge_lag(future, 2,   ["target_rph"], ["lag_2h_rph"])
    future = merge_lag(future, 24,  ["target_rph"], ["lag_24h_rph"])
    future = merge_lag(future, 168, ["target_rph"], ["lag_168h_rph"])

    # Same-hour aggregates -> lag1h_* (mirrors preprocess/features.py shift)
    future = merge_lag(future, 1, raw_lag1h_cols, [f"lag1h_{c}" for c in raw_lag1h_cols])

    future["trip_count_delta_1h"] = future["lag_1h_trip_count"] - future["lag_2h_trip_count"]
    future["rph_delta_1h"]        = future["lag_1h_rph"] - future["lag_2h_rph"]
    future["rph_zscore"] = (future["lag_1h_rph"] - future["rolling_24h_rph_mean"]) / (future["rolling_24h_rph_std"].fillna(0) + 1e-6)
    future["rph_cv"] = future["rolling_24h_rph_std"].fillna(0) / (future["rolling_24h_rph_mean"].fillna(0) + 1e-6)
    future["relative_zone_activity"] = future["lag_1h_trip_count"] / (future["rolling_168h_trip_mean"].replace(0, np.nan))

    return future


def fill_missing(future, df_hist, features):
    hist_features = [c for c in features if c in df_hist.columns]
    zone_medians = (
        df_hist.groupby("PUZone")[hist_features].median().reset_index()
        .rename(columns={c: f"__med_{c}" for c in hist_features})
    )
    future = future.merge(zone_medians, on="PUZone", how="left")

    for col in features:
        if col not in future.columns:
            future[col] = np.nan
        null_mask = future[col].isna()
        if null_mask.any():
            med_col = f"__med_{col}"
            if med_col in future.columns:
                future.loc[null_mask, col] = future.loc[null_mask, med_col]
            still_null = future[col].isna()
            if still_null.any():
                global_med = df_hist[col].median() if col in df_hist.columns else 0.0
                future[col] = future[col].fillna(global_med if not np.isnan(global_med) else 0.0)

    future = future.drop(columns=[c for c in future.columns if c.startswith("__med_")])
    return future


def run_inference(future, models, features, df_hist):
    future = fill_missing(future, df_hist, features)
    X = future[features].copy()
    future["pred_demand"] = models["demand"].predict(X)
    future["pred_rph"] = models["profit"].predict(X)
    future["pred_competition"] = models["competition"].predict(X)
    X_tip = models["tip_imputer"].transform(X)
    future["pred_tip_prob"] = models["tip"].predict_proba(X_tip)[:, 1]
    return future


OUTPUT_COLS = [
    "ts_hour", "PUZone", "PUBorough",
    "rank", "opportunity_score",
    "pred_demand", "pred_rph", "pred_competition", "pred_tip_prob",
    "score_demand", "score_profit", "score_competition", "score_tip",
]


def score_and_rank(future):
    def minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    future["score_demand"]      = minmax(future["pred_demand"])
    future["score_profit"]      = minmax(future["pred_rph"])
    future["score_competition"] = 1 - minmax(future["pred_competition"])
    future["score_tip"]         = minmax(future["pred_tip_prob"])
    future["opportunity_score"] = (
        SCORE_WEIGHTS["demand"]      * future["score_demand"]
        + SCORE_WEIGHTS["profit"]    * future["score_profit"]
        + SCORE_WEIGHTS["competition"] * future["score_competition"]
        + SCORE_WEIGHTS["tip"]       * future["score_tip"]
    )
    future["rank"] = (
        future.groupby("ts_hour")["opportunity_score"]
        .rank(ascending=False, method="first").astype(int)
    )
    return future.sort_values(["ts_hour", "rank"])


def save_forecast(df_out, n_hours, top_n, run_metadata: dict):
    if top_n:
        df_out = df_out[df_out["rank"] <= top_n]

    cols = [c for c in OUTPUT_COLS if c in df_out.columns]
    df_out = df_out[cols].round(4)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"forecast_{n_hours}h_{ts_tag}.csv"
    df_out.to_csv(out_csv, index=False)

    meta_path = OUTPUT_DIR / f"forecast_{n_hours}h_{ts_tag}.meta.json"
    meta_path.write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")

    csv_key = f"{OUTPUT_S3_PREFIX}forecast_{n_hours}h_{ts_tag}.csv"
    meta_key = f"{OUTPUT_S3_PREFIX}forecast_{n_hours}h_{ts_tag}.meta.json"
    with open(out_csv, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, csv_key)
    with open(meta_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, meta_key)
    print(f"Uploaded → s3://{S3_BUCKET}/{csv_key}")

    return out_csv, meta_path, df_out


def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Forecast")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--top",   type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    args = parser.parse_args()

    models, features = load_models()
    df_hist = load_history(models["zone_enc"], models["borough_enc"])
    start_dt = resolve_forecast_start(df_hist, args.start)

    future = build_future_frame(df_hist, start_dt, args.hours, models["zone_enc"])
    future = attach_lag_features(future, df_hist)
    future = run_inference(future, models, features, df_hist)
    future = score_and_rank(future)

    run_metadata = {
        "data_s3_key": DATA_S3_KEY,
        "forecast_start": str(pd.Timestamp(start_dt)),
        "forecast_hours": args.hours,
        "top_n": args.top,
    }

    with mlflow.start_run(run_name=f"taxi_forecast_{args.hours}h"):
        mlflow.set_tag("run_type", "forecast")
        mlflow.log_param("forecast_hours", args.hours)
        mlflow.log_param("top_n", args.top)

        out_csv, meta_path, df_out = save_forecast(future, args.hours, args.top, run_metadata)
        mlflow.log_metric("forecast_rows", len(df_out))
        mlflow.log_artifact(str(out_csv))
        mlflow.log_artifact(str(meta_path))


if __name__ == "__main__":
    main()
