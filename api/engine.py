"""""
NYC TAXI INTELLIGENCE ENGINE — FORECAST PIPELINE (API-friendly)
Models and historical feature store are loaded once and cached in-process.
generate_forecast(...) returns a ranked DataFrame for any time window.
No CLI / mlflow / local file writes — the API layer owns persistence.
"""""

import io
import os
import tempfile
import warnings
from datetime import datetime, timedelta
from typing import Optional

import boto3
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
warnings.filterwarnings("ignore")

MODEL_S3_PREFIX = "models/"
DATA_S3_KEY = "final/hourly_zone_timeseries.parquet"
TMP_TIMESERIES = os.path.join(tempfile.gettempdir(), "hourly_zone_timeseries.parquet")

s3 = boto3.client("s3")

FORECAST_TARGETS = {
    "demand":      "pred_demand",
    "profit":      "pred_rph",
    "competition": "pred_competition",
    "tip":         "pred_tip_prob",
    "opportunity": "opportunity_score",
}

SCORE_WEIGHTS = {
    "demand":      0.35,
    "profit":      0.35,
    "competition": 0.20,
    "tip":         0.10,
}

_CACHE: dict = {}


def download_timeseries_from_s3() -> None:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=DATA_S3_KEY)
    with open(TMP_TIMESERIES, "wb") as f:
        f.write(obj["Body"].read())


def load_models() -> tuple[dict, list[str]]:
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

    if not hasattr(models["tip"], "multi_class"):
        models["tip"].multi_class = "auto"

    return models, features


def load_history(zone_enc, borough_enc) -> pd.DataFrame:
    download_timeseries_from_s3()
    df = pd.read_parquet(TMP_TIMESERIES).sort_values("ts_hour")

    known_mask = (
        df["PUZone"].notna() & df["PUZone"].isin(zone_enc.classes_)
        & df["PUBorough"].notna() & df["PUBorough"].isin(borough_enc.classes_)
    )
    dropped = (~known_mask).sum()
    if dropped:
        print(f"  Dropping {dropped:,} history rows with zones unseen by the trained encoders")
    df = df[known_mask]

    df["PUZone_enc"] = zone_enc.transform(df["PUZone"])
    df["PUBorough_enc"] = borough_enc.transform(df["PUBorough"])
    return df


def get_pipeline_state() -> dict:
    """Load models + historical feature store once and cache for the process lifetime."""
    if not _CACHE:
        models, features = load_models()
        df_hist = load_history(models["zone_enc"], models["borough_enc"])
        _CACHE["models"] = models
        _CACHE["features"] = features
        _CACHE["df_hist"] = df_hist
    return _CACHE


def refresh_pipeline_state() -> dict:
    """Force a reload of models + history (e.g. after a retrain or new data drop)."""
    _CACHE.clear()
    return get_pipeline_state()


def latest_available_hour(df_hist: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df_hist["ts_hour"].max())


def resolve_forecast_start(df_hist: pd.DataFrame, start: Optional[datetime]) -> datetime:
    if start is not None:
        return start
    return (latest_available_hour(df_hist) + pd.Timedelta(hours=1)).to_pydatetime()


def build_future_frame(df_hist, start_dt, n_hours):
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
    hist = df_hist[["PUZone", "ts_hour", "trip_count", "target_rph"]].copy()
    hist = hist.sort_values(["PUZone", "ts_hour"])

    grp = hist.groupby("PUZone", sort=False)
    hist["rolling_24h_trip_mean"] = grp["trip_count"].transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
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
    future = merge_lag(future, 1,   ["trip_count"], ["trip_count"])
    future = merge_lag(future, 1, ["target_rph", "rolling_24h_rph_mean", "rolling_7d_rph_mean", "rolling_24h_rph_std"], ["lag_1h_rph", "rolling_24h_rph_mean", "rolling_7d_rph_mean", "rolling_24h_rph_std"])
    future = merge_lag(future, 2,   ["target_rph"], ["lag_2h_rph"])
    future = merge_lag(future, 24,  ["target_rph"], ["lag_24h_rph"])
    future = merge_lag(future, 168, ["target_rph"], ["lag_168h_rph"])

    future["trip_count_delta_1h"] = future["lag_1h_trip_count"] - future["lag_2h_trip_count"]
    future["rph_delta_1h"]        = future["lag_1h_rph"] - future["lag_2h_rph"]
    future["rph_zscore"] = (future["lag_1h_rph"] - future["rolling_24h_rph_mean"]) / (future["rolling_24h_rph_std"].fillna(0) + 1e-6)
    future["rph_cv"] = future["rolling_24h_rph_std"].fillna(0) / (future["rolling_24h_rph_mean"].fillna(0) + 1e-6)

    return future


def attach_zone_stats(future, df_hist):
    hour_varying_cols = [
        "avg_revenue", "avg_tip", "tip_rate",
        "avg_distance_mi", "avg_duration_min", "avg_speed_mph",
        "congestion_score", "frac_airport_pickup", "frac_airport_dropoff",
        "rolling_airport_activity", "dropoff_count", "net_pickup_flow",
        "demand_vs_baseline", "relative_zone_activity",
        "median_rph", "p75_rph", "pickup_dropoff_ratio",
    ]
    hour_varying_cols = [c for c in hour_varying_cols if c in df_hist.columns]
    zone_hour_stats = df_hist.groupby(["PUZone", "hour"])[hour_varying_cols].mean().reset_index()
    future = future.merge(zone_hour_stats, on=["PUZone", "hour"], how="left")

    weather_cols = ["precipitation", "temperature", "wind_speed", "is_raining", "is_cold", "is_hot", "frac_raining", "precipitation_delta"]
    weather_cols = [c for c in weather_cols if c in df_hist.columns]
    if weather_cols:
        weather_avg = df_hist.groupby("hour")[weather_cols].mean().reset_index()
        future = future.merge(weather_avg, on="hour", how="left")

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


def _predict(model, X_arr):
    try:
        return model.predict(X_arr, validate_features=False)
    except TypeError:
        return model.predict(X_arr)


def run_inference(future, models, features, df_hist):
    future = fill_missing(future, df_hist, features)
    features_list = [str(f) for f in features]
    X_arr = future[features_list].to_numpy()

    future["pred_demand"] = _predict(models["demand"], X_arr)
    future["pred_rph"] = _predict(models["profit"], X_arr)
    future["pred_competition"] = _predict(models["competition"], X_arr)
    X_tip = models["tip_imputer"].transform(X_arr)
    future["pred_tip_prob"] = models["tip"].predict_proba(X_tip)[:, 1]

    return future


def score_and_rank(future, rank_by: str = "opportunity"):
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

    rank_col = FORECAST_TARGETS.get(rank_by, "opportunity_score")
    ascending = rank_col == "pred_competition"

    future["rank"] = (
        future.groupby("ts_hour")[rank_col]
        .rank(ascending=ascending, method="first").astype(int)
    )

    return future.sort_values(["ts_hour", "rank"])


OUTPUT_COLS = [
    "ts_hour", "PUZone", "PUBorough",
    "rank", "opportunity_score",
    "pred_demand", "pred_rph", "pred_competition", "pred_tip_prob",
    "score_demand", "score_profit", "score_competition", "score_tip",
]


def generate_forecast(
    hours: int = 24,
    start: Optional[datetime] = None,
    top: Optional[int] = None,
    rank_by: str = "opportunity",
) -> pd.DataFrame:
    if hours < 1:
        raise ValueError("hours must be >= 1")
    if rank_by not in FORECAST_TARGETS:
        raise ValueError(f"rank_by must be one of {list(FORECAST_TARGETS)}, got {rank_by!r}")

    state = get_pipeline_state()
    models, features, df_hist = state["models"], state["features"], state["df_hist"]

    start_dt = resolve_forecast_start(df_hist, start)

    future = build_future_frame(df_hist, start_dt, hours)
    future = attach_lag_features(future, df_hist)
    future = attach_zone_stats(future, df_hist)
    future = run_inference(future, models, features, df_hist)
    future = score_and_rank(future, rank_by=rank_by)

    if top:
        future = future[future["rank"] <= top]

    cols = [c for c in OUTPUT_COLS if c in future.columns]
    result = future[cols].round(4).reset_index(drop=True)
    float_cols = result.select_dtypes(include="float").columns
    result[float_cols] = result[float_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result
