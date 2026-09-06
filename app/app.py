import io
import json
import os

import boto3
import pandas as pd
import streamlit as st
import plotly.express as px
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


AWS_ACCESS_KEY_ID = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_REGION = get_secret("AWS_REGION")
S3_BUCKET = get_secret("S3_BUCKET")

FORECAST_PREFIX = "forecasts/"
GEOJSON_KEY = "NYC_Taxi_Zones_20260602.geojson"

API_BASE_URL = "https://lcodecorn-taxi-api.hf.space"


# S3
def get_s3_client():
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
        return None

    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


# LOAD FORECAST DATA
@st.cache_data(ttl=300)
def load_forecast_dataframe():
    try:
        s3 = get_s3_client()
        if s3 is None:
            st.error("S3 credentials not configured.")
            return None

        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=FORECAST_PREFIX,
            MaxKeys=100,
        )

        contents = response.get("Contents", [])
        forecast_files = [obj for obj in contents if obj["Key"].endswith(".csv")]

        if not forecast_files:
            st.error("No forecast CSV files found in S3.")
            return None

        latest_file = max(forecast_files, key=lambda x: x["LastModified"])

        buffer = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, latest_file["Key"], buffer)
        buffer.seek(0)

        df = pd.read_csv(buffer)
        df["ts_hour"] = pd.to_datetime(df["ts_hour"])

        return df

    except Exception as e:
        st.exception(e)
        return None


# LOAD GEOJSON
@st.cache_data
def load_boundary_geojson():
    try:
        s3 = get_s3_client()
        if s3 is None:
            st.error("S3 credentials not configured.")
            return None

        buffer = io.BytesIO()
        s3.download_fileobj(
            Bucket=S3_BUCKET,
            Key=GEOJSON_KEY,
            Fileobj=buffer,
        )

        buffer.seek(0)
        return json.loads(buffer.read().decode("utf-8"))

    except Exception as e:
        st.warning(f"Failed to load GeoJSON: {e}")
        return None


# STREAMLIT APP
st.set_page_config(page_title="NYC Taxi Zone Forecast Map", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NYC Taxi Zone Forecast Map")

df = load_forecast_dataframe()
if df is None:
    st.stop()

geojson = load_boundary_geojson()
if geojson is None:
    st.stop()

# SIDEBAR CONTROLS
st.sidebar.header("Controls")

all_hours = sorted(df["ts_hour"].unique())

selected_ts = st.sidebar.selectbox(
    "Forecast hour",
    all_hours,
    index=len(all_hours) - 1,
)

metric = st.sidebar.selectbox(
    "Colour by",
    ["pred_rph", "pred_demand", "pred_competition", "opportunity_score"],
)

with st.sidebar.expander("API access"):
    st.markdown(
        f"""
This map shows a snapshot of the latest pre-computed forecast.
For live, on-demand forecasts (any time window or ranking signal),
use the **Forecast API**:

- Interactive docs: [{API_BASE_URL}/docs]({API_BASE_URL}/docs)
- `GET /forecast?hours=24&top=10&rank_by=opportunity`
- `GET /meta` — latest data hour & ranking options

`rank_by` options: `demand`, `profit`, `competition`, `tip`, `opportunity`.

Example:
```
{API_BASE_URL}/forecast?hours=12&top=5&rank_by=profit
```
"""
    )

# FILTER DATA
hour_df = df[df["ts_hour"] == selected_ts]

zone_df = (
    hour_df.dropna(subset=["PUZone"])
    .groupby("PUZone", as_index=False)
    .agg(
        pred_rph=("pred_rph", "mean"),
        pred_demand=("pred_demand", "mean"),
        pred_competition=("pred_competition", "mean"),
        opportunity_score=("opportunity_score", "mean"),
    )
)

zone_df["PUZone"] = zone_df["PUZone"].astype(str).str.strip()

# METRICS
c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg $/hr", f"${zone_df['pred_rph'].mean():.2f}")
c2.metric("Avg Demand", f"{zone_df['pred_demand'].mean():.1f}")
c3.metric("Avg Competition", f"{zone_df['pred_competition'].mean():.2f}")
c4.metric("Avg Opportunity", f"{zone_df['opportunity_score'].mean():.2f}")

# MAP
label_map = {
    "pred_rph": "Avg $/hr",
    "pred_demand": "Avg Demand",
    "pred_competition": "Avg Competition",
    "opportunity_score": "Avg Opportunity",
}

fig = px.choropleth_mapbox(
    zone_df,
    geojson=geojson,
    locations="PUZone",
    featureidkey="properties.zone",
    color=metric,
    color_continuous_scale="amp",
    labels={metric: label_map[metric]},
    hover_name="PUZone",
    center={"lat": 40.713195, "lon": -73.924156},
    zoom=10,
    mapbox_style="carto-positron",
    opacity=0.65,
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700)

st.plotly_chart(fig, use_container_width=True)

# TABLE
st.subheader("Zone Rankings")

display_df = (
    zone_df.sort_values("opportunity_score", ascending=False)
    .reset_index(drop=True)
)

display_df.index += 1

st.dataframe(display_df, use_container_width=True, height=500)

st.caption(f"Forecast hour: {selected_ts}")