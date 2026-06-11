# NYC Taxi Ride Demand & Opportunity MLOps

An end-to-end MLOps pipeline that ingests NYC yellow taxi trip + weather data,
engineers an hourly zone-level feature store, trains a 4-layer model stack,
generates ranked "where should a driver go next" forecasts, and serves them
through a FastAPI service and a Streamlit map dashboard.

The pipeline runs on Airflow and can be deployed in two ways:

- **Local** — Docker Compose stack on your machine (`docker-compose.yml`)
- **EC2** — A scheduled, self-stopping EC2 box driven by Lambda + EventBridge
  (`docker-compose.ec2.yml`)

---

## Architecture

```
                         ┌───────────────────────────┐
                         │   Ingestion DAG (Airflow) │
                         └───────────────────────────┘
                                      │
            get_month → prune_old_data → [download_data, fetch_weather]
                                      │
                                validate_data
                                      │
                         merge_everything (preprocess/merge.py)
                                      │
                              features (preprocess/features.py)
                                      │
                                      ▼
                         S3: final/hourly_zone_timeseries.parquet
                                      │
        ┌─────────────────────────────────────────────────────────┐
        │ local: triggered manually via taxi_model_training       │
        │ ec2:   auto-triggered via TriggerDagRunOperator         │
        └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────┐
                         │   Training DAG (Airflow)  │
                         └───────────────────────────┘
                          train_model → forecast (→ stop_instance on EC2)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
           MLflow (params, metrics,            S3: models/*.pkl
           artifacts, model registry)          S3: forecasts/*.csv
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
              FastAPI Forecast API             Streamlit Map App
              (api/serve.py, port 7860)        (app/streamlit_app.py)
```

---

## Model stack

Four models trained per zone-hour and combined into a single
`opportunity_score` (`preprocess/features.py` builds the features,
`models/*/train.py` trains, `models/*/forecast.py` / `api/engine.py` predict):

| Layer | Model | Target |
|---|---|---|
| Demand | LGBMRegressor | `trip_count` |
| Profitability | XGBRegressor | `target_rph` (revenue/hour) |
| Competition | LGBMRegressor | composite competition score |
| Tip | LogisticRegression | `high_tip_zone` (binary) |

Hyperparameters for all four layers are tuned with Optuna (TPE) before
the final MLflow-tracked training run (`OPTUNA_N_TRIALS`, default 20).

---

## Repository layout

```
dags/
  local/
    taxi_data_pipeline.py     # taxi_data_ingestion_local DAG
    taxi_model_pipeline.py    # taxi_model_training DAG (manual)
  ec2/
    ingestion.py              # taxi_data_ingestion_ec2 DAG (auto-triggers training)
    training.py               # taxi_model_training_ec2_self_stop DAG

ingestion/
  download.py                 # download_month_to_s3 / prune_s3_months
  weather.py                  # sync_weather_month_to_s3 (Open-Meteo)

preprocess/
  merge.py                     # merges raw trips + weather + zones -> S3 enriched parquet
  features.py                  # DuckDB feature engineering -> hourly zone timeseries

models/
  local/
    train.py                   # Optuna-tuned training (logs to MLflow)
    forecast.py                # generates forecast CSV, writes to S3
  ec2/
    train.py                   # Optuna-tuned training + email alert (logs to MLflow)
    forecast.py                # forecast generation (EC2 variant)

api/
  serve.py                      # FastAPI app (/health, /meta, /forecast)
  engine.py                      # cached model + feature-store loading, inference, ranking

app/
  streamlit_app.py              # Streamlit zone-opportunity map (reads forecasts from S3)
  dockerfile                     # Streamlit container (Hugging Face Spaces)

lambda/
  start_ec2.py                   # EventBridge -> start EC2 (monthly)
  stop_ec2.py                    # EventBridge -> safety-net stop EC2

docker/
  Dockerfile.airflow             # Airflow image (local + EC2)
  Dockerfile.airflow-ec2          # EC2 variant entrypoint
  Dockerfile.api                  # FastAPI image

tests/
  test_data_validation_local.py
  test_data_validation_ec2.py

docker-compose.yml                # local dev stack
docker-compose.ec2.yml             # EC2 stack (IAM role creds, localhost-only ports)
```

---

## Pipelines

### Ingestion DAG (`taxi_data_ingestion_local` / `taxi_data_ingestion_ec2`)

Runs `@monthly`. Resolves the target month (from trigger config, DAG param, or
the previous calendar month), then:

1. `prune_old_data` — drops S3 data older than 12 months
2. `download_data` / `fetch_weather` — fetch trip parquet + Open-Meteo weather, in parallel
3. `validate_data` — runs `tests/test_data_validation_*.py` against the new data
4. `merge_everything` — `preprocess/merge.py`
5. `features` — `preprocess/features.py`
6. *(EC2 only)* `trigger_training` — kicks off the training DAG

### Training DAG (`taxi_model_training` / `taxi_model_training_ec2_self_stop`)

Manual trigger (or auto-triggered on EC2):

1. `train_model` — trains the 4-layer model stack, logs to MLflow, registers models
2. `forecast` — loads production models, generates ranked forecasts, writes to S3
3. *(EC2 only)* `stop_instance` — stops the EC2 host via its IAM role

---

## Forecast API

```
uvicorn api.serve:app --reload
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness check |
| `GET /meta` | latest data hour + valid `rank_by` options |
| `GET /forecast?hours=24&top=10&rank_by=opportunity` | ranked zone-hour forecast |

`rank_by` options: `demand`, `profit`, `competition`, `tip`, `opportunity`.

Models and the feature store are loaded from S3 once per process and cached
in memory (`api/engine.py`).

---

## Streamlit app

Demo: https://lcodecorn-nyc-taxi.hf.space

`app/streamlit_app.py` reads the latest `forecasts/*.csv` from S3, joins it
with the NYC taxi zone GeoJSON, and renders a choropleth map with per-zone
demand/profitability/competition/opportunity scores, plus a sortable zone
ranking table. Deployed as a Docker-based Hugging Face Space (`app/dockerfile`,
port 7860).

---

## Setup

- **Local development** (Docker Compose, Airflow + MLflow + API on your
  machine): see [LOCAL_SETUP.md](LOCAL_SETUP.md)
- **EC2 deployment** (scheduled, self-stopping EC2 box with Lambda +
  EventBridge automation): see [EC2_SETUP.md](EC2_SETUP.md)

Both setups read configuration from a `.env` file — copy `.env.example` to
get started.

## Services & ports (local)

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Forecast API | http://localhost:7860 |
