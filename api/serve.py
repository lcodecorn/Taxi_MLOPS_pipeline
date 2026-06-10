"""""
NYC TAXI INTELLIGENCE ENGINE — FORECAST API
Run:
  uvicorn api.serve:app --reload
Examples:
  GET /forecast?hours=24
  GET /forecast?hours=12&top=10&rank_by=profit
  GET /forecast?start=2024-06-01T08:00:00&hours=6&rank_by=demand
"""""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from api.engine import (
    FORECAST_TARGETS,
    generate_forecast,
    get_pipeline_state,
    latest_available_hour,
)

app = FastAPI(
    title="NYC Taxi Forecast API",
    description="Pick a time frame and a signal to forecast zone-level taxi opportunities.",
    version="1.0.0",
)


class ForecastRow(BaseModel):
    ts_hour: datetime
    PUZone: str
    PUBorough: str
    rank: int
    opportunity_score: float
    pred_demand: float
    pred_rph: float
    pred_competition: float
    pred_tip_prob: float
    score_demand: float
    score_profit: float
    score_competition: float
    score_tip: float


class ForecastResponse(BaseModel):
    forecast_start: datetime
    forecast_end: datetime
    hours: int
    rank_by: str
    row_count: int
    rows: list[ForecastRow]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Latest hour available in the feature store and the valid rank_by options."""
    state = get_pipeline_state()
    return {
        "data_max_ts_hour": str(latest_available_hour(state["df_hist"])),
        "rank_by_options": list(FORECAST_TARGETS),
    }


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    hours: int = Query(24, ge=1, le=168, description="Number of future hours to forecast"),
    start: Optional[datetime] = Query(
        None, description="Forecast start datetime (defaults to last known data hour + 1)"
    ),
    top: Optional[int] = Query(
        None, ge=1, description="Keep only the top-N zones per hour (omit for all zones)"
    ),
    rank_by: str = Query(
        "opportunity",
        description=f"Signal to forecast/rank zones by: one of {list(FORECAST_TARGETS)}",
    ),
):
    if rank_by not in FORECAST_TARGETS:
        raise HTTPException(
            status_code=422,
            detail=f"rank_by must be one of {list(FORECAST_TARGETS)}, got {rank_by!r}",
        )

    try:
        df_out = generate_forecast(hours=hours, start=start, top=top, rank_by=rank_by)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if df_out.empty:
        raise HTTPException(status_code=404, detail="No forecast rows produced for the given window")

    forecast_start = df_out["ts_hour"].min()
    forecast_end = df_out["ts_hour"].max()

    return ForecastResponse(
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        hours=hours,
        rank_by=rank_by,
        row_count=len(df_out),
        rows=df_out.to_dict(orient="records"),
    )
