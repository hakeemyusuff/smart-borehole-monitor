from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import select

from app.core.database import async_session_maker
from app.ml.features import InsufficientData
from app.ml.services import run_inference, get_model
from app.ml.models import Prediction
from app.sensor.models import WaterLevelReading, FlowReading
from app.weather.models import Weather

logger = logging.getLogger(__name__)
job_logger = logging.getLogger("uvicorn.error")


INFERENCE_BOREHOLE_ID = 5
INFERENCE_LOCATION_ID = 5
HORIZON_HOURS = 2

# Bounded read windows
LEVEL_WINDOW = timedelta(hours=12)
FLOW_WINDOW = timedelta(hours=30)


async def run_inference_job() -> None:
    """Hourly: predict the level HORIZON_HOURS ahead for the configured
    borehole and store it, frozen, before the actual reading arrives.
    Logs and skips when there isn't enough recent history."""

    try:
        if get_model() is None:
            logger.info("[Inference] no model loaded - skipping")
            return

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        predicted_for = now + timedelta(hours=HORIZON_HOURS)

        async with async_session_maker() as session:
            level_rows = (
                await session.exec(
                    select(WaterLevelReading.captured_at, WaterLevelReading.water_level)
                    .where(WaterLevelReading.borehole_id == INFERENCE_BOREHOLE_ID)
                    .where(WaterLevelReading.captured_at >= now - LEVEL_WINDOW)
                )
            ).all()

            flow_rows = (
                await session.exec(
                    select(FlowReading.captured_at, FlowReading.abstraction_rate)
                    .where(FlowReading.borehole_id == INFERENCE_BOREHOLE_ID)
                    .where(FlowReading.captured_at >= now - FLOW_WINDOW)
                )
            ).all()

            weather_rows = (
                await session.exec(
                    select(Weather.created_at, Weather.precipitation).where(
                        Weather.location_id == INFERENCE_LOCATION_ID
                    )
                )
            ).all()

            levels = pd.DataFrame(level_rows, columns=["created_at", "water_level"])
            flows = pd.DataFrame(flow_rows, columns=["created_at", "abstraction_rate"])
            weather = pd.DataFrame(
                weather_rows, columns=["created_at", "precipitation"]
            )

            for df in (levels, flows, weather):
                if not df.empty:
                    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

            try:
                result = run_inference(levels, flows, weather, pd.Timestamp(now))
            except InsufficientData as e:
                logger.info(f"[Inference] skipped at {now} - insufficient history: {e}")
                return

            async with async_session_maker() as session:
                stmt = (
                    pg_insert(Prediction)
                    .values(
                        borehole_id=INFERENCE_BOREHOLE_ID,
                        predicted_level_2h=result.predicted_level_2h,
                        predicted_for=predicted_for,
                        confidence_score=result.confidence,
                        horizon_hours=HORIZON_HOURS,
                        created_at=now,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_prediction_borehole_predicted_for"
                    )
                )

                await session.exec(stmt)
                await session.commit()

            logger.info(
                "[inference] stored prediction for %s: %.3f m (confidence %.3f)",
                predicted_for,
                result.predicted_level_2h,
                result.confidence,
            )

    except Exception:
        job_logger.exception("[inference] job failed")
        raise
