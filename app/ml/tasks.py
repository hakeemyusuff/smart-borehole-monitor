"""Read recent water levels and save the hourly two-hour forecast."""

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import select

from app.core.database import async_session_maker
from app.ml.level_forecast import InsufficientData
from app.ml.services import get_model, run_inference
from app.ml.models import Prediction
from app.sensor.models import WaterLevelReading

logger = logging.getLogger(__name__)

HORIZON_HOURS = 2


async def run_inference_job():
    model = get_model()

    if model is None:
        logger.info("Inference skipped: no model loaded")
        return

    generated_at = datetime.now(timezone.utc)

    # Use the start of this hour as the measurement cutoff.
    forecast_at = generated_at.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    predicted_for = forecast_at + timedelta(hours=HORIZON_HOURS)

    borehole_id = model.metadata["borehole_id"]
    sensor_id = model.metadata["sensor_id"]

    async with async_session_maker() as session:
        query = (
            select(
                WaterLevelReading.captured_at,
                WaterLevelReading.created_at,
                WaterLevelReading.water_level,
            )
            .where(
                WaterLevelReading.borehole_id == borehole_id,
                WaterLevelReading.sensor_id == sensor_id,
                WaterLevelReading.captured_at
                >= forecast_at - timedelta(hours=6, minutes=35),
                WaterLevelReading.captured_at <= forecast_at,
                WaterLevelReading.created_at <= forecast_at,
            )
            .order_by(WaterLevelReading.captured_at)
        )

        query_result = await session.exec(query)

        levels = pd.DataFrame(
            query_result.all(),
            columns=["captured_at", "created_at", "water_level"],
        )

        try:
            prediction = run_inference(
                levels,
                now=pd.Timestamp(forecast_at),
            )
        except InsufficientData as error:
            logger.info("Inference skipped: %s", error)
            return

        statement = (
            pg_insert(Prediction)
            .values(
                borehole_id=borehole_id,
                predicted_level_2h=prediction.predicted_level_2h,
                predicted_for=predicted_for,
                created_at=forecast_at,
                generated_at=generated_at,
                horizon_hours=HORIZON_HOURS,
                model_version=prediction.model_version,
                input_level_captured_at=(prediction.input_level_captured_at),
                confidence_score=None,
            )
            .on_conflict_do_nothing(constraint="uq_prediction_borehole_predicted_for")
        )

        await session.exec(statement)
        await session.commit()

        logger.info(
            "Forecast processed for %s using %s",
            predicted_for,
            prediction.model_version,
        )
