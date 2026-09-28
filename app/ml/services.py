from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

import pandas as pd
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.ml.level_forecast import (
    LevelModel,
    FEATURES,
    TOLERANCE,
    compute_level_features,
    forecast_state,
)
from app.ml.models import Prediction
from app.sensor.models import WaterLevelReading
from app.borehole.models import Borehole
from app.location.models import Location

logger = logging.getLogger(__name__)
_model = None


def load_model():
    """Fail closed, including on reload; never fall back to the old forest artifact."""
    global _model
    _model = None
    try:
        path = Path(settings.level_model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        _model = LevelModel.load(path)
        logger.info("Loaded %s", _model.metadata["model_version"])
    except Exception:
        logger.exception("Level model unavailable; inference disabled")


def get_model():
    return _model


def get_features_columns():
    return FEATURES.copy() if _model else None


@dataclass
class InferenceResult:
    predicted_at: datetime
    predicted_level_2h: float
    model_version: str
    input_level_captured_at: datetime
    confidence: None = None


def run_inference(levels, flows=None, weather=None, now=None):
    """Calculate a two-hour water-level forecast without writing to the database."""

    if _model is None:
        raise RuntimeError("No level model loaded")

    if now is None:
        raise ValueError("A forecast cutoff time is required")

    forecast_at = pd.Timestamp(now)

    features, captured_at = compute_level_features(
        levels,
        forecast_at,
    )

    predicted_level = _model.predict(features)

    return InferenceResult(
        predicted_at=forecast_at.to_pydatetime(),
        predicted_level_2h=predicted_level,
        model_version=_model.metadata["model_version"],
        input_level_captured_at=captured_at,
    )


async def require_owner(borehole_id, user_id, session):
    owned = (
        await session.exec(
            select(Borehole)
            .join(Location, Borehole.location_id == Location.id)
            .where(Borehole.id == borehole_id, Location.user_id == user_id)
        )
    ).first()
    if owned is None:
        raise ValueError("Borehole not found for this user")


async def get_prediction_status(borehole_id, user_id, session):
    await require_owner(borehole_id, user_id, session)
    now = datetime.now(timezone.utc)
    response = dict(
        status="unavailable",
        message="Forecast model is unavailable.",
        checked_at=now,
        predicted_level_2h=None,
        issued_at=None,
        predicted_for=None,
        model_version=None,
        current_level=None,
        current_level_captured_at=None,
    )
    if _model is None:
        return response
    meta = _model.metadata
    if borehole_id != meta["borehole_id"]:
        response["message"] = "No forecasting model is configured for this well."
        return response
    latest = (
        await session.exec(
            select(WaterLevelReading)
            .where(
                WaterLevelReading.borehole_id == borehole_id,
                WaterLevelReading.sensor_id == meta["sensor_id"],
                WaterLevelReading.captured_at <= now,
                WaterLevelReading.created_at <= now,
            )
            .order_by(WaterLevelReading.captured_at.desc())
            .limit(1)
        )
    ).first()
    pred = (
        await session.exec(
            select(Prediction)
            .where(
                Prediction.borehole_id == borehole_id,
                Prediction.model_version == meta["model_version"],
                Prediction.created_at <= now,
            )
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
    ).first()
    captured = latest.captured_at if latest else None
    if latest:
        response.update(
            current_level=latest.water_level, current_level_captured_at=captured
        )
    state = forecast_state(
        now,
        pred.created_at if pred else None,
        pred.predicted_for if pred else None,
        captured,
    )
    response.update(
        status=state,
        model_version=meta["model_version"],
        message={
            "fresh": "Two-hour water-level forecast.",
            "stale": "Forecast is out of date. Waiting for a new forecast and recent readings.",
            "unavailable": "No forecast is available yet. Recent readings at the current, 1-, 3- and 6-hour anchors are required.",
        }[state],
    )
    if pred:
        response.update(
            issued_at=getattr(pred, "generated_at", None) or pred.created_at,
            predicted_for=pred.predicted_for,
        )
        # Historical values remain on the chart, never masquerading as a fresh card.
        if state == "fresh":
            response["predicted_level_2h"] = pred.predicted_level_2h
    return response


async def get_prediction_chart(
    borehole_id: int, user_id: int, session: AsyncSession, lookback=timedelta(days=1)
):
    await require_owner(borehole_id, user_id, session)
    if _model is None or borehole_id != _model.metadata["borehole_id"]:
        return []
    now = datetime.now(timezone.utc)
    since = now - lookback
    rows = (
        await session.exec(
            select(Prediction)
            .where(
                Prediction.borehole_id == borehole_id,
                Prediction.model_version == _model.metadata["model_version"],
                Prediction.predicted_for >= since,
                Prediction.created_at <= now,
                Prediction.predicted_for <= now + timedelta(hours=2),
            )
            .order_by(Prediction.predicted_for)
        )
    ).all()
    if not rows:
        return []
    levels = (
        await session.exec(
            select(WaterLevelReading.captured_at, WaterLevelReading.water_level)
            .where(
                WaterLevelReading.borehole_id == borehole_id,
                WaterLevelReading.sensor_id == _model.metadata["sensor_id"],
                WaterLevelReading.captured_at >= since - timedelta(minutes=35),
                WaterLevelReading.captured_at <= now,
                WaterLevelReading.created_at <= now,
            )
            .order_by(WaterLevelReading.captured_at)
        )
    ).all()
    actuals = pd.DataFrame(levels, columns=["captured_at", "value"])
    if not actuals.empty:
        actuals["captured_at"] = pd.to_datetime(actuals.captured_at, utc=True)
    by_time = {pd.Timestamp(r.predicted_for): r for r in rows}
    out = []
    for target in pd.date_range(min(by_time), max(by_time), freq="h"):
        row = by_time.get(target)
        actual = None
        if row and target <= now and not actuals.empty:
            distances = (actuals.captured_at - target).abs()
            i = distances.idxmin()
            if distances.loc[i] <= TOLERANCE:
                actual = float(actuals.loc[i, "value"])
        out.append(
            dict(
                t=target.to_pydatetime(),
                predicted=row.predicted_level_2h if row else None,
                actual=actual,
                confidence=None,
                issued_at=(
                    (getattr(row, "generated_at", None) or row.created_at)
                    if row
                    else None
                ),
                model_version=row.model_version if row else None,
            )
        )
    return out


async def get_pump_recommendation(borehole_id, user_id, session):
    from app.ml.recommendations import assess_recommendation
    from app.ml.schemas import PredictionStatus

    # Reuse ownership, model scoping and freshness checks from forecast status.
    snapshot = PredictionStatus(**await get_prediction_status(borehole_id, user_id, session))
    policy = settings.pump_recommendation_policies.get(borehole_id)
    model = get_model()
    if policy and (model is None or model.metadata["borehole_id"] != borehole_id
                   or model.metadata["sensor_id"] != policy.sensor_id):
        policy = None
    return assess_recommendation(snapshot, policy, datetime.now(timezone.utc))
