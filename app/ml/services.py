from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_ORDER, compute_feature_row, LEVEL_MATCH_TOLERANCE
from app.ml.models import Prediction
from app.sensor.models import WaterLevelReading
from app.borehole.models import Borehole
from app.location.models import Location

logger = logging.getLogger(__name__)

# Module-level singletons, populated once at startup by load_model()
# The inference job and any route handler import these directly.

_model = None
_feature_columns: Optional[list[str]] = None

MODEL_PATH = Path("models/rf_level.joblib")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")


def load_model() -> None:
    """Load the trained model and and its feature-column order into module globals.
    Called once at startup. On any failure, leaves globals as None and logs
    """

    global _model, _feature_columns

    if not MODEL_PATH.exists():
        logger.warning(
            f"No model at {MODEL_PATH} - inference disabled until one is trained."
        )
        return

    if not FEATURE_COLUMNS_PATH.exists():
        logger.warning(
            f"Model present but {FEATURE_COLUMNS_PATH} - inference disabled "
            "(cannot verify feature order).",
        )

    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            columns = json.load(f)
    except Exception:
        logger.exception(
            "Failed to load model or feature columns - inference disabled."
        )
        return

    # checks to see if the model column fit the features column
    if columns != FEATURE_ORDER:
        logger.error(
            "Feature-column mismatch - model disabled. "
            f"On disk: {columns}. Expected: {FEATURE_ORDER}"
        )
        return
    
    _model = model
    _feature_columns = columns
    logger.info(
        f"Model loaded from {MODEL_PATH} ({len(columns)} features)."
    )


def get_model():
    """Return the loaded model or None if unavailable"""
    return _model

def get_features_columns() -> Optional[list[str]]:
    """Return the feature-column order the model was fit on, or None."""
    return _feature_columns


@dataclass
class InferenceResult:
    predicted_at: datetime
    predicted_level_2h: list[float]
    confidence: float


def _reindex_features(feature_row: dict[str, float], columns: list[str]) -> np.ndarray:
    """Turn the feature dict into a (1, n) array in the model's column order.
    Raises KeyError if a column the model expects isn't produced — a loud
    failure is correct here, never silently fill a missing feature."""

    values = [feature_row[col] for col in columns]
    return np.array([values], dtype=float)


def _confidence_from_trees(model, X: np.ndarray) -> float:
    """Turn the feature dict into a (1, n) array in the model's column order.
    Raises KeyError if a column the model expects isn't produced — a loud
    failure is correct here, never silently fill a missing feature."""

    per_tree = np.array([est.predict(X)[0] for est in model.estimators_])
    std = float(per_tree.std())
    return 1.0/ (1.0 + std)


def run_inference(
    levels: pd.DataFrame,
    flows: pd.DataFrame,
    weather: pd.DataFrame,
    now: pd.Timestamp,
) -> InferenceResult:
    """Compute a 24h prediction for `now`. Pure compute — no DB writes.
    Raises InsufficientData (propagated from compute_feature_row) or
    RuntimeError if no model is loaded."""

    model = get_model()
    columns = get_features_columns()
    if model is None or columns is None:
        raise RuntimeError("no model loaded")
    
    feature_row = compute_feature_row(levels, flows, weather, now)
    X = _reindex_features(feature_row, columns)
    
    predicted_2h = float(model.predict(X)[0])
    confidence = _confidence_from_trees(model, X)
    
    return InferenceResult(
        predicted_at=now.to_pydatetime(),
        predicted_level_2h=predicted_2h,
        confidence=confidence,
    )


async def get_prediction_chart(
    borehole_id: int,
    user_id: int, 
    session: AsyncSession,
    lookback: timedelta = timedelta(days=1)
) -> list[dict]:
    """Return predicted-vs-actual pairs for the chart. Each item:
      { t, predicted, actual }
    where `t` is the predicted-for time, `predicted` is the frozen model
    output, and `actual` is the real reading nearest that time (null when the
    target time is still in the future, or no reading landed within tolerance).
    Predictions whose target is still ahead show predicted with actual=null —
    that's the 'where it's heading' segment."""

    # Ownership check
    result = await session.exec(
        select(Borehole)
        .join(Location, Borehole.location_id == Location.id) # type: ignore
        .where(Borehole.id == borehole_id, Location.user_id == user_id)
    )

    if result.first() is None:
        raise ValueError("Borehole not found for this user")

    now = datetime.now(timezone.utc)
    since = now - lookback

    # Prediction whose target falls in the window, oldest first
    pred_rows = (
        await session.exec(
            select(Prediction.predicted_for, Prediction.predicted_level_2h, Prediction.confidence_score,)
            .where(Prediction.borehole_id == borehole_id)
            .where(Prediction.predicted_for >= since)
            .order_by(Prediction.predicted_for)
        )
    ).all()

    if not pred_rows:
        return []

    # Actual readings across the same window (+ a little margin so the nearest
    # match near the window edges has candidates on both sides).
    level_rows = (
        await session.exec(
            select(WaterLevelReading.captured_at, WaterLevelReading.water_level)
            .where(WaterLevelReading.borehole_id == borehole_id)
            .where(WaterLevelReading.captured_at >= since - timedelta(hours=1))
            .order_by(WaterLevelReading.captured_at)
        )
    ).all()

    preds = pd.DataFrame(
        pred_rows, columns=["predicted_for", "predicted", "confidence"]
    )

    preds["predicted_for"] = pd.to_datetime(preds["predicted_for"], utc = True)

    if level_rows:
        actuals = pd.DataFrame(level_rows, columns=["captured_at", "actual"])
        actuals["captured_at"] = pd.to_datetime(actuals["captured_at"], utc=True)
        # Nearest actual reading to each predicted_for, within the SAME
        # tolerance the model uses to define level_now — apples to apples.
        merged = pd.merge_asof(
            preds.sort_values("predicted_for"),
            actuals.sort_values("captured_at"),
            left_on="predicted_for",
            right_on="captured_at",
            direction="nearest",
            tolerance=LEVEL_MATCH_TOLERANCE,
        )
    else:
        merged = preds.copy()
        merged["actual"] = np.nan
        
    
    out = []
    for _, r in merged.iterrows():
        actual = r.get("actual")
        out.append(
            {
                "t": r["predicted_for"].to_pydatetime(),
                "predicted": float(r["predicted"]),
                "actual": None if pd.isna(actual) else float(actual),
                "confidence": float(r["confidence"]),
            }
        )
        
    return out