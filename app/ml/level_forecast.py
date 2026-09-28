"""Pure level-only forecasting contract. No settings, database, or network imports."""
import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['delta_1h', 'delta_3h', 'delta_6h']
TOLERANCE = pd.Timedelta(minutes=35)
HORIZON_HOURS = 2


class InsufficientData(ValueError):
    pass


def compute_level_features(levels, now):
    """Last capture at/before each anchor, within 35min, uploaded by issue time."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Forecast issue time must be timezone-aware')
    if levels.empty:
        raise InsufficientData('No water-level readings available')
    data = levels[['captured_at', 'created_at', 'water_level']].copy()
    for col in ['captured_at', 'created_at']:
        data[col] = pd.to_datetime(data[col], utc=True, format='mixed', errors='raise')
    if data.isna().any().any() or data.captured_at.duplicated().any():
        raise InsufficientData('Missing or duplicate measurement timestamps')
    if (data.created_at < data.captured_at).any():
        raise InsufficientData('A reading arrived before its capture time')
    if not np.isfinite(data.water_level.to_numpy(dtype=float)).all() or (data.water_level < 0).any():
        raise InsufficientData('Invalid water-level measurement')
    data = data.sort_values('captured_at')
    readings = {}
    for lag in [0, 1, 3, 6]:
        anchor = now-pd.Timedelta(hours=lag)
        candidates = data.loc[(data.captured_at <= anchor) & (data.captured_at >= anchor-TOLERANCE)
                              & (data.created_at <= now)]
        if candidates.empty:
            raise InsufficientData(f'Missing available water-level reading at {lag}-hour anchor')
        readings[lag] = candidates.iloc[-1]
    features = {'level_now': float(readings[0].water_level)}
    features.update({f'delta_{lag}h':features['level_now']-float(readings[lag].water_level) for lag in [1,3,6]})
    return features, readings[0].captured_at.to_pydatetime()


def artifact_version(payload):
    content = {key: value for key, value in payload.items() if key != "model_version"}

    encoded = json.dumps(
        content,
        sort_keys=True,
        allow_nan=False,
    ).encode()

    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]

    return "level-change-linear-" + fingerprint


@dataclass(frozen=True)
class LevelModel:
    metadata: dict

    @classmethod
    def load(cls, path: Path):
        metadata = json.loads(path.read_text())

        # Make sure this is the kind of model our code understands.
        expected = {
            "schema_version": 2,
            "model_type": "linear_regression",
            "prediction_type": "level_change",
            "features": FEATURES,
            "horizon_hours": HORIZON_HOURS,
            "tolerance_minutes": 35,
            "feature_contract": ("captured_before_anchor_arrived_by_issue_v1"),
        }

        for field, expected_value in expected.items():
            if metadata.get(field) != expected_value:
                raise ValueError(f"Unexpected model setting: {field}")

        if metadata.get("model_version") != artifact_version(metadata):
            raise ValueError("Model content/version mismatch")

        coefficients = np.asarray(
            metadata["coefficients"],
            dtype=float,
        )

        intercept = float(metadata["intercept"])

        if coefficients.shape != (len(FEATURES),):
            raise ValueError("The model must have three coefficients")

        if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
            raise ValueError("Invalid regression parameters")

        for field in ["borehole_id", "sensor_id"]:
            value = metadata.get(field)

            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid model setting: {field}")

        return cls(metadata)

    def predict(self, features):
        # Only the three changes go into the learned equation.
        changes = np.array(
            [features[name] for name in FEATURES],
            dtype=float,
        )

        current_level = float(features["level_now"])

        if (
            not np.isfinite(changes).all()
            or not np.isfinite(current_level)
            or current_level < 0
        ):
            raise InsufficientData("Invalid prediction inputs")

        predicted_change = float(
            self.metadata["intercept"] + np.dot(self.metadata["coefficients"], changes)
        )

        # Convert the predicted change into a predicted water level.
        predicted_level = current_level + predicted_change

        if not np.isfinite(predicted_level) or predicted_level < 0:
            raise InsufficientData("Model produced an invalid water-level forecast")

        return predicted_level


def expected_issue_time(now):
    """Hourly job runs at :05 UTC; allow it five minutes before marking overdue."""
    return (pd.Timestamp(now)-pd.Timedelta(minutes=10)).floor('h').to_pydatetime()


def forecast_state(now, issued_at, predicted_for, latest_capture):
    if issued_at is None:
        return 'unavailable'
    if latest_capture is None or now-latest_capture > timedelta(minutes=35):
        return 'stale'
    if issued_at < expected_issue_time(now) or issued_at > now or predicted_for <= now:
        return 'stale'
    return 'fresh'
