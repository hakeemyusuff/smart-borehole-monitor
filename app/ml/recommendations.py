"""Pure, advisory scheduling rules. No database writes or pump commands."""
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.schemas import PredictionStatus
from app.ml.level_forecast import next_forecast_review


class RecommendationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sensor_id: int = Field(gt=0)
    minimum_level_m: float = Field(gt=0)
    consideration_level_m: float = Field(gt=0)
    basis: str = Field(min_length=10, max_length=1000)

    @model_validator(mode="after")
    def check_levels(self):
        if self.consideration_level_m <= self.minimum_level_m:
            raise ValueError("Consideration level must exceed minimum operating level")
        if len(self.basis.strip()) < 10:
            raise ValueError("Record the engineering basis for these thresholds")
        return self


class PumpRecommendation(BaseModel):
    status: Literal["unavailable", "defer", "reassess", "consider"]
    reason_code: str
    reason: str
    assessed_at: datetime
    valid_until: datetime
    next_review_at: datetime
    suggested_start_at: datetime | None = None
    current_level_m: float | None = None
    current_level_captured_at: datetime | None = None
    forecast_level_m: float | None = None
    forecast_target_at: datetime | None = None
    model_version: str | None = None
    policy: RecommendationPolicy | None = None
    rules_version: str = "advisory-level-thresholds-v1"
    advisory_only: bool = True
    limitation: str = (
        "Operator advice only. The forecast does not include future pumping drawdown. "
        "This is not a pump command, guaranteed operating duration, or safe abstraction volume."
    )


def assess_recommendation(
    snapshot: PredictionStatus,
    policy: RecommendationPolicy | None,
    now: datetime,
) -> PumpRecommendation:
    """Assess fresh data only; never extrapolate a recovery time from one forecast."""
    next_review = next_forecast_review(now)
    result = PumpRecommendation(
        status="unavailable", reason_code="configuration_required",
        reason="Operating thresholds and their basis must be configured for this well and sensor.",
        assessed_at=now, valid_until=now + timedelta(minutes=2),
        next_review_at=next_review, policy=policy,
        current_level_m=snapshot.current_level,
        current_level_captured_at=snapshot.current_level_captured_at,
        model_version=snapshot.model_version,
    )
    if policy is None:
        return result
    captured = snapshot.current_level_captured_at
    level = snapshot.current_level
    if level is not None and not 0 <= level < float("inf"):
        result.current_level_m = None
    if (captured is None or captured.tzinfo is None or level is None
            or not 0 <= level < float("inf")
            or not timedelta(0) <= now - captured <= timedelta(minutes=35)):
        result.reason_code = "measurement_unavailable"
        result.reason = "A recent valid measurement is required before assessing pumping."
        return result
    result.valid_until = min(result.valid_until, captured + timedelta(minutes=35))
    # A measured low level must not be overridden by a favourable/missing forecast.
    if level <= policy.minimum_level_m:
        result.status = "defer"
        result.reason_code = "measured_below_minimum"
        result.reason = "Measured level is at or below the configured operating minimum. Defer starting; if pumping, arrange an immediate operator check."
        result.next_review_at = now
        return result
    forecast = snapshot.predicted_level_2h
    target = snapshot.predicted_for
    if (snapshot.status != "fresh" or forecast is None
            or not 0 <= forecast < float("inf")
            or snapshot.checked_at.tzinfo is None
            or not timedelta(0) <= now - snapshot.checked_at < timedelta(minutes=2)
            or target is None or target.tzinfo is None or target <= now):
        result.reason_code = "forecast_unavailable"
        result.reason = "Wait for a fresh forecast before assessing a pumping start."
        return result
    result.forecast_level_m = forecast
    result.forecast_target_at = target
    result.valid_until = min(result.valid_until, target, snapshot.checked_at + timedelta(minutes=2))
    if forecast <= policy.minimum_level_m:
        result.status = "defer"
        result.reason_code = "forecast_below_minimum"
        result.reason = "The forecast reaches or falls below the configured minimum. Defer starting and reassess after the next forecast update."
    elif min(level, forecast) < policy.consideration_level_m:
        result.status = "reassess"
        result.reason_code = "below_consideration_level"
        result.reason = "Measured or forecast level is below the configured consideration threshold. Wait and reassess; no recovery time has been established."
    else:
        result.status = "consider"
        result.reason_code = "thresholds_met"
        result.reason = "Measured and forecast levels meet the configured consideration threshold. An operator may consider starting now, subject to continued monitoring and site operating requirements."
        result.suggested_start_at = now
    return result
