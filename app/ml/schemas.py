from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class PredictionChartPoint(BaseModel):
    t: datetime
    predicted: float | None = None
    actual: float | None = None
    confidence: float | None = None
    issued_at: datetime | None = None
    model_version: str | None = None


class PredictionStatus(BaseModel):
    status: Literal['fresh','stale','unavailable']
    message: str
    checked_at: datetime
    predicted_level_2h: float | None = None
    issued_at: datetime | None = None
    predicted_for: datetime | None = None
    model_version: str | None = None
    current_level: float | None = None
    current_level_captured_at: datetime | None = None
