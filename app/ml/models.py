from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import text
from app.core.schemas import timestamp_field


class Prediction(SQLModel, table=True):
    __tablename__ = "prediction"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    predicted_recharge_rate: float
    predicted_abstraction_volume: float
    confidence_score: float
    created_at: datetime = timestamp_field()
