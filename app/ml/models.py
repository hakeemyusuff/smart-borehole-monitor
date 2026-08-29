from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import UniqueConstraint
from app.core.schemas import optional_timestamp_field, timestamp_field


class Prediction(SQLModel, table=True):
    __tablename__ = "prediction"  # type: ignore
    __table_args__ = (
        UniqueConstraint(
            "borehole_id",
            "predicted_for",
            name="uq_prediction_borehole_predicted_for",
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    predicted_recovery_at: Optional[datetime] = optional_timestamp_field()
    predicted_level_2h: float
    confidence_score: float
    horizon_hours: int = Field(default=2)
    predicted_for: datetime = optional_timestamp_field()
    created_at: datetime = timestamp_field()


class AbstractionWindow(SQLModel, table=True):
    __tablename__ = "abstraction_window"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: int = Field(foreign_key="prediction.id")
    start_time: datetime
    end_time: datetime
    safe_volume_litres: float
