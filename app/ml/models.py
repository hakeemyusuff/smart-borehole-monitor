from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import text
from app.core.schemas import timestamp_field


class Prediction(SQLModel, table=True):
    __tablename__ = "prediction"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    predicted_recovery_at: Optional[datetime] = Field(default=None)
    predicted_level_24h: float
    confidence_score: float
    horizon_hours: int = Field(default=24)
    created_at: datetime = timestamp_field()
    
    
class AbstractionWindow(SQLModel, table=True):
    __tablename__ = "abstraction_window" # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: int = Field(foreign_key="prediction.id")
    start_time: datetime
    end_time: datetime
    safe_volume_litres: float