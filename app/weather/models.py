from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import text
from app.core.schemas import timestamp_field


class Weather(SQLModel, table=True):
    __tablename__ = "weather"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    location_id: Optional[int] = Field(default=None, foreign_key="location.id")
    temperature: Optional[float] = Field(default=None)
    humidity: Optional[float] = Field(default=None)
    precipitation: Optional[float] = Field(default=None)
    created_at: datetime = timestamp_field()
