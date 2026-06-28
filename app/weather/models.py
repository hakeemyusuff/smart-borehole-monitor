from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone


class Weather(SQLModel, table=True):
    __tablename__ = "weather"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    location_id: Optional[int] = Field(default=None, foreign_key="location.id")
    temperature: float
    humidity: float
    precipitation: float
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
    )
