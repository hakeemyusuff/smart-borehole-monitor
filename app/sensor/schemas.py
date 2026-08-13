from typing import Optional
from pydantic import BaseModel, field_validator, Field
from datetime import datetime, timezone, timedelta
from app.sensor.models import Sensor, SensorType, SensorStatus
from pydantic import ConfigDict

# from sqlmodel import SQLModel

MAX_BUFFER_AGE = timedelta(hours=6)
MAX_CLOCK_SKEW = timedelta(minutes=2)
MAX_BATCH_SIZE = 400


class SensorCreate(BaseModel):
    borehole_id: int
    type: SensorType


class SensorPublic(SensorCreate):
    id: int
    status: SensorStatus
    last_seen: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SensorCreateResponse(BaseModel):
    sensor: SensorPublic
    device_key: Optional[str] = None


class ReadingIn(BaseModel):
    reading: float
    captured_at: datetime

    @field_validator("captured_at", mode="before")
    @classmethod
    def _unix_to_datetime(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v

    @field_validator("captured_at")
    @classmethod
    def _within_window(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        if v > now + MAX_CLOCK_SKEW:
            raise ValueError("captured_at is in the future - check device clock")
        if v < now - MAX_BUFFER_AGE:
            raise ValueError("captured_at is older than 6h buffer window")

        return v

class ReadingBatchIn(BaseModel):
    sensor_id: int
    readings: list[ReadingIn] = Field(min_length=1, max_length=MAX_BATCH_SIZE)