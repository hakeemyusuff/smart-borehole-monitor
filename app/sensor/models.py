from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import text
from app.core.schemas import optional_timestamp_field, timestamp_field


class SensorStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAULTY = "faulty"


class SensorType(str, Enum):
    PRESSURE_TRANSDUCER = "pressure_transducer"
    FLOW_METER = "flow_meter"
    ESP32 = "esp32"


class Sensor(SQLModel, table=True):
    __tablename__ = "sensor"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: int = Field(foreign_key="borehole.id")
    status: SensorStatus = Field(default=SensorStatus.INACTIVE)
    device_key: Optional[str] = Field(default=None)
    type: SensorType
    last_seen: Optional[datetime] = optional_timestamp_field()


class ReadingMixin(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    sensor_id: Optional[int] = Field(default=None, foreign_key="sensor.id")
    created_at: datetime = timestamp_field()


class WaterLevelReading(ReadingMixin, table=True):
    __tablename__ = "water_level_reading"  # type: ignore

    water_level: float

class FlowReading(ReadingMixin, table=True):
    __tablename__ = "flow_reading"  # type: ignore

    abstraction_rate: float