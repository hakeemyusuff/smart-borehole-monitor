from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import text

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
    name: str
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    status: SensorStatus = Field(default=SensorStatus.INACTIVE)
    type: SensorType
    last_seen: Optional[datetime] = Field(default=None)


class ReadingMixin(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    sensor_id: Optional[int] = Field(default=None, foreign_key="sensor.id")
    raw_reading: float
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


class WaterLevelReading(ReadingMixin, table=True):
    __tablename__ = "water_level_reading"  # type: ignore

    calculated_water_depth: float


class FlowReading(ReadingMixin, table=True):
    __tablename__ = "flow_reading"  # type: ignore

    calculated_flow_rate: float
    cummulative_volume: float
