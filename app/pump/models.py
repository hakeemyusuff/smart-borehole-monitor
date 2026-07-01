from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import text
from app.core.schemas import timestamp


class PumpStatus(str, Enum):
    ON = "on"
    OFF = "off"


class PumpAction(str, Enum):
    TURNED_ON = "turned_on"
    TURNED_OFF = "turned_off"


class PumpTrigger(str, Enum):
    AUTOMATIC_SCHEDULE = "automatic_schedule"
    MANUAL_OVERRIDE = "manual_override"
    CRITICAL_SAFETY = "critical_safety"


class DaysOfTheWeek(str, Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class ScheduleStatus(str, Enum):
    UPCOMING = "upcoming"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class Pump(SQLModel, table=True):
    __tablename__ = "pump"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    status: PumpStatus = Field(default=PumpStatus.OFF)
    power_rating: float
    depth: float
    last_status_change: Optional[datetime] = Field(default=None)


class PumpHistory(SQLModel, table=True):
    __tablename__ = "pump_history"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    pump_id: Optional[int] = Field(default=None, foreign_key="pump.id")
    action: PumpAction
    triggered_by: PumpTrigger
    created_at: datetime = timestamp


class Schedule(SQLModel, table=True):
    __tablename__ = "schedule"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    pump_id: Optional[int] = Field(default=None, foreign_key="pump.id")
    borehole_id: Optional[int] = Field(default=None, foreign_key="borehole.id")
    prediction_id: Optional[int] = Field(
        default=None,
        foreign_key="prediction.id",
        nullable=True,
    )
    start_time: datetime
    end_time: datetime
    days_of_the_week: DaysOfTheWeek
    status: ScheduleStatus = Field(default=ScheduleStatus.UPCOMING)
