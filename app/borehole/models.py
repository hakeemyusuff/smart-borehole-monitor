from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import datetime, timezone


class Borehole(SQLModel, table=True):
    __tablename__ = "borehole"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    total_depth: float
    location_id: Optional[int] = Field(default=None, foreign_key="location.id")
    critical_low_level: float
    optimal_high_level: float
    soil_characteristic: str
    water_body_proximity: float
    topography: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
    )
