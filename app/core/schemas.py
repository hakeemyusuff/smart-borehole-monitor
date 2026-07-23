from typing import TypeVar, Generic, Optional
from pydantic import BaseModel
from sqlmodel import Field
from sqlalchemy import text, DateTime
from datetime import datetime, timezone

def timestamp_field():
    return Field(
    default_factory=lambda: datetime.now(timezone.utc),
    sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    sa_type=DateTime(timezone=True), # type: ignore
)

def optional_timestamp_field():
    return Field(
        default=None,
        sa_type=DateTime(timezone=True), # type: ignore
    )

T = TypeVar("T")
Y = TypeVar("Y")

class ApiResponse(BaseModel, Generic[T]):
    status: str
    message: str
    data: Optional[T] = None
    
class PaginatedDataEnvelope(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    
class StatusDataEnvelope(BaseModel, Generic[T, Y]):
    pump: T
    pump_history: Optional[Y]