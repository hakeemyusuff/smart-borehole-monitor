from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timezone
from app.sensor.models import Sensor, SensorType, SensorStatus
from pydantic import ConfigDict
# from sqlmodel import SQLModel

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
    sensor_id: int
    reading: float
    

    
    