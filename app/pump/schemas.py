from datetime import datetime

from pydantic import BaseModel
from app.pump.models import PumpStatus

class PumpCreate(BaseModel):
    borehole_id: int
    power_rating: float
    depth: float
    
class StatusChange(BaseModel):
    new_status: PumpStatus
    
class PumpWindow(BaseModel):
    start: datetime
    end: datetime
    volume_litres: float
    duration_min: float
    avg_rate: float