from pydantic import BaseModel
from app.pump.models import PumpStatus

class PumpCreate(BaseModel):
    borehole_id: int
    power_rating: float
    depth: float
    
class StatusChange(BaseModel):
    new_status: PumpStatus
    