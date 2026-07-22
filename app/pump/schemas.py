from pydantic import BaseModel
from typing import Any

class PumpCreate(BaseModel):
    borehole_id: int
    power_rating: float
    depth: float
    
    