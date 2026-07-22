from pydantic import BaseModel

class PumpCreate(BaseModel):
    borehole_id: int
    power_rating: float
    depth: float
    
    