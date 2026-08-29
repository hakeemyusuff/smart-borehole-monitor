from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PredictionChartPoint(BaseModel):
    t: datetime  # the predicted-for time (target hour)
    predicted: float  # frozen model output for that time
    actual: Optional[float] = None  # nearest real reading, null if future/missing
    confidence: float  # inter-tree agreement, 0..1
