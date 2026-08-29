from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User
from app.core.database import get_session
from app.auth.dependencies import get_current_user
from app.core.schemas import ApiResponse
from app.sensor.services import Range, RANGE_CONFIG
from app.ml.services import get_prediction_chart
from app.ml.schemas import PredictionChartPoint

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get(
    "/{borehole_id}/chart", response_model=ApiResponse[list[PredictionChartPoint]]
)
async def prediction_chart(
    borehole_id: int,
    range_: Range = Query(Range.day),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    try:
        data = await get_prediction_chart(
            borehole_id,
            current_user.id,  # type: ignore
            session,
            lookback=RANGE_CONFIG[range_]["lookback"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[list[PredictionChartPoint]](
        status="success",
        message="prediction chart data",
        data=data,
    )
