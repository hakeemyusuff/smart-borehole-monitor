from fastapi import HTTPException, APIRouter, status, Depends, Header, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.models import User
from app.auth.dependencies import get_current_user
from app.core.database import get_session
from app.core.schemas import ApiResponse, PaginatedDataEnvelope
from app.sensor.services import _verify_borehole_ownership
from app.pump.services import (
    create_pump,
    get_pump,
    change_pump_status,
    get_pump_history,
)
from app.pump.models import Pump, PumpHistory, PumpAction, PumpStatus, PumpTrigger
from app.pump.schemas import PumpCreate

router = APIRouter(prefix="/pumps", tags=["pumps"])


@router.post(
    "/",
    response_model=ApiResponse[Pump],
    status_code=status.HTTP_201_CREATED,
)
async def create(
    payload: PumpCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        pump = await create_pump(
            payload.model_dump(),
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[Pump](
        status="success",
        message="Pump added successfully",
        data=pump,
    )
