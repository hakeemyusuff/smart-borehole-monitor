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


@router.get("/{borehole_id}", response_model=ApiResponse[Pump])
async def retrieve_pump(
    borehole_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        pump = await get_pump(current_user.id, borehole_id, session)  # type: ignore
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    return ApiResponse[Pump](
        status="success",
        message="",
        data=pump,
    )


@router.get(
    "/pump-histories/{borehole_id}",
    response_model=ApiResponse[PaginatedDataEnvelope[PumpHistory]],
)
async def list_pump_histories(
    borehole_id: int,
    skip: int = Query(0, ge=0, description="Items to skip (offset)"),
    limit: int = Query(
        50,
        ge=1,
        le=1000,
        description="Max items to return (limit)",
    ),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        histories, total_count = await get_pump_history(
            borehole_id,
            current_user.id,  # type: ignore
            session,
            skip=skip,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[PaginatedDataEnvelope[PumpHistory]](
        status="success",
        message="ok",
        data=PaginatedDataEnvelope(
            items=histories,
            total=total_count,
            limit=limit,
            offset=skip,
        ),
    )
