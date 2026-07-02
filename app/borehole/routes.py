from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession
from app.borehole.services import create_borehole, get_boreholes, get_borehole
from app.borehole.models import Borehole
from app.auth.models import User
from app.auth.dependencies import get_current_user
from app.core.database import get_session
from app.core.schemas import ApiResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/boreholes", tags=["Boreholes"])


class BoreholeCreate(BaseModel):
    name: str
    total_depth: float
    location_id: int
    critical_low_level: float
    optimal_high_level: float
    soil_characteristic: str
    water_body_proximity: float
    topography: str


@router.post(
    "/",
    response_model=ApiResponse[Borehole],
    status_code=status.HTTP_201_CREATED,
)
async def create(
    payload: BoreholeCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        borehole = await create_borehole(
            payload.model_dump(), user_id=current_user.id, session=session  # type: ignore
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ApiResponse[Borehole](
        status="success",
        message="Borehole added successfully",
        data=borehole,
    )


@router.get("/", response_model=ApiResponse[list[Borehole]])
async def list_all(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    boreholes = await get_boreholes(current_user.id, session)  # type: ignore
    return ApiResponse[list[Borehole]](
        status="success",
        message="",
        data=boreholes,
    )


@router.get("/{borehole_id}", response_model=ApiResponse[Borehole])
async def retrieve(
    borehole_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        borehole = await get_borehole(
            borehole_id,
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    return ApiResponse[Borehole](
        status="success",
        message="",
        data=borehole,
    )
