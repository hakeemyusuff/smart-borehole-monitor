from fastapi import Depends, APIRouter, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.core.database import get_session
from app.core.schemas import ApiResponse
from app.location.services import create_location, get_location, get_locations
from app.location.models import Location
from pydantic import BaseModel

router = APIRouter(prefix="/locations", tags=["Locations"])


class LocationCreate(BaseModel):
    name: str


@router.post(
    "/",
    response_model=ApiResponse[Location],
    status_code=status.HTTP_201_CREATED,
)
async def create(
    payload: LocationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No user with this ID"
        )

    location = await create_location(payload.name, current_user.id, session)
    return ApiResponse[Location](
        status="success",
        message="Location added successfully.",
        data=location,
    )


@router.get("/", response_model=ApiResponse[list[Location]])
async def list_all(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No user with this ID"
        )
    locations = await get_locations(current_user.id, session)
    return ApiResponse[list[Location]](
        status="success",
        message="",
        data=locations,
    )


@router.get("/{location_id}", response_model=ApiResponse[Location])
async def retrieve(
    location_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No user with this ID"
        )
    try:
        location = await get_location(current_user.id, location_id, session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ApiResponse[Location](
        status="success",
        message="",
        data=location,
    )
