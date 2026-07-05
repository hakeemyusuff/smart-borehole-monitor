from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.schemas import ApiResponse
from app.weather.models import Weather
from app.weather.services import (
    LATITUDE,
    LONGITUDE,
    _verify_location_ownership,
    fetch_and_save_weather,
    get_weathers,
)
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.core.database import get_session

router = APIRouter(prefix="/weathers", tags=["Weathers"])


@router.post(
    "/fetch/{location_id}",
    response_model=ApiResponse[Weather],
    status_code=status.HTTP_201_CREATED,
)
async def trigger_weather_fetch(
    location_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        location = await _verify_location_ownership(
            current_user.id,  # type: ignore
            location_id,
            session,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
        
    lat = location.latitude if location.latitude is not None else LATITUDE
    long = location.longitude if location.longitude is not None else LONGITUDE
    
    try:
        weather = await fetch_and_save_weather(location_id, lat, long, session)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )

    return ApiResponse[Weather](
        status="success",
        message="Weather fetched and stored",
        data=weather,
    )


@router.get(
    "/{location_id}",
    response_model=ApiResponse[list[Weather]],
)
async def list_all(
    location_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        weathers = await get_weathers(
            location_id,
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[list[Weather]](
        status="success",
        message="",
        data=weathers,
    )
