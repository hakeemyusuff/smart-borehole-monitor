from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.models import User
from app.auth.dependencies import get_current_user
from app.core.database import get_session
from app.core.schemas import ApiResponse
from app.sensor.services import create_sensor, get_sensor, get_sensors
from app.sensor.schemas import SensorCreate, SensorCreateResponse, SensorPublic
from app.sensor.models import Sensor

router = APIRouter(
    prefix="/sensors",
    tags=[
        "Sensors",
    ],
)


@router.post(
    "/",
    response_model=ApiResponse[SensorCreateResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create(
    payload: SensorCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        sensor, raw_key = await create_sensor(
            payload.model_dump(),
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    message = (
        "ESP32 registered. Save this device key now, it will not be shown again."
        if raw_key
        else "Sensor registered successfully"
    )

    return ApiResponse[SensorCreateResponse](
        status="success",
        message=message,
        data=SensorCreateResponse(sensor=sensor, device_key=raw_key),
    )


@router.get(
    "/boreholes/{borehole_id}",
    response_model=ApiResponse[list[SensorPublic]],
)
async def list_sensors_under_borehole(
    borehole_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        sensors = await get_sensors(borehole_id, current_user.id, session)  # type: ignore
        public_sensors = [SensorPublic.model_validate(s) for s in sensors]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[list[SensorPublic]](
        status="success",
        message="",
        data=public_sensors,
    )


@router.get("/{sensor_id}", response_model=ApiResponse[SensorPublic])
async def retrieve(
    sensor_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        sensor = await get_sensor(sensor_id, current_user.id, session)  # type: ignore
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[SensorPublic](
        status="success",
        message="",
        data=SensorPublic.model_validate(sensor),
    )
