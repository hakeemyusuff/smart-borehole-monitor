from fastapi import APIRouter, HTTPException, Depends, status, Header, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.models import User
from app.auth.dependencies import get_current_user
from app.core.database import get_session
from app.core.schemas import ApiResponse
from app.sensor.services import (
    _verify_borehole_ownership,
    create_sensor,
    get_sensor,
    get_sensors,
    ingest_reading,
    list_flow_readings,
    list_water_levels,
    Range,
    get_readings_for_range,
)
from app.sensor.schemas import (
    SensorCreate,
    SensorCreateResponse,
    SensorPublic,
    ReadingIn,
)
from app.sensor.models import Sensor, SensorType, FlowReading, WaterLevelReading

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


@router.post(
    "/readings/water-level",
    status_code=status.HTTP_201_CREATED,
    response_model=ApiResponse,
)
async def ingest_water_level(
    payload: ReadingIn,
    x_device_id: int = Header(...),
    x_device_key: str = Header(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        await ingest_reading(
            esp32_id=x_device_id,
            device_key=x_device_key,
            reading_sensor_id=payload.sensor_id,
            reading_value=payload.reading,
            expected_type=SensorType.PRESSURE_TRANSDUCER,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ApiResponse(
        status="success",
        message="Reading recorded",
        data=None,
    )


@router.get(
    "/readings/water-level/{borehole_id}/{sensor_id}",
    response_model=ApiResponse[list[WaterLevelReading]],
)
async def list_all_water_level_readings(
    sensor_id: int,
    borehole_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        readings = await list_water_levels(
            sensor_id,
            borehole_id,
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[list[WaterLevelReading]](
        status="success",
        message="",
        data=readings,
    )


@router.post(
    "/readings/flow-reading",
    response_model=ApiResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_flow_reading(
    payload: ReadingIn,
    x_device_id: int = Header(...),
    x_device_key: str = Header(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        await ingest_reading(
            esp32_id=x_device_id,
            device_key=x_device_key,
            reading_sensor_id=payload.sensor_id,
            reading_value=payload.reading,
            expected_type=SensorType.FLOW_METER,
            session=session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ApiResponse(
        status="success",
        message="Reading recorded",
        data=None,
    )


@router.get(
    "/readings/flow-reading/{borehole_id}/{sensor_id}",
    response_model=ApiResponse[list[FlowReading]],
)
async def list_all_flow_readings(
    sensor_id: int,
    borehole_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    try:
        readings = await list_flow_readings(
            sensor_id,
            borehole_id,
            current_user.id,  # type: ignore
            session,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return ApiResponse[list[FlowReading]](
        status="success",
        message="",
        data=readings,
    )


@router.get("/water-level/{borehole_id}/{sensor_id}/chart")
async def water_level_chart(
    borehole_id: int,
    sensor_id: int,
    range_: Range = Query(Range.day),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    await _verify_borehole_ownership(borehole_id, current_user.id, session)  # type: ignore
    data = await get_readings_for_range(
        session,
        WaterLevelReading,
        WaterLevelReading.water_level,
        sensor_id,
        borehole_id,
        range_,
    )

    return ApiResponse(
        status="success",
        message="water level chart data",
        data=data,
    )


@router.get("/flow-reading/{borehole_id}/{sensor_id}/chart")
async def flow_chart(
    borehole_id: int,
    sensor_id: int,
    range_: Range = Query(Range.day),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    await _verify_borehole_ownership(borehole_id, current_user.id, session)  # type: ignore
    data = await get_readings_for_range(
        session,
        FlowReading,
        FlowReading.raw_reading,
        sensor_id,
        borehole_id,
        range_,
    )

    return ApiResponse(
        status="success",
        message="Flow chart data",
        data=data,
    )