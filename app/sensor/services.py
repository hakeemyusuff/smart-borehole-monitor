import secrets
from enum import Enum
from typing import Any, Optional
from sqlmodel import select, func, text
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.services import hash_password, verify_password
from app.sensor.models import (
    Sensor,
    SensorStatus,
    SensorType,
    FlowReading,
    WaterLevelReading,
)
from app.borehole.models import Borehole
from app.location.models import Location
from datetime import datetime, timezone, timedelta


class Range(str, Enum):
    day = "day"
    week = "week"
    month = "month"


RANGE_CONFIG = {
    Range.day: {"lookback": timedelta(days=1), "bucket": None},
    Range.week: {"lookback": timedelta(days=7), "bucket": "1 hour"},
    Range.month: {"lookback": timedelta(days=30), "bucket": "24 hours"},
}


async def _verify_borehole_ownership(
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
):
    result = await session.exec(
        select(Borehole)
        .join(Location, Borehole.location_id == Location.id)  # type: ignore
        .where(Borehole.id == borehole_id, Location.user_id == user_id)
    )
    borehole = result.first()

    if borehole is None:
        raise ValueError("Borehole not found for this user")

    return borehole


async def create_sensor(
    data: dict[str, Any],
    user_id: int,
    session: AsyncSession,
) -> tuple[Sensor, Optional[str]]:
    """
    Returns (sensor, raw_key)
    raw_key is only non-None for ESP32 sensors, and it is the ONE time the
    plaintext key is ever available. It is stored hashed.
    """

    await _verify_borehole_ownership(data["borehole_id"], user_id, session)

    raw_key: Optional[str] = None

    if data.get("type") == SensorType.ESP32:
        raw_key = secrets.token_urlsafe(32)
        data["device_key"] = hash_password(raw_key)

    sensor = Sensor(**data)
    session.add(sensor)
    await session.commit()
    await session.refresh(sensor)

    return sensor, raw_key


async def get_sensors(
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
) -> list[Sensor]:
    await _verify_borehole_ownership(borehole_id, user_id, session)
    result = await session.exec(
        select(Sensor).where(Sensor.borehole_id == borehole_id),
    )

    sensors = list(result.all())
    return sensors


async def get_sensor(
    sensor_id: int,
    user_id: int,
    session: AsyncSession,
) -> Sensor:
    result = await session.exec(
        select(Sensor)
        .join(Borehole, Sensor.borehole_id == Borehole.id)  # type: ignore
        .join(Location, Borehole.location_id == Location.id)  # type: ignore
        .where(Sensor.id == sensor_id, Location.user_id == user_id)
    )

    sensor = result.first()
    if sensor is None:
        raise ValueError("Sensor not found for this user")
    return sensor


async def ingest_reading(
    *,
    esp32_id: int,
    reading_sensor_id: int,
    device_key: str,
    reading_value: float,
    expected_type: SensorType,
    session: AsyncSession,
):
    """
    This authenticate the ESP32, validates that the reading is coming from the
    right sensor and it belongs to the same borehole, and it is of expected type,
    stores the reading and updates last_seen on sensors.
    Returns the created reading object (WaterLevelReading or FlowReading).
    """
    # Lookup and Authenticate the ESP32
    result = await session.exec(select(Sensor).where(Sensor.id == esp32_id))
    esp32 = result.first()

    if esp32 is None or esp32.device_key is None:
        raise ValueError("Invalid device credentials")
    if not verify_password(device_key, esp32.device_key):
        raise ValueError("Invalid device credentials")

    # LookUP THE READING PRODUCING SENSOR
    result = await session.exec(
        select(Sensor).where(Sensor.id == reading_sensor_id),
    )
    producing_sensor = result.first()
    if producing_sensor is None:
        raise ValueError("Reading Sensor not found")

    # Check that the producing sensor is in the same borehole with the ESP32
    if not (esp32.borehole_id == producing_sensor.borehole_id):
        raise ValueError("Sensor does not belong to this device's borehole")

    # Ensure the type of Sensor matches the readings
    if not (producing_sensor.type == expected_type):
        raise ValueError(
            f"Endpoint expected {expected_type.value}, got {producing_sensor.type.value}"
        )

    # Build the correct reading level
    if expected_type == SensorType.PRESSURE_TRANSDUCER:
        reading = WaterLevelReading(
            borehole_id=producing_sensor.borehole_id,
            sensor_id=producing_sensor.id,
            water_level=reading_value,
        )
    else:  # FLOW READING
        reading = FlowReading(
            borehole_id=producing_sensor.borehole_id,
            sensor_id=producing_sensor.id,
            abstraction_rate=reading_value,
        )

    # Update heartbeats on both sensors
    now = datetime.now(timezone.utc)
    producing_sensor.status = SensorStatus.ACTIVE
    producing_sensor.last_seen = now
    esp32.status = SensorStatus.ACTIVE
    esp32.last_seen = now

    # Commit changes to database
    session.add(reading)
    session.add(producing_sensor)
    session.add(esp32)
    await session.commit()
    await session.refresh(reading)

    return reading


async def list_water_levels(
    sensor_id: int,
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[WaterLevelReading], int]:
    await _verify_borehole_ownership(borehole_id, user_id, session)

    count_stmt = select(func.count(WaterLevelReading.id)).where(
        WaterLevelReading.sensor_id == sensor_id,
        WaterLevelReading.borehole_id == borehole_id,
    )
    count_result = await session.exec(count_stmt)
    total_count = count_result.first() or 0

    data_stmt = (
        select(WaterLevelReading)
        .where(
            WaterLevelReading.sensor_id == sensor_id,
            WaterLevelReading.borehole_id == borehole_id,
        )
        .order_by(WaterLevelReading.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    result = await session.exec(data_stmt)
    readings = result.all()

    return list(readings), total_count


async def list_flow_readings(
    sensor_id: int,
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[FlowReading], int]:
    await _verify_borehole_ownership(borehole_id, user_id, session)

    count_stmt = select(func.count(FlowReading.id)).where(
        FlowReading.sensor_id == sensor_id,
        FlowReading.borehole_id == borehole_id,
    )

    count_result = await session.exec(count_stmt)
    total_count = count_result.first() or 0

    data_stmt = (
        select(FlowReading)
        .where(
            FlowReading.sensor_id == sensor_id,
            FlowReading.borehole_id == borehole_id,
        )
        .order_by(FlowReading.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    result = await session.exec(data_stmt)
    readings = result.all()

    return list(readings), total_count


async def get_raw_readings(
    session,
    model,
    value_column,
    sensor_id,
    borehole_id,
    since,
):
    stmt = (
        select(model.created_at, value_column)
        .where(model.sensor_id == sensor_id)
        .where(model.borehole_id == borehole_id)
        .where(model.created_at >= since)
        .order_by(model.created_at)
    )
    result = await session.exec(stmt)
    rows = result.all()

    return [{"t": r[0].isoformat(), "value": r[1]} for r in rows]


async def get_bucketed_readings(
    session,
    model,
    value_column,
    sensor_id,
    borehole_id,
    since,
    bucket,
):
    origin = datetime(2000, 1, 1, tzinfo=timezone.utc)

    bucket_expr = func.date_bin(
        text(f"INTERVAL '{bucket}'"),
        model.created_at,
        origin,
    )
    bucket_col = bucket_expr.label("bucket")
    avg_col = func.avg(value_column).label("avg_value")

    stmt = (
        select(bucket_col, avg_col)
        .where(model.sensor_id == sensor_id)
        .where(model.borehole_id == borehole_id)
        .where(model.created_at >= since)
        .group_by(bucket_expr)
        .order_by(bucket_expr)
    )

    result = await session.exec(stmt)
    rows = result.all()

    return [
        {"t": r[0].isoformat(), "value": float(r[1]) if r[1] is not None else None}
        for r in rows
    ]


async def get_readings_for_range(
    session,
    model,
    value_column,
    sensor_id,
    borehole_id,
    range_,
):
    cfg = RANGE_CONFIG[range_]
    since = datetime.now(timezone.utc) - cfg["lookback"]

    if cfg["bucket"] is None:
        return await get_raw_readings(
            session,
            model,
            value_column,
            sensor_id,
            borehole_id,
            since,
        )
    return await get_bucketed_readings(
        session, model, value_column, sensor_id, borehole_id, since, cfg["bucket"]
    )
