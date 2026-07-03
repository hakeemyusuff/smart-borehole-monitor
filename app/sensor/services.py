import secrets
from typing import Any, Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.auth.services import hash_password, verify_password
from app.sensor.models import Sensor, SensorStatus, SensorType
from app.borehole.models import Borehole
from app.location.models import Location
from datetime import datetime, timezone


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
        .join(Borehole, Sensor.borehole_id == Borehole.id) # type: ignore
        .join(Location, Borehole.location_id == Location.id) # type: ignore
        .where(Sensor.id == sensor_id, Location.user_id == user_id)
    )
    
    sensor = result.first()
    if sensor is None:
        raise ValueError("Sensor not found for this user")
    return sensor


async def authenticate_device(
    sensor_id: int, raw_key: str, session: AsyncSession
) -> Sensor:
    """
    Called by ingestion. Looks up the sensor by id, then verifies the presented
    raw key against the stored hash.
    No user-session auth - this is the device auth path.
    """
    
    result = await session.exec(
        select(Sensor).where(Sensor.id == sensor_id)
    )
    
    sensor = result.first()
    if sensor is None:
        raise ValueError("Invalid device credentials")
    
    if not verify_password(raw_key, sensor.device_key): # type: ignore
        raise ValueError("Invalid device credentials")
    
    sensor.status = SensorStatus.ACTIVE
    sensor.last_seen = datetime.now(timezone.utc)
    session.add(sensor)
    await session.commit()
    await session.refresh(sensor)
    return sensor
    
