from datetime import datetime, timezone
from typing import Any
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import func, select
from app.auth.models import User
from app.pump.models import (
    Pump,
    PumpHistory,
    PumpAction,
    PumpStatus,
    PumpTrigger,
)
from app.borehole.models import Borehole
from app.location.models import Location
from app.sensor.services import _verify_borehole_ownership


async def create_pump(
    data: dict[str, Any],
    user_id: int,
    session: AsyncSession,
) -> Pump:

    borehole = await _verify_borehole_ownership(data["borehole_id"], user_id, session)
    result = await session.exec(
        select(Pump).where(
            Pump.borehole_id == borehole.id,
        )
    )

    existing_pump = result.first()
    if existing_pump is not None:
        raise ValueError("This borehole already has a pump installed.")

    pump = Pump(**data)
    session.add(pump)
    await session.commit()
    await session.refresh(pump)

    return pump


async def get_pump(user_id: int, borehole_id: int, session: AsyncSession) -> Pump:
    await _verify_borehole_ownership(borehole_id, user_id, session)
    result = await session.exec(
        select(Pump)
        .join(Borehole, Borehole.id == Pump.borehole_id)  # type: ignore
        .where(Pump.borehole_id == borehole_id)
    )

    pump = result.first()
    if pump is None:
        raise ValueError("Pump not found for this user")

    return pump


async def change_pump_status(
    borehole_id: int,
    new_status: PumpStatus,
    pump_trigger: PumpTrigger,
    session: AsyncSession,
) -> tuple[Pump, PumpHistory | None]:
    result = await session.exec(
        select(Pump).where(
            Pump.borehole_id == borehole_id,
        )
    )
    pump = result.first()

    if pump is None:
        raise ValueError("No pump installed in this borehole")

    if pump.status == new_status:
        return pump, None

    pump.status = new_status
    pump.last_status_change = datetime.now(timezone.utc)
    if new_status == PumpStatus.ON:
        pump_action = PumpAction.TURNED_ON
    else:
        pump_action = PumpAction.TURNED_OFF

    pump_history = PumpHistory(
        pump_id=pump.id,
        action=pump_action,
        triggered_by=pump_trigger,
    )

    session.add(pump)
    session.add(pump_history)
    await session.commit()
    await session.refresh(pump)
    await session.refresh(pump_history)

    return pump, pump_history


async def get_pump_history(
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[PumpHistory], int]:
    await _verify_borehole_ownership(borehole_id, user_id, session)

    count_result = await session.exec(
        select(func.count(PumpHistory.id))
        .join(Pump, PumpHistory.pump_id == Pump.id) # type: ignore
        .where(Pump.borehole_id == borehole_id)
    )
    total_count = count_result.first() or 0
    
    data = await session.exec(
        select(PumpHistory)
        .join(Pump, PumpHistory.pump_id == Pump.id) # type: ignore
        .where(Pump.borehole_id == borehole_id)
        .order_by(PumpHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    
    pump_histories = data.all()
    
    return list(pump_histories), total_count