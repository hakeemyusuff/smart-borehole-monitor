import pandas as pd

from datetime import datetime, timezone, timedelta
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
from app.sensor.models import FlowReading
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


async def get_pump_windows(
    borehole_id: int,
    user_id: int,
    session: AsyncSession,
    lookback: timedelta = timedelta(days=30),
    gap_minutes: float = 10.0,
) -> list[dict]:
    """Group flow readings into pumping windows (runs of activity separated by
    gaps > gap_minutes) and sum the volume pumped in each.

    Volume is integrated as rate (L/min) × minutes-since-previous-sample within
    a window — an ESTIMATE from the instantaneous rate, not a metered total."""

    await _verify_borehole_ownership(borehole_id, user_id, session)

    now = datetime.now(timezone.utc)

    rows = (
        await session.exec(
            select(FlowReading.captured_at, FlowReading.abstraction_rate)
            .where(FlowReading.borehole_id == borehole_id)
            .where(FlowReading.captured_at >= now - lookback)
            .where(FlowReading.abstraction_rate > 0)
            .order_by(FlowReading.captured_at)
        )
    ).all()

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["captured_at", "rate"])
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)

    # Minutes since previous row; a gap > threshold starts a new window.
    gap = df["captured_at"].diff().dt.total_seconds() / 60.0
    df["new_window"] = (gap > gap_minutes) | gap.isna()
    df["window_id"] = df["new_window"].cumsum()

    # Duration each row represents = minutes to the NEXT row within the window
    # (last row of a window represents a nominal 1 min, since we can't see past it).
    windows = []
    for _, g in df.groupby("window_id"):
        g = g.sort_values("captured_at").reset_index(drop=True)
        # minutes to next sample; last row gets the window's median interval
        dt_next = g["captured_at"].shift(-1) - g["captured_at"]
        dt_min = dt_next.dt.total_seconds() / 60.0
        median_interval = dt_min.median()
        dt_min = dt_min.fillna(median_interval if pd.notna(median_interval) else 1.0)
        volume = float((g["rate"] * dt_min).sum())
        windows.append(
            {
                "start": g["captured_at"].iloc[0].to_pydatetime(),
                "end": g["captured_at"].iloc[-1].to_pydatetime(),
                "volume_litres": round(volume, 1),
                "duration_min": round(float(dt_min.sum()), 1),
                "avg_rate": round(float(g["rate"].mean()), 1),
            }
        )

    windows.sort(key=lambda w: w["start"], reverse=True)  # Newest first
    return windows
