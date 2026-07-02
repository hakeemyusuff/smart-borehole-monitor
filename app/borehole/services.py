from typing import Any
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.borehole.models import Borehole
from app.location.models import Location


async def create_borehole(
    data: dict[str, Any],
    user_id: int,
    session: AsyncSession,
) -> Borehole:
    # To verify the location belong to the user before attaching borehole
    result = await session.exec(
        select(Location).where(Location.id == data["location_id"])
    )
    location = result.first()
    if location is None:
        raise ValueError("Location not found for this user.")

    borehole = Borehole(**data)
    session.add(borehole)
    await session.commit()
    await session.refresh(borehole)

    return borehole


async def get_boreholes(user_id: int, session: AsyncSession) -> list[Borehole]:
    boreholes = await session.exec(
        select(Borehole)
        .join(
            Location,
            Borehole.location_id == Location.id,  # type: ignore
        )
        .where(Location.user_id == user_id)
    )

    return list(boreholes.all())


async def get_borehole(
    borehole_id: int, user_id: int, session: AsyncSession
) -> Borehole:
    result = await session.exec(
        select(Borehole)
        .join(
            Location,
            Borehole.location_id == Location.id, #type: ignore
        )
        .where(Location.user_id == user_id, Borehole.id == borehole_id)
    )
    borehole = result.first()

    if borehole is None:
        raise ValueError("Borehole not found for this user")

    return borehole
