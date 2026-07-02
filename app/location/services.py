from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from app.location.models import Location


async def create_location(
    name: str,
    user_id: int,
    session: AsyncSession,
) -> Location:
    location = Location(name=name, user_id=user_id)
    session.add(location)
    await session.commit()
    await session.refresh(location)
    return location


async def get_locations(user_id: int, session: AsyncSession) -> list[Location]:
    result = await session.exec(
        select(Location).where(Location.user_id == user_id),
    )

    return list(result.all())


async def get_location(
    user_id: int, location_id: int, session: AsyncSession
) -> Location:
    result = await session.exec(
        select(Location).where(
            Location.user_id == user_id,
            Location.id == location_id,
        )
    )
    location = result.first()

    if location is None:
        raise ValueError("No Location found for this user")

    return location
