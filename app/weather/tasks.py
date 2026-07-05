from sqlmodel import select
from app.core.database import async_session_maker
from app.location.models import Location
from app.weather.services import fetch_and_save_weather


async def fetch_weathers_for_all_locations():
    """
    Scheduler entry point. Iterate locations that have coordinates and
    fetches wether for each. Errors are isolated per each location so a failure
    doesn't cascade to others.
    """

    async with async_session_maker() as session:
        result = await session.exec(
            select(Location).where(
                Location.latitude != None,
                Location.longitude != None,
            )
        )

        locations = list(result.all())

    for loc in locations:
        try:
            async with async_session_maker() as session:
                await fetch_and_save_weather(
                    loc.id,  # type: ignore
                    loc.latitude,  # type: ignore
                    loc.longitude,  # type: ignore
                    session,
                )
                print(f"[weather job] success for location {loc.id}")
        except Exception as e:
            print(f"[weather job] failed for location {loc.id}: {e}")
