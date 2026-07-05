import httpx
from typing import Any
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.weather.models import Weather
from app.location.models import Location
from app.auth.models import User

# Hardocoded Ibadan Longitude and Latitude
# TODO: Make use of longitude and latitude from location

LONGITUDE = 3.90
LATITUDE = 7.44

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def _verify_location_ownership(
    user_id: int,
    location_id: int,
    session: AsyncSession,
) -> Location:
    result = await session.exec(
        select(Location).where(
            Location.id == location_id,
            Location.user_id == user_id,
        )
    )
    location = result.first()

    if location is None:
        raise ValueError("No location found for this user.")

    return location


async def fetch_weather(url: str, lat: float, long: float) -> dict[str, Any]:
    """
    This calls the open-meteo api and returns the parsed weather values
    """

    params = {
        "latitude": lat,
        "longitude": long,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ValueError(f"Weather API request failed: {e}")

        data = response.json()

        current = data.get("current")
        if current is None:
            raise ValueError("Unexpected weather API response: missing 'current'")

        return {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
        }


async def fetch_and_save_weather(
    location_id: int,
    lat: float,
    long: float,
    session: AsyncSession,
) -> Weather:
    """
    Fetches weather data for the given location and saves it to the database
    """
    parsed_data = await fetch_weather(OPEN_METEO_URL, lat, long)

    weather = Weather(
        location_id=location_id,
        temperature=parsed_data["temperature"],
        humidity=parsed_data["humidity"],
        precipitation=parsed_data["precipitation"],
    )

    session.add(weather)
    await session.commit()
    await session.refresh(weather)

    return weather


async def get_weathers(
    location_id: int,
    user_id: int,
    session: AsyncSession,
) -> list[Weather]:
    await _verify_location_ownership(user_id, location_id, session)

    result = await session.exec(
        select(Weather).where(Weather.location_id == location_id),
    )

    weathers = result.all()

    return list(weathers)
