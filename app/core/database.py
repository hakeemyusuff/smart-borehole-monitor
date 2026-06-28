from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from auth.models import User
from borehole.models import Borehole
from location.models import Location
from ml.models import Prediction
from pump.models import Pump, PumpHistory,Schedule
from sensor.models import Sensor, WaterLevelReading, FlowReading
from weather.models import Weather


#This create the engine for database connection
engine = create_async_engine(settings.database_url, echo=settings.debug)

#It handles the current session for the database connection
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# A function that can be used as a depency injection
async def get_session():
    async with async_session_maker() as session:
        yield session