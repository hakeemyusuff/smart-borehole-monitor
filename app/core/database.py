from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

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