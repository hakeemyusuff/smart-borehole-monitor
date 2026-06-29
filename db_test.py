from app.core.database import engine
from sqlalchemy import text
import asyncio

async def create_connection():
    async with engine.connect() as connection:
        query = await connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
        )
        print(query.fetchall())


asyncio.run(create_connection())
