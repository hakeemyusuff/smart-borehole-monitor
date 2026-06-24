from app.core.database import engine
from sqlalchemy import text
import asyncio

async def create_connection():
    async with engine.connect() as connection:
        query = await connection.execute(text("SELECT 1"))
        print(query)
        

asyncio.run(create_connection())