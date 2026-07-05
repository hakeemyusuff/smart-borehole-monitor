from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.weather.tasks import fetch_weathers_for_all_locations

scheduler = AsyncIOScheduler()

def start_scheduler():
    scheduler.add_job(
        fetch_weathers_for_all_locations,
        trigger="interval",
        minutes=60,
        id="fetch_weather",
        replace_existing=True,
    )
    scheduler.start()
    
def stop_scheduler():
    scheduler.shutdown()
    
    