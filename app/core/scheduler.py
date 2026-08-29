from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.weather.tasks import fetch_weathers_for_all_locations
from app.ml.tasks import run_inference_job

scheduler = AsyncIOScheduler()

def start_scheduler():
    scheduler.add_job(
        fetch_weathers_for_all_locations,
        trigger="interval",
        minutes=60,
        id="fetch_weather",
        replace_existing=True,
    )
    scheduler.add_job(
        run_inference_job,
        trigger="cron",
        minute=5,
        id="run_inference",
        replace_existing=True,
    )
    scheduler.start()
    
def stop_scheduler():
    scheduler.shutdown()
    
    