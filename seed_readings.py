import asyncio
import math
import random
from datetime import datetime, timedelta, timezone

from app.core.database import async_session_maker
from app.sensor.models import WaterLevelReading, FlowReading

WATER_SENSOR_ID = 12
WATER_BOREHOLE_ID = 3
FLOW_SENSOR_ID = 10
FLOW_BOREHOLE_ID = 3

DAYS_OF_HISTORY = 1
CHUNK_SIZE = 5000

IDLE_INTERVAL_MIN = 30
PUMP_INTERVAL_SEC = 30

OPTIMAL_LEVEL = 50.0
CRITICAL_LOW = 8.0
DAILY_SWING = 0.4
NOISE_M = 0.05

RECOVERY_TAU_MIN = 120.0
DRAWDOWN_RATE_M_PER_MIN = 0.9

FLOW_BASE = 15.0
FLOW_NOISE = 1.2

PUMP_WINDOWS = [(7, 0), (18, 0)]


def pump_windows_for_day(day):
    windows = []
    for start_hour, start_min in PUMP_WINDOWS:
        start = day.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        span_m = (OPTIMAL_LEVEL - CRITICAL_LOW) / DRAWDOWN_RATE_M_PER_MIN
        jitter = (day.toordinal() * (start_hour + 1)) % 15
        end = start + timedelta(minutes=span_m + jitter)
        windows.append((start, end))
    return windows


def active_window(ts, windows):
    for start, end in windows:
        if start <= ts < end:
            return start, end
    return None


def most_recent_end(ts, windows):
    ends = [end for _, end in windows if ts >= end]
    return max(ends) if ends else None


def daily_component(ts):
    sec = ts.hour * 3600 + ts.minute * 60 + ts.second
    return DAILY_SWING * math.sin(2 * math.pi * sec / 86400)


def water_level_at(ts, windows):
    win = active_window(ts, windows)
    if win:
        start, _ = win
        elapsed = (ts - start).total_seconds() / 60.0
        level = OPTIMAL_LEVEL - DRAWDOWN_RATE_M_PER_MIN * elapsed
        level = max(level, CRITICAL_LOW)
    else:
        last_end = most_recent_end(ts, windows)
        if last_end is not None:
            since = (ts - last_end).total_seconds() / 60.0
            frac = 1 - math.exp(-since / RECOVERY_TAU_MIN)
            level = CRITICAL_LOW + (OPTIMAL_LEVEL - CRITICAL_LOW) * frac
        else:
            level = OPTIMAL_LEVEL
    level += daily_component(ts) + random.uniform(-NOISE_M, NOISE_M)
    return round(min(max(level, 0.0), OPTIMAL_LEVEL), 3)


def generate_water_rows(start, end):
    ts = start
    while ts < end:
        windows = pump_windows_for_day(ts) + pump_windows_for_day(
            ts - timedelta(days=1)
        )
        pumping = active_window(ts, windows) is not None
        yield WaterLevelReading(
            borehole_id=WATER_BOREHOLE_ID,
            sensor_id=WATER_SENSOR_ID,
            water_level=water_level_at(ts, windows),
            calculated_water_depth=None,
            created_at=ts,
        )
        if pumping:
            ts += timedelta(seconds=PUMP_INTERVAL_SEC)
        else:
            ts += timedelta(minutes=IDLE_INTERVAL_MIN)


def generate_flow_rows(start, end):
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end
    while day < end_day:
        for w_start, w_end in pump_windows_for_day(day):
            ts = w_start
            while ts < w_end and ts < end:
                if ts >= start:
                    yield FlowReading(
                        borehole_id=FLOW_BOREHOLE_ID,
                        sensor_id=FLOW_SENSOR_ID,
                        raw_reading=round(
                            FLOW_BASE + random.uniform(-FLOW_NOISE, FLOW_NOISE), 3
                        ),
                        calculated_flow_rate=None,
                        cummulative_volume=None,
                        created_at=ts,
                    )
                ts += timedelta(seconds=PUMP_INTERVAL_SEC)
        day += timedelta(days=1)


async def insert_in_chunks(session, row_iter, label):
    batch, total = [], 0
    for row in row_iter:
        batch.append(row)
        if len(batch) >= CHUNK_SIZE:
            session.add_all(batch)
            await session.commit()
            total += len(batch)
            print(f"  {label}: committed {total} rows...")
            batch = []
    if batch:
        session.add_all(batch)
        await session.commit()
        total += len(batch)
    print(f"  {label}: DONE — {total} rows total.")
    return total


async def main():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=DAYS_OF_HISTORY)
    print(
        f"Seeding {DAYS_OF_HISTORY} day(s): {start.isoformat()} -> {now.isoformat()}\n"
    )
    async with async_session_maker() as session:
        print("Water level:")
        await insert_in_chunks(session, generate_water_rows(start, now), "water")
        print("\nFlow:")
        await insert_in_chunks(session, generate_flow_rows(start, now), "flow")
    print(
        "\nDone. Verify created_at is spread and flow only exists during pump windows."
    )


if __name__ == "__main__":
    asyncio.run(main())
