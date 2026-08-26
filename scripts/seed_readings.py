from __future__ import annotations

import math
import os
import random
import sys
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

import requests 
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import delete, select

from app.auth.models import User
from app.borehole.models import Borehole
from app.location.models import Location
from app.sensor.models import Sensor, SensorStatus, WaterLevelReading, FlowReading
from app.pump.models import Pump, PumpHistory, PumpAction, PumpStatus, PumpTrigger
from app.weather.models import Weather
from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — everything tunable lives here
# ─────────────────────────────────────────────────────────────────────────────

# ── Database targets (your freshly created records) ──
LOCATION_ID = 5
BOREHOLE_ID = 5
PUMP_ID = 3
PT_SENSOR_ID = 13  # pressure transducer
FM_SENSOR_ID = 14  # flow meter
ESP32_SENSOR_ID = 15  # controller (owns last_seen for transmissions)

# ── Site ──
LATITUDE = 7.44
LONGITUDE = 3.90
LOCAL_TZ = ZoneInfo("Africa/Lagos")

# ── Window ──
SEED_DAYS = 45
WARMUP_DAYS = 14  # extra rainfall fetched BEFORE the window to spin up wetness index

# ── Borehole geometry / thresholds (must match the DB row) ──
TOTAL_DEPTH_M = 50.0
OPTIMAL_HIGH_M = 45.0  # ambient head can never exceed this
CRITICAL_LOW_M = 10.0  # ESP32 safety cutoff

# ── Pump behaviour ──
PUMP_WINDOW_STARTS = (time(8, 0), time(16, 0))  # local time, daily
RUN_MINUTES_RANGE = (45, 95)  # realistic tank-fill duration, randomised per run
NOMINAL_FLOW_LPM = 60.0  # ~1.0 L/s for a 1 HP submersible
FLOW_SAG_FRACTION = 0.12  # flow droops up to this fraction as lift increases
FLOW_NOISE_LPM = 1.2
MANUAL_RUN_COUNT = 3  # extra manual_override runs sprinkled across the window
MANUAL_RUN_MINUTES = (20, 40)  # shorter, midday-ish

# ── Aquifer physics (the dials that shape the level curve) ──
# Effective storage: litres of water per metre of level change. A bare 6-inch
# casing is only ~18 L/m; gravel pack + local aquifer storage make the
# *effective* value much larger. This sets how fast the level moves.
STORAGE_LPM_PER_M = 120.0
# Inflow: Q_in = K_AQ * (ambient_head - level)  [L/min]. Sets pumping
# equilibrium: level settles at ambient - NOMINAL_FLOW_LPM / K_AQ.
K_AQ = 2.4
# Ambient head (the level everything recovers toward) from rainfall wetness:
#   wetness = antecedent precipitation index (exponential-decay rain memory)
#   ambient = clamp(AMBIENT_BASE_M + AMBIENT_PER_WETNESS * wetness, AMBIENT_MIN_M, OPTIMAL_HIGH_M)
API_DECAY_HOURS = 48.0  # rain memory time constant
AMBIENT_BASE_M = 26.5
AMBIENT_PER_WETNESS = 0.25
AMBIENT_MIN_M = 25.0

# ── Measurement realism ──
LEVEL_NOISE_M = 0.03
TRANSMISSION_DROP_RATE = 0.004  # fraction of readings lost to WiFi gaps
POST_PUMP_WATCH_MIN = 60  # 1-min cadence continues this long after pump-off

# ── Housekeeping ──
RNG_SEED = 42
WIPE_FIRST = (
    True  # delete previous readings/history/weather for THESE ids before inserting
)
CHUNK_SIZE = 2000
DATABASE_URL = settings.database_url

rng = random.Random(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo fetch
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _parse_hours(payload: dict) -> dict[datetime, dict[str, float]]:
    hourly = payload["hourly"]
    out: dict[datetime, dict[str, float]] = {}
    for i, stamp in enumerate(hourly["time"]):
        if hourly["temperature_2m"][i] is None:
            continue  # archive tail returns nulls where data isn't ready yet
        hour = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        out[hour] = {
            "temp": float(hourly["temperature_2m"][i]),
            "humidity": float(hourly["relative_humidity_2m"][i]),
            "precip": hourly["precipitation"][i] or 0.0,
        }
    return out


def fetch_weather(start: date, end: date) -> dict[datetime, dict[str, float]]:
    """Archive for the bulk of history, forecast API's past_days for the recent
    tail the archive hasn't caught up to. Same variable names on both."""
    common = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "UTC",
    }
    resp = requests.get(
        ARCHIVE_URL,
        params={
            **common,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=60,
    )
    resp.raise_for_status()
    hours = _parse_hours(resp.json())

    resp = requests.get(
        FORECAST_URL,
        params={
            **common,
            "past_days": 10,
            "forecast_days": 1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    for hour, vals in _parse_hours(resp.json()).items():
        hours.setdefault(hour, vals)  # archive wins where both exist

    return hours


def _none_to_nan(v):
    return float("nan") if v is None else float(v)


def build_ambient_head(
    weather: dict[datetime, dict], sim_start: datetime
) -> dict[datetime, float]:
    """Ambient aquifer head per UTC hour, from an antecedent-precipitation index.

    Hours before sim_start are warm-up: they feed the index but produce no rows.
    """
    decay = math.exp(-1.0 / API_DECAY_HOURS)
    wetness = 0.0
    ambient: dict[datetime, float] = {}
    for hour in sorted(weather.keys()):
        wetness = wetness * decay + weather[hour]["precip"]
        head = AMBIENT_BASE_M + AMBIENT_PER_WETNESS * wetness
        ambient[hour] = max(AMBIENT_MIN_M, min(OPTIMAL_HIGH_M, head))
    return ambient


# ─────────────────────────────────────────────────────────────────────────────
# Schedule — the single source of truth all three tables derive from
# ─────────────────────────────────────────────────────────────────────────────


def build_runs(
    sim_start: datetime, sim_end: datetime
) -> list[tuple[datetime, datetime, PumpTrigger]]:
    """Every planned pump run in the window: (start_utc, planned_end_utc, trigger).

    Critical-safety cutoffs are NOT planned here — they emerge from the physics.
    """
    runs: list[tuple[datetime, datetime, PumpTrigger]] = []
    day = sim_start.astimezone(LOCAL_TZ).date()
    last_day = sim_end.astimezone(LOCAL_TZ).date()

    all_days: list[date] = []
    while day <= last_day:
        all_days.append(day)
        day += timedelta(days=1)

    for d in all_days:
        for start_local in PUMP_WINDOW_STARTS:
            if rng.random() < 0.10:
                continue  # some days, this window just doesn't happen
            jitter = timedelta(minutes=rng.randint(-90, 90))
            start = (
                datetime.combine(d, start_local, tzinfo=LOCAL_TZ).astimezone(
                    timezone.utc
                )
                + jitter
            )
            duration = rng.randint(*RUN_MINUTES_RANGE)
            runs.append(
                (
                    start,
                    start + timedelta(minutes=duration),
                    PumpTrigger.AUTOMATIC_SCHEDULE,
                )
            )

    # A few manual_override runs on random days, midday-ish
    manual_days = rng.sample(all_days, min(MANUAL_RUN_COUNT, len(all_days)))
    for d in manual_days:
        start_local = time(12, rng.randint(0, 45))
        start = datetime.combine(d, start_local, tzinfo=LOCAL_TZ).astimezone(
            timezone.utc
        )
        duration = rng.randint(*MANUAL_RUN_MINUTES)
        runs.append(
            (start, start + timedelta(minutes=duration), PumpTrigger.MANUAL_OVERRIDE)
        )

    runs.sort(key=lambda r: r[0])
    return [(s, e, t) for (s, e, t) in runs if sim_start <= s < sim_end]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────


def simulate(
    sim_start: datetime, sim_end: datetime, ambient: dict[datetime, float]
) -> dict:
    runs = build_runs(sim_start, sim_end)
    run_idx = 0

    level = ambient[sim_start.replace(minute=0, second=0, microsecond=0)]
    pump_on = False
    scheduled_end: datetime | None = None
    end_trigger = PumpTrigger.AUTOMATIC_SCHEDULE
    watch_until = sim_start - timedelta(minutes=1)

    level_rows: list[WaterLevelReading] = []
    flow_rows: list[FlowReading] = []
    history_rows: list[PumpHistory] = []
    critical_events = 0
    last_transition: datetime | None = None

    t = sim_start
    one_min = timedelta(minutes=1)
    while t < sim_end:
        hour_key = t.replace(minute=0, second=0, microsecond=0)
        amb = ambient[hour_key]

        # 1) Scheduled pump-on
        while run_idx < len(runs) and runs[run_idx][0] <= t:
            start, end, trigger = runs[run_idx]
            run_idx += 1
            if not pump_on:
                pump_on = True
                scheduled_end = end
                end_trigger = trigger
                history_rows.append(
                    PumpHistory(
                        pump_id=PUMP_ID,
                        action=PumpAction.TURNED_ON,
                        triggered_by=trigger,
                        created_at=t,
                    )
                )
                last_transition = t

        # 2) Scheduled shutoff happens BEFORE this minute's flow, so flow rows
        #    fall strictly inside [ON, OFF). Critical shutoff stays after the
        #    physics step below — it reacts to the level this minute produced.
        if pump_on and scheduled_end is not None and t >= scheduled_end:
            pump_on = False
            scheduled_end = None
            history_rows.append(
                PumpHistory(
                    pump_id=PUMP_ID,
                    action=PumpAction.TURNED_OFF,
                    triggered_by=end_trigger,
                    created_at=t,
                )
            )
            last_transition = t
            watch_until = t + timedelta(minutes=POST_PUMP_WATCH_MIN)

        # 3) Flow while pumping
        outflow = 0.0
        if pump_on:
            sag = FLOW_SAG_FRACTION * (OPTIMAL_HIGH_M - level) / OPTIMAL_HIGH_M
            outflow = max(
                0.0, NOMINAL_FLOW_LPM * (1.0 - sag) + rng.gauss(0.0, FLOW_NOISE_LPM)
            )
            if rng.random() > TRANSMISSION_DROP_RATE:
                flow_rows.append(
                    FlowReading(
                        borehole_id=BOREHOLE_ID,
                        sensor_id=FM_SENSOR_ID,
                        abstraction_rate=round(outflow, 3),
                        created_at=t,
                        captured_at=t,
                    )
                )

        # 4) Physics step (one minute)
        inflow = K_AQ * (amb - level)
        level += (inflow - outflow) / STORAGE_LPM_PER_M
        level = max(0.0, min(OPTIMAL_HIGH_M, level))

        # 5) Critical-safety shutoff — reacts to the level this minute produced
        if pump_on and level <= CRITICAL_LOW_M:
            pump_on = False
            scheduled_end = None
            critical_events += 1
            history_rows.append(
                PumpHistory(
                    pump_id=PUMP_ID,
                    action=PumpAction.TURNED_OFF,
                    triggered_by=PumpTrigger.CRITICAL_SAFETY,
                    created_at=t,
                )
            )
            last_transition = t
            watch_until = t + timedelta(minutes=POST_PUMP_WATCH_MIN)

        # 6) Water-level reading per cadence rules
        fast_cadence = pump_on or t <= watch_until
        due = fast_cadence or (t.minute % 30 == 0)
        if due and rng.random() > TRANSMISSION_DROP_RATE:
            measured = max(0.0, level + rng.gauss(0.0, LEVEL_NOISE_M))
            level_rows.append(
                WaterLevelReading(
                    borehole_id=BOREHOLE_ID,
                    sensor_id=PT_SENSOR_ID,
                    water_level=round(measured, 3),
                    created_at=t,
                    captured_at=t,
                )
            )

        t += one_min

    return {
        "levels": level_rows,
        "flows": flow_rows,
        "history": history_rows,
        "critical_events": critical_events,
        "last_transition": last_transition,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def make_engine():
    if not DATABASE_URL:
        sys.exit("Set DATABASE_URL (postgresql+asyncpg://user:pass@host/db)")
    return create_async_engine(DATABASE_URL, echo=False)


async def wipe(session: AsyncSession) -> None:
    await session.exec(
        delete(WaterLevelReading).where(WaterLevelReading.borehole_id == BOREHOLE_ID)
    )
    await session.exec(
        delete(FlowReading).where(FlowReading.borehole_id == BOREHOLE_ID)
    )
    await session.exec(delete(PumpHistory).where(PumpHistory.pump_id == PUMP_ID))
    await session.exec(delete(Weather).where(Weather.location_id == LOCATION_ID))
    await session.commit()


async def insert_chunked(session: AsyncSession, rows: list) -> None:
    for i in range(0, len(rows), CHUNK_SIZE):
        session.add_all(rows[i : i + CHUNK_SIZE])
        await session.commit()


async def main() -> None:
    sim_end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    sim_start = sim_end - timedelta(days=SEED_DAYS)
    fetch_start = sim_start.date() - timedelta(days=WARMUP_DAYS)

    print(
        f"Fetching Open-Meteo archive {fetch_start} → {sim_end.date()} "
        f"({LATITUDE}, {LONGITUDE})..."
    )
    weather = fetch_weather(fetch_start, sim_end.date())
    total_rain = sum(v["precip"] for h, v in weather.items() if h >= sim_start)
    print(f"  {len(weather)} hours fetched; {total_rain:.1f} mm rain inside the window")

    ambient = build_ambient_head(weather, sim_start)

    print(f"Simulating {SEED_DAYS} days minute-by-minute...")
    result = simulate(sim_start, sim_end, ambient)
    last_level = result["levels"][-1].created_at if result["levels"] else None
    last_flow = result["flows"][-1].created_at if result["flows"] else None
    
    weather_rows = [
        Weather(
            location_id=LOCATION_ID,
            temperature=v["temp"],
            humidity=v["humidity"],
            precipitation=v["precip"],
            created_at=h,
        )
        for h, v in sorted(weather.items())
        if sim_start <= h < sim_end and not math.isnan(v["temp"])
    ]

    on_count = sum(1 for r in result["history"] if r.action == PumpAction.TURNED_ON)
    print(f"  water-level readings : {len(result['levels'])}")
    print(f"  flow readings        : {len(result['flows'])}")
    print(
        f"  pump runs            : {on_count} "
        f"({result['critical_events']} ended by critical_safety)"
    )
    print(f"  weather rows         : {len(weather_rows)}")

    engine = make_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        if WIPE_FIRST:
            print("Wiping previous rows for these ids...")
            await wipe(session)

        print("Inserting...")
        await insert_chunked(session, result["levels"])
        await insert_chunked(session, result["flows"])
        await insert_chunked(session, result["history"])
        await insert_chunked(session, weather_rows)

       
        for sensor_id, seen in (
            (PT_SENSOR_ID, last_level),
            (FM_SENSOR_ID, last_flow),
            (ESP32_SENSOR_ID, last_level),
        ):
            sensor = (
                await session.exec(select(Sensor).where(Sensor.id == sensor_id))
            ).first()
            if sensor is not None:
                sensor.status = SensorStatus.ACTIVE
                sensor.last_seen = seen
                session.add(sensor)

        pump = (await session.exec(select(Pump).where(Pump.id == PUMP_ID))).first()
        if pump is not None:
            pump.status = PumpStatus.OFF
            pump.last_status_change = result["last_transition"]
            session.add(pump)

        await session.commit()

    print(
        "Done. The level curve, flow rows, pump history and weather all "
        "derive from one schedule and one rainfall record — they cannot disagree."
    )


if __name__ == "__main__":
    asyncio.run(main())
