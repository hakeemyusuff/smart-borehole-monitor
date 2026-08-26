from __future__ import annotations

import asyncio
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from app.auth.models import User  # noqa: F401  (metadata registration)
from app.borehole.models import Borehole  # noqa: F401
from app.location.models import Location  # noqa: F401
from app.sensor.models import WaterLevelReading, FlowReading
from app.weather.models import Weather
from app.core.config import settings
from app.ml.features import API_DECAY_HOURS, LEVEL_MATCH_TOLERANCE

# ── Config ──────────────────────────────────────────────────────────────────
BOREHOLE_ID = 5
LOCATION_ID = 5
HORIZON_HOURS = 24
OUTPUT_PATH = "data/training_table.csv"

# ── Pure transformation (testable without a database) ───────────────────────


def build_training_table(
    levels: pd.DataFrame,  # columns: created_at (tz-aware UTC), water_level
    flows: pd.DataFrame,  # columns: created_at, abstraction_rate (L/min, one row ≈ one minute)
    weather: pd.DataFrame,  # columns: created_at (hourly), precipitation
) -> pd.DataFrame:
    if levels.empty:
        raise ValueError(
            "No water-level readings for this borehole — nothing to build a training table from."
        )
    if weather.empty:
        raise ValueError(
            "No weather rows for this location — rain features cannot be computed."
        )

    levels = levels.sort_values("created_at").reset_index(drop=True)
    flows = flows.sort_values("created_at").reset_index(drop=True)
    weather = weather.sort_values("created_at").reset_index(drop=True)

    # Hourly grid: first full hour with 6h of history behind it, last hour
    # with a full horizon of future ahead of it.
    t0 = levels["created_at"].min().ceil("h") + pd.Timedelta(hours=6)
    t1 = levels["created_at"].max().floor("h") - pd.Timedelta(hours=HORIZON_HOURS)
    if t1 <= t0:
        raise ValueError("Not enough history to build even one training row.")
    grid = pd.DataFrame({"t": pd.date_range(t0, t1, freq="h")})

    # Level at arbitrary times via nearest-reading lookup
    lv = levels.rename(columns={"created_at": "ts"})

    def level_at(times: pd.Series, colname: str) -> pd.DataFrame:
        probe = pd.DataFrame({"ts": times}).sort_values("ts")
        merged = pd.merge_asof(
            probe,
            lv,
            on="ts",
            direction="nearest",
            tolerance=LEVEL_MATCH_TOLERANCE,
        )
        return merged.rename(columns={"water_level": colname})[["ts", colname]]

    out = grid.copy()
    out["level_now"] = level_at(grid["t"], "v")["v"].values
    for h, name in ((1, "lag_1h"), (3, "lag_3h"), (6, "lag_6h")):
        out[name] = level_at(grid["t"] - pd.Timedelta(hours=h), "v")["v"].values
    out["delta_1h"] = out["level_now"] - out["lag_1h"]
    out["delta_3h"] = out["level_now"] - out["lag_3h"]
    out["delta_6h"] = out["level_now"] - out["lag_6h"]
    out = out.drop(columns=["lag_1h", "lag_3h", "lag_6h"])

    # Pumping features from flow rows (each row ≈ 1 minute at abstraction_rate L/min)
    ft = flows["created_at"].values
    fv = flows["abstraction_rate"].to_numpy(dtype=float)
    cum_vol = np.concatenate(([0.0], np.cumsum(fv)))  # litres, since L/min × 1 min

    def flow_window(ts: pd.Series, hours: float) -> tuple[np.ndarray, np.ndarray]:
        end = np.searchsorted(ft, ts.values, side="right")
        start = np.searchsorted(
            ft, (ts - pd.Timedelta(hours=hours)).values, side="right"
        )
        return (end - start).astype(float), cum_vol[end] - cum_vol[start]

    out["pumped_min_1h"], _ = flow_window(grid["t"], 1.0)
    _, out["vol_24h_l"] = flow_window(grid["t"], 24.0)

    # Rain features from hourly weather
    wt = weather["created_at"].values
    wp = weather["precipitation"].fillna(0.0).to_numpy(dtype=float)
    cum_rain = np.concatenate(([0.0], np.cumsum(wp)))

    def rain_window(ts: pd.Series, hours: float) -> np.ndarray:
        end = np.searchsorted(wt, ts.values, side="right")
        start = np.searchsorted(
            wt, (ts - pd.Timedelta(hours=hours)).values, side="right"
        )
        return cum_rain[end] - cum_rain[start]

    out["rain_1h"] = rain_window(grid["t"], 1.0)
    out["rain_24h"] = rain_window(grid["t"], 24.0)
    out["rain_72h"] = rain_window(grid["t"], 72.0)

    # Wetness index: same exponential rain-memory the recharge physics uses
    decay = math.exp(-1.0 / API_DECAY_HOURS)
    wetness_vals = np.empty(len(weather))
    w = 0.0
    for i, p in enumerate(wp):
        w = w * decay + p
        wetness_vals[i] = w
    wet = pd.DataFrame({"ts": weather["created_at"], "wetness": wetness_vals})
    out["wetness"] = pd.merge_asof(
        grid.rename(columns={"t": "ts"}),
        wet,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(hours=6),
    )["wetness"].values

    out["hour_of_day"] = grid["t"].dt.hour

    # Targets: strictly future
    for k in range(1, HORIZON_HOURS + 1):
        out[f"y_{k}"] = level_at(grid["t"] + pd.Timedelta(hours=k), "v")["v"].values

    before = len(out)
    na_counts = out.isna().sum()
    print("NaNs per column before dropna:")
    print(na_counts[na_counts > 0].to_string())
    out = out.dropna().reset_index(drop=True)
    out.attrs["dropped_rows"] = before - len(out)
    return out          # <- must be here, inside the function, and last


# ── DB I/O ──────────────────────────────────────────────────────────────────


async def fetch_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = create_async_engine(settings.database_url, echo=False)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        lv = (
            await session.exec(
                select(
                    WaterLevelReading.captured_at, WaterLevelReading.water_level
                ).where(WaterLevelReading.borehole_id == BOREHOLE_ID)
            )
        ).all()
        fl = (
            await session.exec(
                select(FlowReading.captured_at, FlowReading.abstraction_rate).where(
                    FlowReading.borehole_id == BOREHOLE_ID
                )
            )
        ).all()
        wx = (
            await session.exec(
                select(Weather.created_at, Weather.precipitation).where(
                    Weather.location_id == LOCATION_ID
                )
            )
        ).all()

    levels = pd.DataFrame(lv, columns=["created_at", "water_level"])
    flows = pd.DataFrame(fl, columns=["created_at", "abstraction_rate"])
    weather = pd.DataFrame(wx, columns=["created_at", "precipitation"])
    for df in (levels, flows, weather):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return levels, flows, weather


async def main() -> None:
    levels, flows, weather = await fetch_frames()
    print(
        f"Loaded: {len(levels)} level rows, {len(flows)} flow rows, "
        f"{len(weather)} weather rows"
    )

    try:
        table = build_training_table(levels, flows, weather)
    except ValueError as e:
        print(f"Cannot build training table: {e}")
        return

    print(
        f"Training table: {len(table)} rows × {len(table.columns)} cols "
        f"({table.attrs.get('dropped_rows', 0)} dropped for gaps/edges)"
    )

    # Sanity diagnostic: the relationship we KNOW is in the data. Wetness sets
    # the ambient head the level recovers toward, so FUTURE LEVEL should track
    # wetness. (Don't test wetness vs *change* — that's confounded: when the
    # level is already high in wet periods, the change is ~zero.)
    corr = float(np.corrcoef(table["wetness"], table["y_24"])[0, 1])
    print(
        f"Sanity check — corr(wetness, level 24h out): {corr:+.3f} "
        f"({'plausible' if corr > 0.15 else 'SUSPICIOUS — investigate before training'})"
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
