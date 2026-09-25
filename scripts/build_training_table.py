from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.sensor.models import WaterLevelReading


# The well and sensor the model is being developed for
BOREHOLE_ID = 2
SENSOR_ID = 4

HORIZON_HOURS = 2
MATCH_TOLERANCE = pd.Timedelta(minutes=35)

OUTPUT_PATH = Path("data/level_training_table.csv")

async def fetch_levels() -> pd.DataFrame:
    """Read real water-level measurements without changing database"""
    
    engine = create_async_engine(
        settings.database_url,
        echo=False,
    )

    try:
        async with AsyncSession(engine) as session:
            statement = (
                select(
                    WaterLevelReading.captured_at,
                    WaterLevelReading.created_at,
                    WaterLevelReading.water_level,
                )
                .where(
                    WaterLevelReading.borehole_id == BOREHOLE_ID,
                    WaterLevelReading.sensor_id == SENSOR_ID,
                )
                .order_by(WaterLevelReading.captured_at) #type: ignore
            )
            
            result = await session.exec(statement)
            rows = result.all()
            
        return pd.DataFrame(
            rows,
            columns=["captured_at", "created_at", "water_level"],
        )
        
    finally:
        await engine.dispose()


def build_training_table(levels: pd.DataFrame) -> pd.DataFrame:
    """Create hourly examples using past inputs and a future target."""

    if levels.empty:
        raise ValueError("No water-level readings found.")

    levels = levels.copy()

    for column in ["captured_at", "created_at"]:
        levels[column] = pd.to_datetime(
            levels[column],
            utc=True,
            format="mixed",
            errors="raise",
        )

    levels["water_level"] = pd.to_numeric(
        levels["water_level"],
        errors="raise",
    )

    if levels.isna().any().any():
        raise ValueError("The readings contain missing values.")

    if not np.isfinite(levels["water_level"]).all():
        raise ValueError("The readings contains invalid numerical values")

    if (levels["water_level"] < 0).any():
        raise ValueError("The readings contains negative water levels.")

    if levels["captured_at"].duplicated().any():
        raise ValueError("Duplicate measurement timestamps were found.")

    if (levels["created_at"] < levels["captured_at"]).any():
        raise ValueError("Some readings arrived before their capture time.")

    first_hour = levels["captured_at"].min().ceil("h") + pd.Timedelta(hours=6)

    last_hour = levels["captured_at"].max().floor("h") - pd.Timedelta(
        hours=HORIZON_HOURS
    )

    if last_hour < first_hour:
        raise ValueError("Not enough history to build a training example.")

    forecast_times = pd.date_range(
        first_hour,
        last_hour,
        freq="h",
    )

    examples = []

    for forecast_at in forecast_times:
        selected_levels = {}

        # Build inputs using only information available at forecast_at.
        for lag_hours in [0, 1, 3, 6]:
            anchor = forecast_at - pd.Timedelta(hours=lag_hours)

            candidates = levels.loc[
                (levels["captured_at"] <= anchor)
                & (levels["captured_at"] >= anchor - MATCH_TOLERANCE)
                & (levels["created_at"] <= forecast_at)
            ]

            if candidates.empty:
                break

            selected_levels[lag_hours] = float(candidates.iloc[-1]["water_level"])

        # An example needs all four input readings.
        if len(selected_levels) != 4:
            continue

        target_at = forecast_at + pd.Timedelta(hours=HORIZON_HOURS)

        # The future observation is the answer, never an input.
        distances = (levels["captured_at"] - target_at).abs()
        target_index = distances.idxmin()

        if distances.loc[target_index] > MATCH_TOLERANCE:
            continue

        target = levels.loc[target_index]
        level_now = selected_levels[0]

        examples.append(
            {
                "forecast_at": forecast_at,
                "level_now": level_now,
                "delta_1h": level_now - selected_levels[1],
                "delta_3h": level_now - selected_levels[3],
                "delta_6h": level_now - selected_levels[6],
                "target_at": target_at,
                "target_captured_at": target["captured_at"],
                "target_arrived_at": target["created_at"],
                "level_2h": float(target["water_level"]),
            }
        )

    if not examples:
        raise ValueError(
            "No complete examples were found. "
            "Check the measurement coverage and gaps."
        )

    table = pd.DataFrame(examples)

    # Preserve the complete time-grid boundaries for chronological splitting.
    table.attrs["first_forecast_at"] = first_hour
    table.attrs["last_forecast_at"] = last_hour

    print(f"Hourly opportunities: {len(forecast_times)}")
    print(f"Complete examples:   {len(table)}")
    print(f"Skipped for gaps:    {len(forecast_times) - len(table)}")

    return table


async def main() -> None:
    # Refuse to overwrite an earlier training snapshot.
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Choose a new OUTPUT_PATH to create another snapshot."
        )

    levels = await fetch_levels()
    print(f"Loaded {len(levels)} water-level readings.")

    table = build_training_table(levels)

    # Repeated metadata columns survive saving/loading the CSV.
    # These are NOT model inputs.
    table["grid_start"] = table.attrs["first_forecast_at"]
    table["grid_end"] = table.attrs["last_forecast_at"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x") as file:
        table.to_csv(file, index=False)

    print(f"Saved training table to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
