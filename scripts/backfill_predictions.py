from __future__ import annotations

import asyncio
from datetime import timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.database import async_session_maker
from app.ml.features import InsufficientData
from app.ml.services import load_model, run_inference, get_model
from app.ml.models import Prediction
from scripts.build_training_table import fetch_frames

BOREHOLE_ID = 5
HORIZON_HOURS = 2
BACKFILL_DAYS = 5


async def main() -> None:
    load_model()
    if get_model() is None:
        print("No model loaded — aborting.")
        return

    levels, flows, weather = await fetch_frames()
    if levels.empty:
        print("No level data — aborting.")
        return

    # Hourly grid over the last BACKFILL_DAYS of seeded history, but stop
    # HORIZON_HOURS before the last reading so each prediction's target time
    # still falls inside the data (so there's an actual to compare against).
    last = levels["created_at"].max()
    end = (last - pd.Timedelta(hours=HORIZON_HOURS)).floor("h")
    start = end - pd.Timedelta(days=BACKFILL_DAYS)
    grid = pd.date_range(start, end, freq="h")

    print(f"Backfilling {len(grid)} hourly predictions: {start} → {end}")

    written = 0
    skipped = 0
    for now in grid:
        try:
            result = run_inference(levels, flows, weather, now)
        except InsufficientData:
            skipped += 1
            continue

        predicted_for = now + pd.Timedelta(hours=HORIZON_HOURS)
        async with async_session_maker() as session:
            stmt = (
                pg_insert(Prediction)
                .values(
                    borehole_id=BOREHOLE_ID,
                    predicted_level_2h=result.predicted_level_2h,
                    predicted_for=predicted_for.to_pydatetime(),
                    confidence_score=result.confidence,
                    horizon_hours=HORIZON_HOURS,
                    created_at=now.to_pydatetime(),
                )
                .on_conflict_do_nothing(
                    constraint="uq_prediction_borehole_predicted_for"
                )
                .returning(Prediction.id)
            )
            res = await session.exec(stmt)
            await session.commit()
            if res.all():
                written += 1

    print(f"\nDone. {written} written, {skipped} skipped (insufficient history).")
    print("Note: these are RETROSPECTIVE predictions computed now for past target")
    print("times — a development/evaluation aid, not live pre-arrival predictions.")


if __name__ == "__main__":
    asyncio.run(main())
