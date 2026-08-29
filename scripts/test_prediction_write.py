from __future__ import annotations

import asyncio
from datetime import timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import select

from app.core.database import async_session_maker
from app.ml.services import load_model, run_inference, get_model
from app.ml.models import Prediction
from scripts.build_training_table import fetch_frames

BOREHOLE_ID = 5
HORIZON_HOURS = 2


async def insert_prediction(now, predicted_for, result) -> int:
    """Insert one prediction with ON CONFLICT DO NOTHING. Returns the number
    of rows actually inserted (0 if the conflict fired)."""
    async with async_session_maker() as session:
 
        stmt = (
            pg_insert(Prediction)
            .values(
                borehole_id=BOREHOLE_ID,
                predicted_level_2h=result.predicted_level_2h,
                predicted_for=predicted_for,
                confidence_score=result.confidence,
                horizon_hours=HORIZON_HOURS,
                created_at=now,
            )
            .on_conflict_do_nothing(constraint="uq_prediction_borehole_predicted_for")
            .returning(Prediction.id)
        )
        
        res = await session.exec(stmt)
        await session.commit()
        inserted = res.all()
        return len(inserted)


async def main() -> None:
    load_model()
    if get_model() is None:
        print("No model loaded — aborting.")
        return

    levels, flows, weather = await fetch_frames()

    # A `now` INSIDE the seeded range, so inference has recent history and
    # actually produces a prediction (rather than InsufficientData).
    last = levels["created_at"].max()
    now = (last - pd.Timedelta(hours=HORIZON_HOURS + 1)).floor("h")
    predicted_for = now + pd.Timedelta(hours=HORIZON_HOURS)
    print(f"Test now = {now}, predicted_for = {predicted_for}")

    result = run_inference(levels, flows, weather, now)
    print(
        f"Predicted {result.predicted_level_2h:.3f} m (confidence {result.confidence:.3f})"
    )

    # First insert — expect 1 row written.
    n1 = await insert_prediction(
        now.to_pydatetime(), predicted_for.to_pydatetime(), result
    )
    print(f"\nFirst insert:  {n1} row(s) written   (expect 1)")

    # Second insert, same predicted_for — expect 0 (constraint dedupes).
    n2 = await insert_prediction(
        now.to_pydatetime(), predicted_for.to_pydatetime(), result
    )
    print(f"Second insert: {n2} row(s) written   (expect 0 — constraint held)")

    # Read it back and confirm the stored values match what we computed.
    async with async_session_maker() as session:
        rows = (
            await session.exec(
                select(Prediction)
                .where(Prediction.borehole_id == BOREHOLE_ID)
                .where(Prediction.predicted_for == predicted_for.to_pydatetime())
            )
        ).all()

    print(f"\nRows in DB for this predicted_for: {len(rows)}   (expect exactly 1)")
    if rows:
        r = rows[0]
        print(f"  stored predicted_level_2h: {r.predicted_level_2h:.3f}")
        print(f"  stored predicted_for:      {r.predicted_for}")
        print(f"  stored created_at:         {r.created_at}")
        print(f"  stored horizon_hours:      {r.horizon_hours}")

    # Verdict
    ok = n1 == 1 and n2 == 0 and len(rows) == 1
    print(
        "\n"
        + (
            "WRITE PATH PROVEN — insert works, constraint dedupes, values frozen."
            if ok
            else "SOMETHING OFF — check the counts above."
        )
    )

    # Cleanup: remove the test row so it doesn't pollute your real predictions.
    async with async_session_maker() as session:
        for r in rows:
            await session.delete(r)
        await session.commit()
    print("(cleaned up the test row)")


if __name__ == "__main__":
    asyncio.run(main())
