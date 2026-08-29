from __future__ import annotations

import asyncio

import pandas as pd

from app.ml.services import load_model, run_inference, get_model
from scripts.build_training_table import fetch_frames


async def main() -> None:
    load_model()
    if get_model() is None:
        print("No model loaded — aborting.")
        return

    levels, flows, weather = await fetch_frames()

    # Pick a `now` well inside the seeded range: 24h before the last level
    # reading, floored to the hour (so a full predicted_level_2h of "future" exists to eyeball).
    last = levels["created_at"].max()
    now = (last - pd.Timedelta(hours=2)).floor("h")
    print(f"Running inference for now = {now}")

    result = run_inference(levels, flows, weather, now)

    print(f"\npredicted_at:      {result.predicted_at}")
    print(f"confidence:        {result.confidence:.4f}")
    print(f"predicted_level_2h: {result.predicted_level_2h:.3f}")

    cur = levels.sort_values("created_at").iloc[-3]["water_level"]  # a level near `now`
    print(f"(for reference, a level near `now`: {cur:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
