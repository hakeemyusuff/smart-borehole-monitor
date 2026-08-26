from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from app.ml.features import compute_feature_row, FEATURE_ORDER, InsufficientData
from scripts.build_training_table import fetch_frames, BOREHOLE_ID, LOCATION_ID


async def main() -> None:
    # Same data the builder used
    levels, flows, weather = await fetch_frames()
    print("levels cols:", list(levels.columns))
    print("flows cols:", list(flows.columns))
    print("weather cols:", list(weather.columns))

    # The training table the builder already produced — our ground truth
    table = pd.read_csv("data/training_table.csv")
    table["t"] = pd.to_datetime(table["t"], utc=True)

    checked = 0
    skipped = 0
    mismatches = 0

    for _, ref in table.iterrows():
        now = ref["t"]
        try:
            row = compute_feature_row(levels, flows, weather, now)
        except InsufficientData as e:
            # The builder produced this row, so single-row SHOULD succeed too.
            # If it doesn't, that's itself a discrepancy worth seeing.
            print(f"  UNEXPECTED InsufficientData @ {now}: {e}")
            skipped += 1
            continue

        checked += 1
        for col in FEATURE_ORDER:
            single = row[col]
            batch = float(ref[col])
            if abs(single - batch) > 1e-9:
                mismatches += 1
                if mismatches <= 10:
                    print(
                        f"  MISMATCH @ {now} {col}: "
                        f"single={single} batch={batch} diff={single - batch}"
                    )

    print(
        f"\nchecked {checked} rows, {skipped} unexpected-skips, {mismatches} mismatches"
    )
    if mismatches == 0 and skipped == 0:
        print(
            "PARITY PROVEN — compute_feature_row matches build_training_table on real seeded data."
        )
    else:
        print("PARITY FAILED — investigate before wiring inference.")


if __name__ == "__main__":
    asyncio.run(main())
