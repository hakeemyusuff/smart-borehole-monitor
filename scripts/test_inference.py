"""Offline smoke test of the shipped artifact; never connects to the database."""

import json
from pathlib import Path
import pandas as pd
from app.ml.level_forecast import LevelModel, compute_level_features


def main():
    model = LevelModel.load(Path("models/level_change_linear.json"))
    levels = pd.read_csv("/mnt/c/Users/Ysf/Downloads/water_level_reading.csv")
    meta = model.metadata
    levels = levels.loc[
        (levels.borehole_id == meta["borehole_id"])
        & (levels.sensor_id == meta["sensor_id"])
    ]
    example = pd.read_csv("analysis/level_only_2026-09-22/examples.csv").iloc[-1]
    now = pd.Timestamp(example.forecast_at)
    features, captured = compute_level_features(levels, now)
    print(
        json.dumps(
            dict(
                model_version=meta["model_version"],
                issued_at=str(now),
                predicted_for=str(now + pd.Timedelta(hours=2)),
                predicted_level_2h=model.predict(features),
                input_captured_at=str(captured),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
