from __future__ import annotations

import math

import numpy as np
import pandas as pd

API_DECAY_HOURS = 48.0
LEVEL_MATCH_TOLERANCE = pd.Timedelta(minutes=35)
WETNESS_TOLERANCE = pd.Timedelta(hours=6)

FEATURE_ORDER = [
    "level_now",
    "delta_1h",
    "delta_3h",
    "delta_6h",
    "pumped_min_1h",
    "vol_24h_l",
    "rain_1h",
    "rain_24h",
    "rain_72h",
    "wetness",
    "hour_of_day",
]


class InsufficientData(Exception):
    """Raised when there isn't enough history to build a feature row"""


def compute_feature_row(
    levels: pd.DataFrame,
    flows: pd.DataFrame,
    weather: pd.DataFrame,
    now: pd.Timestamp,
) -> dict[str, float]:
    """Compute one feature vector at `now`, identical to the value
    build_training_table would produce for that same hour. Raises InsufficientData
    rather than returning NaN so callers can log and skip.
    """
    if levels.empty:
        raise InsufficientData("no level readings")
    if weather.empty:
        raise InsufficientData("no weather rows")

    levels = levels.sort_values("created_at").reset_index(drop=True)
    flows = flows.sort_values("created_at").reset_index(drop=True)
    weather = weather.sort_values("created_at").reset_index(drop=True)

    lv = levels.rename(columns={"created_at": "ts"})

    def level_at(t: pd.Timestamp) -> float:
        probe = pd.DataFrame({"ts": [t]})
        merged = pd.merge_asof(
            probe,
            lv,
            on="ts",
            direction="nearest",
            tolerance=LEVEL_MATCH_TOLERANCE,
        )
        return merged["water_level"].iloc[0]

    level_now = level_at(now)
    lag_1h = level_at(now - pd.Timedelta(hours=1))
    lag_3h = level_at(now - pd.Timedelta(hours=3))
    lag_6h = level_at(now - pd.Timedelta(hours=6))
    if any(pd.isna(x) for x in (level_now, lag_1h, lag_3h, lag_6h)):
        raise InsufficientData("missing level history within tolerance")

    # -- Pumping features from flow rows (each row ~ one minute at L/min) ----
    ft = flows["created_at"]
    fv = flows["abstraction_rate"].to_numpy(dtype=float)
    cum_vol = np.concatenate(([0.0], np.cumsum(fv)))

    def flow_window(hours: float) -> tuple[float, float]:
        end = int(ft.searchsorted(now, side="right"))
        start = int(ft.searchsorted(now - pd.Timedelta(hours=hours), side="right"))
        return float(end - start), float(cum_vol[end] - cum_vol[start])

    pumped_min_1h, _ = flow_window(1.0)
    _, vol_24h_l = flow_window(24.0)

    # -- Rain features from hourly weather -----------------------
    wt = weather["created_at"]
    wp = weather["precipitation"].fillna(0.0).to_numpy(dtype=float)
    cum_rain = np.concatenate(([0.0], np.cumsum(wp)))

    def rain_window(hours: float) -> float:
        end = int(wt.searchsorted(now, side="right"))
        start = int(wt.searchsorted(now - pd.Timedelta(hours=hours), side="right"))
        return float(cum_rain[end] - cum_rain[start])

    rain_1h = rain_window(1.0)
    rain_24h = rain_window(24.0)
    rain_72h = rain_window(72.0)

    # --- Wetness: exponential rain memory, value as of `now` ---------------
    decay = math.exp(-1.0 / API_DECAY_HOURS)
    w = 0.0
    wetness_now = np.nan
    last_ts = None
    for ts, p in zip(weather["created_at"], wp):
        w = w * decay + p
        if ts <= now:
            wetness_now = w
            last_ts = ts

    if last_ts is None or (now - last_ts) > WETNESS_TOLERANCE:
        raise InsufficientData("no weather within wetness tolerance")

    return {
        "level_now": float(level_now),
        "delta_1h": float(level_now - lag_1h),
        "delta_3h": float(level_now - lag_3h),
        "delta_6h": float(level_now - lag_6h),
        "pumped_min_1h": pumped_min_1h,
        "vol_24h_l": vol_24h_l,
        "rain_1h": rain_1h,
        "rain_24h": rain_24h,
        "rain_72h": rain_72h,
        "wetness": float(wetness_now),
        "hour_of_day": float(now.hour),
    }
