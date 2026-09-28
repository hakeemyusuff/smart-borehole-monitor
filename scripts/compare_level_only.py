"""Offline experiment. Reads CSV bytes only; never imports the app or database.

Run with Python -B to avoid bytecode files. Outputs require a NEW directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

FEATURES = ["level_now", "delta_1h", "delta_3h", "delta_6h"]
GRID = {"n_estimators": [100, 300], "max_depth": [3, 6, None],
        "min_samples_leaf": [1, 3, 8]}
TOLERANCE = pd.Timedelta(minutes=35)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_examples(raw: pd.DataFrame, borehole_id: int, sensor_id: int):
    data = raw.loc[(raw.borehole_id == borehole_id) & (raw.sensor_id == sensor_id)].copy()
    if data.empty:
        raise ValueError("No readings for selected borehole and sensor")
    for col in ["captured_at", "created_at"]:
        data[col] = pd.to_datetime(data[col], utc=True, format="mixed", errors="raise")
    data["water_level"] = pd.to_numeric(data.water_level, errors="raise")
    if data[["captured_at", "created_at", "water_level"]].isna().any().any():
        raise ValueError("Missing required values: inspect source; no automatic repair")
    if not np.isfinite(data.water_level).all() or (data.water_level < 0).any():
        raise ValueError("Invalid water level: inspect source; no automatic repair")
    if data.duplicated(["sensor_id", "captured_at"]).any():
        raise ValueError("Duplicate capture timestamps: inspect source")
    if (data.created_at < data.captured_at).any():
        raise ValueError("Arrival precedes capture: inspect clock discrepancy")
    data = data.sort_values("captured_at")
    times = pd.date_range(data.captured_at.min().ceil("h") + pd.Timedelta(hours=6),
                          data.captured_at.max().floor("h") - pd.Timedelta(hours=2), freq="h")
    accepted, rejected = [], []
    for t in times:
        row = {"forecast_at": t, "target_at": t + pd.Timedelta(hours=2)}
        reasons = []
        for lag in [0, 1, 3, 6]:
            anchor = t - pd.Timedelta(hours=lag)
            candidates = data.loc[(data.captured_at <= anchor)
                                  & (data.captured_at >= anchor - TOLERANCE)
                                  & (data.created_at <= t)]
            if candidates.empty:
                reasons.append(f"missing_available_level_{lag}h")
                continue
            reading = candidates.iloc[-1]
            row[f"level_{lag}h"] = float(reading.water_level)
            row[f"input_{lag}h_captured_at"] = reading.captured_at
            row[f"input_{lag}h_arrived_at"] = reading.created_at
        distance = (data.captured_at - row["target_at"]).abs()
        # Sorted input resolves an equidistant tie to the earlier measurement.
        outcome = data.loc[distance.idxmin()]
        if abs(outcome.captured_at - row["target_at"]) > TOLERANCE:
            reasons.append("missing_target")
        else:
            row.update(actual=float(outcome.water_level), target_captured_at=outcome.captured_at,
                       target_arrived_at=outcome.created_at,
                       target_offset_minutes=(outcome.captured_at-row["target_at"]).total_seconds()/60)
        if reasons:
            rejected.append({"forecast_at": t, "reasons": ";".join(reasons)})
        else:
            row["level_now"] = row.pop("level_0h")
            for lag in [1, 3, 6]:
                row[f"delta_{lag}h"] = row["level_now"] - row.pop(f"level_{lag}h")
            accepted.append(row)
    return pd.DataFrame(accepted), pd.DataFrame(rejected, columns=["forecast_at", "reasons"]), times


def available_before(table, cutoff):
    """Require both nominal target and actual label availability before cutoff."""
    return ((table.forecast_at < cutoff) & (table.target_at < cutoff)
            & (table.target_captured_at < cutoff) & (table.target_arrived_at < cutoff))


def chronological_splits(table, start, stop):
    # Equal elapsed-time validation windows, despite gaps among accepted rows.
    edges = pd.date_range(start, stop, periods=5)
    splits, descriptions = [], []
    for fold in range(1, 4):
        left, right = edges[fold], edges[fold + 1]
        train = np.flatnonzero(available_before(table, left).to_numpy())
        valid = np.flatnonzero(((table.forecast_at >= left) & (table.forecast_at < right)).to_numpy())
        if len(train) < 20 or len(valid) < 5:
            raise ValueError("Insufficient rows for planned CV; inspect coverage instead of silently changing splits")
        splits.append((train, valid))
        descriptions.append({"fold": fold, "train_rows": len(train), "validation_rows": len(valid),
                             "validation_start": str(left), "validation_end_exclusive": str(right)})
    return splits, descriptions


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Metrics require matching one-dimensional arrays")
    return {"mae_m": float(mean_absolute_error(actual, predicted)),
            "rmse_m": float(np.sqrt(mean_squared_error(actual, predicted)))}


def run(source: Path, output: Path, protected: list[Path], borehole_id=2, sensor_id=4):
    paths = list(dict.fromkeys([source.resolve(), *(p.resolve() for p in protected)]))
    before = {str(p): digest(p) for p in paths}
    if output.exists():
        raise ValueError("Output directory already exists; use a new directory to preserve previous results")
    raw = pd.read_csv(io.BytesIO(source.read_bytes()))
    table, exclusions, grid = build_examples(raw, borehole_id, sensor_id)
    if len(table) < 50 or len(grid) < 10:
        raise ValueError("Insufficient examples for the planned experiment")
    # Reserve last 20% of hourly forecast opportunities by elapsed calendar time.
    cutoff = grid[int(len(grid) * .8)]
    train_mask = available_before(table, cutoff)
    test_mask = table.forecast_at >= cutoff
    development = table.loc[train_mask].reset_index(drop=True)
    test = table.loc[test_mask].reset_index(drop=True)
    if len(test) < 10:
        raise ValueError("Insufficient final test examples")
    splits, fold_info = chronological_splits(development, grid[0], cutoff)
    search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), GRID,
                          scoring="neg_mean_absolute_error", cv=splits, n_jobs=1,
                          refit=True, error_score="raise", return_train_score=False)
    search.fit(development[FEATURES], development.actual)
    results = pd.DataFrame(search.cv_results_)
    results["mean_validation_mae_m"] = -results.mean_test_score
    baseline_folds = []
    for _, valid in splits:
        part = development.iloc[valid]
        baseline_folds.append(metrics(part.actual, part.level_now)["mae_m"])
    # Test accessed for scoring only after parameter selection/refit is complete.
    test["persistence"] = test.level_now
    test["random_forest"] = search.predict(test[FEATURES])
    scores = {name: metrics(test.actual, test[name]) for name in ["persistence", "random_forest"]}
    for name in scores:
        test[f"{name}_absolute_error_m"] = (test.actual-test[name]).abs()
    table["partition"] = np.where(test_mask, "test", np.where(train_mask, "development", "boundary_excluded"))
    after = {str(p): digest(p) for p in paths}
    if before != after:
        raise RuntimeError("Source fingerprints changed during experiment")
    summary = {"input_sha256_before": before, "input_sha256_after": after,
               "inputs_unchanged": before == after, "borehole_id": borehole_id, "sensor_id": sensor_id,
               "source_rows": len(raw), "hourly_opportunities": len(grid), "accepted_rows": len(table),
               "excluded_rows": len(exclusions), "development_rows": len(development), "test_rows": len(test),
               "boundary_excluded_rows": int((~train_mask & ~test_mask).sum()),
               "test_cutoff_utc": str(cutoff), "test_first_forecast": str(test.forecast_at.min()),
               "test_last_forecast": str(test.forecast_at.max()), "features": FEATURES, "parameter_grid": GRID,
               "best_params": search.best_params_, "cv_best_mean_mae_m": -search.best_score_,
               "cv_persistence_fold_mae_m": baseline_folds, "folds": fold_info, "test_scores": scores,
               "test_target_max_offset_minutes": float(test.target_offset_minutes.abs().max()),
               "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                            "numpy": np.__version__, "sklearn": sklearn.__version__}}
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "examples.csv", index=False)
    exclusions.to_csv(output / "excluded_hours.csv", index=False)
    results.to_csv(output / "grid_search.csv", index=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Water-level-only experiment

Real CSV observations; no synthetic training data. No model deployed or saved to the application.
All {len(paths)} protected source CSV fingerprints match before and after execution.

## Method

Predict height above the fixed sensor approximately two hours ahead, hourly in UTC.
Inputs: current level and changes over 1, 3 and 6 hours. Each input is the last
capture at or before its anchor, within 35 minutes, and must have arrived by forecast time.
Targets are nearest observed levels within ±35 minutes of the two-hour target;
this is approximate target matching, not measurement at exactly two hours.
No interpolation, imputation, scaling, or source repairs. Rejected hours are recorded.

Last 20% of the original hourly time grid reserved for testing, starting {cutoff}.
Earlier examples whose target or target arrival reaches that boundary are excluded.
Three expanding CV folds use equal elapsed-time validation windows inside development.
At every fold boundary, training labels must already have been captured AND received.
GridSearchCV tests 18 combinations (54 fits), minimizing equally weighted fold MAE.
Grid: trees 100/300; depth 3/6/unlimited; minimum leaf samples 1/3/8.
Tree depth and leaf size explore regularization; tree count compares ensemble size.
These are predefined practical choices, not an exhaustive search or proof of optimality.
All other forest parameters retain installed scikit-learn defaults; random seed 42.
The chosen forest is refit on eligible development examples before one final test.

## Results

- Hourly opportunities: {len(grid)}; accepted: {len(table)}; rejected: {len(exclusions)}.
- Development: {len(development)}; test: {len(test)}; boundary exclusions: {summary['boundary_excluded_rows']}.
- Best settings: `{search.best_params_}`.
- Best mean CV MAE: {-search.best_score_:.5f} m.
- Persistence mean CV MAE: {np.mean(baseline_folds):.5f} m.

| Final test model | MAE (m) | RMSE (m) |
|---|---:|---:|
| Persistence | {scores['persistence']['mae_m']:.5f} | {scores['persistence']['rmse_m']:.5f} |
| Tuned Random Forest | {scores['random_forest']['mae_m']:.5f} | {scores['random_forest']['rmse_m']:.5f} |

## Interpretation limits

One well, a short observation period, and correlated hourly examples. Excluded
outages are not scored, so these errors describe eligible forecast times, not continuous
service availability. Historical replay accounts for recorded arrival times; it is not
a prospective live trial. Target tolerances blur rapid changes. Fold variability and
one held-out period do not establish generalization across seasons or other wells.
No flow, weather, pumping-state classification, safe-volume claim, or confidence probability
is included. Do not choose a new grid based on this test and call the same test unseen.

## Reproduction and evidence

`summary.json` records source hashes, software versions, folds and configuration.
`examples.csv` includes input and outcome timestamps and split membership.
`excluded_hours.csv` lists overlapping exclusion reasons; `grid_search.csv` contains all
candidate scores; `test_predictions.csv` contains the paired held-out predictions.

References: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
and [time-series splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Custom timestamp splits are supplied to GridSearchCV because missing hourly examples
make an equal-row split unequal in elapsed time.
"""
    (output / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", default=[])
    parser.add_argument("--borehole-id", type=int, default=2)
    parser.add_argument("--sensor-id", type=int, default=4)
    args = parser.parse_args()
    run(args.input, args.output, args.protect, args.borehole_id, args.sensor_id)
