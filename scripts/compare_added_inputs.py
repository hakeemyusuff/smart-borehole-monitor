"""CSV-only exploratory feature ablation; source data and prior results are read-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from scripts.compare_level_only import FEATURES, GRID, chronological_splits, digest, metrics
from scripts.explain_level_results import read_times

FLOW_FEATURES = ["recorded_volume_1h_l", "recorded_volume_24h_l",
                 "recorded_flow_minutes_1h", "level_coverage_24h"]
WEATHER_FEATURES = ["temperature_latest", "humidity_latest", "precip_snapshot_latest",
                    "precip_snapshot_mean_24h", "weather_coverage_24h", "weather_age_h"]
SETS = {"level_only": FEATURES, "level_flow": FEATURES+FLOW_FEATURES,
        "level_weather": FEATURES+WEATHER_FEATURES,
        "level_flow_weather": FEATURES+FLOW_FEATURES+WEATHER_FEATURES}


def flow_features(flows, levels, t):
    f = flows.loc[(flows.captured_at <= t) & (flows.created_at <= t)].sort_values("captured_at").copy()
    # Rate describes the preceding firmware accumulation window. Exact duration
    # is absent from the export. Infer adjacent <=2 min intervals; otherwise use
    # one nominal minute, never integrate across a long unobserved interval.
    seconds = f.captured_at.diff().dt.total_seconds()
    seconds = seconds.where((seconds > 0) & (seconds <= 120), 60.)
    f["window_start"] = f.captured_at-pd.to_timedelta(seconds, unit="s")
    out = {}
    for h in [1, 24]:
        start = t-pd.Timedelta(hours=h)
        left = f.window_start.clip(lower=start)
        minutes = ((f.captured_at-left).dt.total_seconds()/60).clip(lower=0)
        out[f"recorded_volume_{h}h_l"] = float((minutes*f.abstraction_rate).sum())
        if h == 1:
            out["recorded_flow_minutes_1h"] = float(minutes[f.abstraction_rate > 0].sum())
    available = levels.loc[(levels.captured_at > t-pd.Timedelta(hours=24))
                           & (levels.captured_at <= t) & (levels.created_at <= t)]
    out["level_coverage_24h"] = min(1., available.captured_at.dt.floor("30min").nunique()/48)
    return out


def weather_features(weather, t):
    w = weather.loc[weather.created_at <= t].sort_values("created_at")
    if w.empty or t-w.created_at.iloc[-1] > pd.Timedelta(hours=2):
        return None
    last = w.iloc[-1]
    recent = w.loc[w.created_at > t-pd.Timedelta(hours=24)].copy()
    recent["hour"] = recent.created_at.dt.floor("h")
    recent = recent.drop_duplicates("hour", keep="last")
    return {"temperature_latest": float(last.temperature), "humidity_latest": float(last.humidity),
            "precip_snapshot_latest": float(last.precipitation),
            "precip_snapshot_mean_24h": float(recent.precipitation.mean()),
            "weather_coverage_24h": min(1., len(recent)/24),
            "weather_age_h": (t-last.created_at).total_seconds()/3600}


def run(previous, csv_dir, output):
    if output.exists():
        raise ValueError("Use an unused output directory")
    original = json.loads((previous/"summary.json").read_text())
    protected = [csv_dir/name for name in ["water_level_reading.csv", "flow_reading.csv", "weather.csv"]]
    protected += [p for p in previous.iterdir() if p.is_file()]
    before = {str(p.resolve()): digest(p) for p in protected}
    for name, expected in original["input_sha256_before"].items():
        if digest(csv_dir/Path(name).name) != expected:
            raise ValueError("CSV differs from original experiment")
    level = read_times(csv_dir/"water_level_reading.csv")
    flow = read_times(csv_dir/"flow_reading.csv")
    weather = read_times(csv_dir/"weather.csv")
    level = level.loc[(level.borehole_id == 2) & (level.sensor_id == 4)]
    flow = flow.loc[(flow.borehole_id == 2) & (flow.sensor_id == 5)]
    weather = weather.loc[weather.location_id == 2]
    if flow.empty or weather.empty:
        raise ValueError("Missing selected flow/weather source")
    for data, columns in [(flow, ["abstraction_rate"]), (weather, ["temperature", "humidity", "precipitation"])]:
        if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
            raise ValueError("Invalid source values; no automatic repairs")
    if (flow.abstraction_rate < 0).any() or (weather.precipitation < 0).any():
        raise ValueError("Negative flow or precipitation")
    examples = read_times(previous/"examples.csv")
    excluded = read_times(previous/"excluded_hours.csv")
    rows, missing = [], []
    for _, row in examples.iterrows():
        t = row.forecast_at
        w = weather_features(weather, t)
        if w is None:
            missing.append({"forecast_at": t, "reason": "no_weather_snapshot_within_2h"})
            continue
        rows.append({**row.to_dict(), **flow_features(flow, level, t), **w})
    table = pd.DataFrame(rows)
    dev = table.loc[table.partition == "development"].reset_index(drop=True)
    test = table.loc[table.partition == "test"].reset_index(drop=True)
    if len(test) < 10:
        raise ValueError("Too few common test examples")
    start = min(examples.forecast_at.min(), excluded.forecast_at.min())
    splits, folds = chronological_splits(dev, start, pd.Timestamp(original["test_cutoff_utc"]))
    baseline_cv = [metrics(dev.iloc[v].actual, dev.iloc[v].level_now)["mae_m"] for _, v in splits]
    results = [{"feature_set": "level_only", "model": "persistence",
                "cv_mae_m": float(np.mean(baseline_cv)), "params": {}}]
    trained, searches, fold_scores = {}, {}, []
    # Finish all selection using development data before scoring any test model.
    for set_name, columns in SETS.items():
        linear_cv = []
        for fold, (tr, va) in enumerate(splits, 1):
            lm = LinearRegression().fit(dev.iloc[tr][columns], dev.iloc[tr].actual)
            score = metrics(dev.iloc[va].actual, lm.predict(dev.iloc[va][columns]))["mae_m"]
            linear_cv.append(score)
            fold_scores.append({"feature_set": set_name, "model": "linear", "fold": fold, "mae_m": score})
        linear = LinearRegression().fit(dev[columns], dev.actual)
        trained[(set_name, "linear")] = linear
        results.append({"feature_set": set_name, "model": "linear", "cv_mae_m": float(np.mean(linear_cv)),
                        "params": {"intercept": float(linear.intercept_), "coefficients": dict(zip(columns, linear.coef_.tolist()))}})
        search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), GRID,
                              scoring="neg_mean_absolute_error", cv=splits, n_jobs=1,
                              refit=True, error_score="raise")
        search.fit(dev[columns], dev.actual)
        trained[(set_name, "forest")] = search.best_estimator_
        searches[set_name] = pd.DataFrame(search.cv_results_)
        results.append({"feature_set": set_name, "model": "forest", "cv_mae_m": -search.best_score_, "params": search.best_params_})
        for fold in range(1, len(splits)+1):
            fold_scores.append({"feature_set": set_name, "model": "forest", "fold": fold,
                                "mae_m": -float(search.cv_results_[f"split{fold-1}_test_score"][search.best_index_])})
        print(f"Finished development CV: {set_name}", flush=True)
    selected = min(results, key=lambda r:r["cv_mae_m"])
    selected_identity = {"feature_set": selected["feature_set"], "model": selected["model"]}
    for result in results:
        name, kind = result["feature_set"], result["model"]
        prediction = test.level_now.to_numpy() if kind == "persistence" else trained[(name,kind)].predict(test[SETS[name]])
        test[f"prediction_{name}_{kind}"] = prediction
        result.update({f"test_{k}":v for k,v in metrics(test.actual, prediction).items()})
    after = {str(p.resolve()): digest(p) for p in protected}
    if before != after:
        raise RuntimeError("Protected input changed during run")
    summary = {"status": "exploratory_same_previously_inspected_test_period", "results": results,
               "selected_by_development_cv": selected_identity,
               "source_hashes_before": before, "source_hashes_after": after, "inputs_unchanged": True,
               "common_development_rows": len(dev), "common_test_rows": len(test),
               "additional_excluded_rows": len(missing), "folds": folds, "baseline_cv_fold_mae_m": baseline_cv,
               "feature_sets": SETS, "forest_grid": GRID, "original_versions": original["versions"]}
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output/"common_examples.csv", index=False)
    test.to_csv(output/"exploratory_predictions.csv", index=False)
    pd.DataFrame(missing, columns=["forecast_at", "reason"]).to_csv(output/"additional_exclusions.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(output/"validation_fold_scores.csv", index=False)
    for name, search in searches.items():
        search.to_csv(output/f"grid_{name}.csv", index=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    report = """# Exploratory level, flow and weather comparison

No original CSV or prior experiment artifact was changed. No model deployed.
This is a follow-up on a previously inspected test period, not independent final validation.

## Comparison design

Four feature sets, each with ordinary linear regression and a Random Forest tuned
on the original 18-setting grid. Three expanding time-based CV windows retain the
original date boundaries and label-availability purge. All methods, including the
level-only controls and persistence, use the SAME common development and test rows.
Primary metric: MAE. Select the provisional method by mean development CV MAE,
not by whichever test score is smallest. No horizon/target/level-history changes.

## Added feature meanings and engineering assumptions

- Flow rates describe preceding measurement windows. The CSV omits their actual
  durations: infer duration from the preceding available sample when separation
  is at most two minutes; otherwise assign one nominal minute. Clip windows at
  the feature lookback boundary. Recorded volume = sum(rate L/min × duration min).
- Record estimated volume over 1 h and 24 h, and recorded positive-flow duration
  over 1 h. These describe logged intervals, NOT guaranteed complete abstraction.
  An empty flow window contributes zero recorded volume, not proof of no pumping.
- Level coverage is occupied 30-minute capture bins / 48 over the previous day,
  using only readings uploaded by forecast time. It is an availability proxy,
  not a calibrated probability that flow observations are complete.
- Flow inputs require both capture and upload times <= forecast time. No future
  pumping decisions are supplied to either model.
- Weather inputs are the latest temperature, humidity and precipitation SNAPSHOT,
  plus mean precipitation snapshot in the preceding 24 h, its hour-bin coverage,
  and latest snapshot age. Retain only the last available snapshot per hour for
  the mean. Latest snapshot must be <=2 h old; no missing-weather zero filling.
- Weather timestamps are database arrival times; provider observation times were
  not retained. These are model-based weather estimates, not an on-site rain gauge.
  Current-weather snapshots are NOT a complete hourly precipitation accumulation.
  No rain_24h total, rain_72h total, or recharge coefficient is inferred from them.

This tests whether the EXPORTED signals add forecasting information. It does not
establish whether a complete rainfall series or verified total abstraction would help.
Future deployment must implement these same feature definitions if selected.

## Results

"""
    report += f"Common development examples: {len(dev)}; test examples: {len(test)}; additional excluded hours: {len(missing)}.\n\n"
    report += "| Inputs | Method | Mean CV MAE (cm) | Test MAE (cm) | Test RMSE (cm) |\n|---|---|---:|---:|---:|\n"
    for r in results:
        report += f"| {r['feature_set']} | {r['model']} | {r['cv_mae_m']*100:.2f} | {r['test_mae_m']*100:.2f} | {r['test_rmse_m']*100:.2f} |\n"
    report += f"\nProvisional selection by development CV: **{selected_identity}**.\n"
    report += """
## Engineering interpretation

For a cylindrical well, A = pi D²/4. Conservation of water within the well gives
A × dh/dt = Q_net_inflow − Q_pump, with Q in m³/time and h in metres.
This is a simplified control-volume balance, not a fitted aquifer recharge model.
During pumping, measured drawdown reflects both removal and replenishment; after
pumping, recovery can reflect surrounding groundwater redistribution. Rainfall
does not instantaneously become well inflow. Recovery rate is not automatically
regional recharge or sustainable yield. Geometry, measurement uncertainty and
time alignment are needed before interpreting litres-per-metre relationships.

The measured diameter and pump-intake reference are still needed for a stored-water
estimate and a defensible stopping level. Short-term level forecasts alone do not
establish sustainable daily abstraction or dry-run protection.

## Thesis placement and limits

Chapter 3: measurement definitions, calibration, feature equations, split and search
methodology. Chapter 4: coverage, monitoring behaviour, model/feature comparisons,
drawdown/recovery observations and application demonstration. Chapter 5: conclusions,
limitations and remaining field validation. Present every comparison, not just a winner.
One well and sparse CV coverage remain; no statistical significance claim. Reserve
newly collected data for prospective confirmation after freezing the chosen method.

References:
- https://open-meteo.com/en/docs (current vs hourly weather)
- https://water.usgs.gov/ogw/gwrp/methods/wtf/issues_limititations.html
- https://pubs.usgs.gov/circ/circ1186/html/gw_dev.html

Reproduction (use a NEW output folder):
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m scripts.compare_added_inputs \\
  --previous analysis/level_only_2026-09-22 \\
  --csv-dir /mnt/c/Users/Ysf/Downloads \\
  --output analysis/added_inputs_reproduction
```
"""
    (output/"report.md").write_text(report)
    print(json.dumps({k:summary[k] for k in ["common_development_rows", "common_test_rows", "additional_excluded_rows", "selected_by_development_cv", "results"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.previous, args.csv_dir, args.output)
