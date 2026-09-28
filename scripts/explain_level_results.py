"""Exploratory diagnostics and ordinary linear regression on frozen experiment splits.

No database access, app imports, source CSV edits, or retraining of the forest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.compare_level_only import FEATURES, chronological_splits, digest, metrics

MODELS = ["persistence", "random_forest", "linear_regression"]


def read_times(path):
    frame = pd.read_csv(path)
    for name in frame.columns:
        if name.endswith("_at"):
            frame[name] = pd.to_datetime(frame[name], utc=True, format="mixed")
    return frame


def movement_groups(change):
    # Descriptive bins, selected before this diagnostic run; not model inputs.
    magnitude = change.abs().round(8)
    return np.select([magnitude <= .02, magnitude <= .05],
                     ["0–2 cm", ">2–5 cm"], default=">5 cm")


def run(previous: Path, output: Path):
    if output.exists():
        raise ValueError("Choose a new output directory; previous reports are preserved")
    summary = json.loads((previous / "summary.json").read_text())
    protected = [Path(name) for name in summary["input_sha256_before"]]
    protected += list(previous.glob("*"))
    before = {str(p): digest(p) for p in protected if p.is_file()}
    for name, sha in summary["input_sha256_before"].items():
        if before[name] != sha:
            raise ValueError("Original CSV differs from frozen experiment; do not mix datasets")
    examples = read_times(previous / "examples.csv")
    excluded = read_times(previous / "excluded_hours.csv")
    development = examples.loc[examples.partition == "development"].reset_index(drop=True)
    test = read_times(previous / "test_predictions.csv")
    start = min(examples.forecast_at.min(), excluded.forecast_at.min())
    splits, fold_info = chronological_splits(development, start, pd.Timestamp(summary["test_cutoff_utc"]))
    if fold_info != summary["folds"]:
        raise ValueError("Validation splits differ from original experiment")
    cv_scores = []
    for train, valid in splits:
        model = LinearRegression()
        model.fit(development.iloc[train][FEATURES], development.iloc[train].actual)
        cv_scores.append(metrics(development.iloc[valid].actual,
                                 model.predict(development.iloc[valid][FEATURES]))["mae_m"])
    model = LinearRegression().fit(development[FEATURES], development.actual)
    test["linear_regression"] = model.predict(test[FEATURES])
    scores = {name: metrics(test.actual, test[name]) for name in MODELS}
    for name in MODELS[:2]:
        for metric, score in scores[name].items():
            if not np.isclose(score, summary["test_scores"][name][metric], atol=1e-12, rtol=0):
                raise ValueError("Frozen baseline/forest scores could not be reproduced")
    test["observed_change_m"] = test.actual - test.level_now
    test["movement_group"] = movement_groups(test.observed_change_m)
    rows = []
    for group in ["0–2 cm", ">2–5 cm", ">5 cm"]:
        part = test.loc[test.movement_group == group]
        if part.empty:
            continue
        row = {"group": group, "count": len(part), "percent": len(part)/len(test)*100}
        for name in MODELS:
            row[f"{name}_mae_cm"] = metrics(part.actual, part[name])["mae_m"]*100
            row[f"{name}_sum_absolute_error_m"] = float((part.actual-part[name]).abs().sum())
        rows.append(row)
    groups = pd.DataFrame(rows)
    for name in MODELS:
        test[f"{name}_absolute_error_m"] = (test.actual-test[name]).abs()
    test["forest_minus_persistence_error_m"] = test.random_forest_absolute_error_m-test.persistence_absolute_error_m
    details = {"status": "exploratory_followup_after_test_inspection", "test_scores": scores,
               "linear_cv_fold_mae_m": cv_scores, "linear_cv_mean_mae_m": float(np.mean(cv_scores)),
               "linear_coefficients": dict(zip(FEATURES, model.coef_.tolist())),
               "linear_intercept_m": float(model.intercept_), "groups": rows,
               "forest_better_rows": int((test.forest_minus_persistence_error_m < -1e-12).sum()),
               "forest_worse_rows": int((test.forest_minus_persistence_error_m > 1e-12).sum()),
               "folds": fold_info, "source_hashes": before}
    if before != {str(p): digest(p) for p in protected if p.is_file()}:
        raise RuntimeError("Protected source files changed")
    details["protected_inputs_unchanged"] = True
    output.mkdir(parents=True, exist_ok=False)
    test.to_csv(output / "exploratory_predictions.csv", index=False)
    groups.to_csv(output / "errors_by_movement.csv", index=False)
    test.sort_values("random_forest_absolute_error_m", ascending=False).head(15).to_csv(output / "largest_forest_errors.csv", index=False)
    (output / "summary.json").write_text(json.dumps(details, indent=2)+"\n")
    report = """# Exploratory water-level-only follow-up

This follow-up was proposed AFTER viewing the initial test scores. It reuses the
same data, four features and chronological boundaries. It is exploratory, not a
new untouched final test. All original CSVs and all original experiment artifacts
are fingerprinted and remain unchanged. No app, database or device was modified.

## Why compare linear regression?

Persistence predicts the latest available level. Ordinary least-squares linear
regression learns one weighted formula using current level and changes over 1, 3
and 6 hours, plus an intercept. It fits squared errors on development data only;
MAE remains our primary comparison metric. This is multiple linear regression
(four inputs), not a straight line extrapolated along calendar time.

It uses standard LinearRegression with an intercept and no polynomial terms,
feature selection or parameter search. No grid search is needed for this plain
least-squares coefficient fit. Three identical expanding CV splits are retained
for comparison; no forest is retrained and no forest parameters are changed.

## Results

| Method | MAE (cm) | RMSE (cm) |
|---|---:|---:|
"""
    for name in MODELS:
        report += f"| {name} | {scores[name]['mae_m']*100:.2f} | {scores[name]['rmse_m']*100:.2f} |\n"
    report += f"\nLinear regression mean CV MAE: {np.mean(cv_scores)*100:.2f} cm. Fold MAEs: {', '.join(f'{v*100:.2f}' for v in cv_scores)} cm.\n"
    report += "\n## Diagnostic groups\n\nGroups describe the absolute difference between the latest input measurement and\nthe eventual target observation. They use future outcomes for diagnosis ONLY, never\nas model inputs or operational selectors. Small endpoint change does not prove\nthere was no pumping or drawdown/recovery between those endpoints. These are not\npumping-state labels. Bins are descriptive choices, not scientific thresholds.\n\n| Endpoint change | Examples | Share | Persistence MAE (cm) | Forest MAE (cm) | Linear MAE (cm) |\n|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        report += f"| {row['group']} | {row['count']} | {row['percent']:.1f}% | {row['persistence_mae_cm']:.2f} | {row['random_forest_mae_cm']:.2f} | {row['linear_regression_mae_cm']:.2f} |\n"
    report += f"\nThe forest had lower absolute error on {details['forest_better_rows']} examples and higher error on {details['forest_worse_rows']}.\n"
    report += "\n## Fitted formula\n\nCoefficients describe associations in this dataset, not physical causes or\naquifer parameters. They are fitted on the 326 development examples only.\n\n```text\npredicted_level_m = " + f"{model.intercept_:.8f}" + "".join(f" {coef:+.8f} * {feature}" for feature, coef in zip(FEATURES, model.coef_)) + "\n```\n"
    report += """
## Interpretation limits and next evaluation

One well, limited history and correlated examples; sparse first validation fold
(13 rows) remains. Outcome-defined groups have different difficulty by construction:
persistence error IS absolute endpoint change. The informative comparison is how
other methods behave within those same groups, not a causal explanation of pumping.
No significance claim, calibrated uncertainty, volume estimate or seasonal claim.
Future model selection should use development evidence and be confirmed on newly
collected observations before calling another result an independent final test.

Reference: [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

Reproduce from the backend root, choosing an unused output directory:

```bash
.venv/bin/python -B -m scripts.explain_level_results \\
  --previous analysis/level_only_2026-09-22 \\
  --output analysis/level_only_followup_reproduction
```
"""
    (output / "report.md").write_text(report)
    print(json.dumps({k:v for k,v in details.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.previous, args.output)
