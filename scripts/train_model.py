"""
BoreSense model trainer — the study session.

Reads data/training_table.csv (the flashcards), splits time-wise into
study material (first 80% of hours) and a held-back exam (last 20%),
trains a RandomForest to predict all 24 future levels at once, then grades it:

  - MAE (mean absolute error, in metres) at 1h, 12h, 24h ahead
  - against the "lazy student" baseline: predict the level just stays put
  - feature importances: which inputs the forest actually found useful

Artifacts saved for the inference step:
  models/rf_level.joblib        - the trained forest
  models/feature_columns.json   - exact feature order the model expects

Run:  python3 -m scripts.train_model
Requires: pandas, scikit-learn, joblib  (uv add scikit-learn joblib)
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ── Config ──────────────────────────────────────────────────────────────────
TABLE_PATH = "data/training_table.csv"
MODEL_PATH = "models/rf_level.joblib"
FEATURES_PATH = "models/feature_columns.json"
TRAIN_FRACTION = 0.8  # first 80% of time = study, last 20% = exam
N_TREES = 300
RANDOM_STATE = 42
HORIZON_HOURS = 2
MIN_ROWS = 200


def main() -> None:
    if not os.path.exists(TABLE_PATH):
        print(f"No training table at {TABLE_PATH} — run build_training_table first.")
        return

    table = pd.read_csv(TABLE_PATH, parse_dates=["t"])
    table = table.sort_values("t").reset_index(drop=True)

    if len(table) < MIN_ROWS:
        print(
            f"Only {len(table)} rows — need at least {MIN_ROWS} for a meaningful "
            f"train/test split. Collect more data first."
        )
        return

    target_cols = [f"y_{HORIZON_HOURS}"]
    feature_cols = [c for c in table.columns if c != "t" and c not in target_cols]

    X = table[feature_cols].to_numpy(dtype=float)
    Y = table[target_cols].to_numpy(dtype=float).ravel()

    # Time-based split: NEVER shuffle time series. The exam must be the
    # future relative to the study material, exactly like production.
    split = int(len(table) * TRAIN_FRACTION)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    t_split = table["t"].iloc[split]
    print(f"Flashcards: {len(table)} total → {split} study / {len(table) - split} exam")
    print(
        f"Exam covers {t_split} → {table['t'].iloc[-1]}  (the model never sees this period)"
    )

    print(f"Training RandomForest ({N_TREES} trees) on {len(feature_cols)} features...")
    model = RandomForestRegressor(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)

    # ── Grading ──
    pred = model.predict(X_test)  # shape (n_exam, 24)
    level_now_idx = feature_cols.index("level_now")
    lazy = X_test[:, [level_now_idx]]

    print("\nExam results — mean absolute error in metres (lower is better):")
    print(f"{'horizon':>8} | {'forest':>7} | {'lazy baseline':>13}")
    y_true = Y_test.ravel()
    mae_model = float(np.mean(np.abs(pred - y_true)))
    mae_lazy = float(np.mean(np.abs(lazy - y_true)))
    verdict = "beats lazy" if mae_model < mae_lazy else "LOSES TO LAZY — investigate"
    print(f"{HORIZON_HOURS:>6}h  | {mae_model:>6.2f}m | {mae_lazy:>12.2f}m   {verdict}")

    print("\nFeature importances (what the forest found useful):")
    ranked = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda p: p[1], reverse=True
    )
    for name, imp in ranked:
        bar = "#" * max(1, round(imp * 60))
        print(f"  {name:>14}  {imp:5.1%}  {bar}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved feature order → {FEATURES_PATH}")


if __name__ == "__main__":
    main()
