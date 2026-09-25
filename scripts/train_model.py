import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = Path("data/level_training_table.csv")
MODEL_PATH = Path("models/level_change_linear.json")

FEATURES = [
    "delta_1h",
    "delta_3h",
    "delta_6h",
]

TARGET = "level_2h"
TRAIN_FRACTION = 0.8


def main() -> None:
    if MODEL_PATH.exists():
        raise FileExistsError(
            f"{MODEL_PATH} already exists. "
            "The saved model has been preserved."
        )
    
    table = pd.read_csv(INPUT_PATH)

    timestamp_columns = [
        "forecast_at",
        "target_at",
        "target_captured_at",
        "target_arrived_at",
        "grid_start",
        "grid_end",
    ]

    for column in timestamp_columns:
        table[column] = pd.to_datetime(
            table[column],
            utc=True,
            errors="raise",
        )

    table = table.sort_values("forecast_at").reset_index(drop=True)

    if table.empty:
        raise ValueError("The training table is empty.")

    if not np.isfinite(table[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("The training table contains missing or invalid values.")

    # Split the original timeline, including hours skipped for missing data.
    timeline = pd.date_range(
        table["grid_start"].iloc[0],
        table["grid_end"].iloc[0],
        freq="h",
    )

    split_time = timeline[int(len(timeline) * TRAIN_FRACTION)]

    # Every training answer must already be available before evaluation starts.
    train_mask = (
        (table["forecast_at"] < split_time)
        & (table["target_at"] < split_time)
        & (table["target_captured_at"] < split_time)
        & (table["target_arrived_at"] < split_time)
    )

    test_mask = table["forecast_at"] >= split_time

    train = table.loc[train_mask]
    test = table.loc[test_mask]

    print("\nWater-level ranges in metres:")

    for name, part in [("Training", train), ("Evaluation", test)]:
        print(
            f"{name}: "
            f"current level {part['level_now'].min():.2f}"
            f" to {part['level_now'].max():.2f}; "
            f"future level {part[TARGET].min():.2f}"
            f" to {part[TARGET].max():.2f}"
        )

    excluded = len(table) - len(train) - len(test)

    if len(train) < 2 or len(test) < 2:
        raise ValueError("Not enough examples for training and evaluation.")

    X_train = train[FEATURES]

    # The answer we now teach the model is the change in level.
    y_train_change = train[TARGET] - train["level_now"]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train_change)

    predicted_change = model.predict(X_test)

    # Convert the predicted change back into a water-level prediction.
    predictions = (
        test["level_now"].to_numpy() + predicted_change
    )

    # Persistence predicts that the future level equals the current level.
    persistence = test["level_now"].to_numpy()

    comparison = test[
    ["forecast_at", "level_now", "level_2h"]
    ].copy()

    comparison["linear_prediction"] = predictions
    comparison["persistence_prediction"] = persistence

    comparison["linear_error_cm"] = (
        comparison["linear_prediction"] - comparison["level_2h"]
    ) * 100

    comparison["persistence_error_cm"] = (
        comparison["persistence_prediction"] - comparison["level_2h"]
    ) * 100

    print("\nLast 20 evaluation examples:")
    print(
        comparison.tail(20).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("\nAverage signed error in centimetres:")
    print(
        comparison[
            ["linear_error_cm", "persistence_error_cm"]
        ].mean().round(3)
    )

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    persistence_mae = mean_absolute_error(y_test, persistence)
    persistence_rmse = np.sqrt(mean_squared_error(y_test, persistence))

    print(f"Total examples:    {len(table)}")
    print(f"Training examples: {len(train)}")
    print(f"Test examples:     {len(test)}")
    print(f"Boundary excluded: {excluded}")
    print(f"Test starts at:    {split_time}")

    print("\nErrors in centimetres — lower is better:")
    print(f"{'Method':<20} {'MAE':>10} {'RMSE':>10}")
    print(
        f"{'Linear regression':<20}"
        f" {model_mae * 100:>10.3f}"
        f" {model_rmse * 100:>10.3f}"
    )
    print(
        f"{'Persistence':<20}"
        f" {persistence_mae * 100:>10.3f}"
        f" {persistence_rmse * 100:>10.3f}"
    )

    print("\nLearned equation for the two-hour CHANGE in level:")
    print(f"Intercept: {model.intercept_:.6f}")

    for feature, coefficient in zip(FEATURES, model.coef_):
        print(f"{feature}: {coefficient:+.6f}")


        saved_model = {
        "schema_version": 2,
        "model_type": "linear_regression",
        "prediction_type": "level_change",
        "features": FEATURES,
        "feature_contract": "captured_before_anchor_arrived_by_issue_v1",
        "horizon_hours": 2,
        "tolerance_minutes": 35,
        "borehole_id": 2,
        "sensor_id": 4,
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "training_rows": len(train),
        "training_cutoff_utc": str(split_time),
        "training_table_sha256": hashlib.sha256(
            INPUT_PATH.read_bytes()
        ).hexdigest(),
        "evaluation_rows": len(test),
        "evaluation_status": "exploratory_after_inspecting_evaluation_errors",
        "evaluation_metrics": {
            "mae_m": float(model_mae),
            "rmse_m": float(model_rmse),
            "persistence_mae_m": float(persistence_mae),
            "persistence_rmse_m": float(persistence_rmse),
        },
    }

    model_contents = json.dumps(
        saved_model,
        sort_keys=True,
        allow_nan=False,
    ).encode()

    saved_model["model_version"] = (
        "level-change-linear-"
        + hashlib.sha256(model_contents).hexdigest()[:16]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("x") as file:
        json.dump(saved_model, file, indent=2)
        file.write("\n")

    print(f"\nSaved model to {MODEL_PATH}")
    print("This model predicts a change in metres.")
    print("Future level = current level + predicted change.")
    
    
if __name__ == "__main__":
    main()
