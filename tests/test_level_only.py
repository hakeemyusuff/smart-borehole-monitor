"""Timing and metric regression checks; fixtures never enter the real experiment."""
import unittest
import numpy as np
import pandas as pd

from scripts.compare_level_only import build_examples, available_before, chronological_splits, metrics


class LevelOnlyTests(unittest.TestCase):
    def readings(self):
        t = pd.date_range("2026-01-01", periods=49, freq="30min", tz="UTC")
        return pd.DataFrame({"borehole_id": 2, "sensor_id": 4, "captured_at": t,
                             "created_at": t, "water_level": np.arange(len(t))/100+4})

    def test_late_arrival_and_future_capture_not_used(self):
        raw = self.readings()
        t = pd.Timestamp("2026-01-01T10:00:00Z")
        raw.loc[raw.captured_at == t, "created_at"] = t+pd.Timedelta(hours=1)
        examples, _, _ = build_examples(raw, 2, 4)
        row = examples.loc[examples.forecast_at == t].iloc[0]
        self.assertEqual(row.input_0h_captured_at, t-pd.Timedelta(minutes=30))
        for lag in [0, 1, 3, 6]:
            self.assertLessEqual(row[f"input_{lag}h_arrived_at"], t)
            self.assertLessEqual(row[f"input_{lag}h_captured_at"], t-pd.Timedelta(hours=lag))

    def test_long_gap_excluded_instead_of_imputed(self):
        raw = self.readings()
        t = pd.Timestamp("2026-01-01T10:00:00Z")
        raw = raw.loc[~raw.captured_at.between(t-pd.Timedelta(hours=1), t)]
        examples, exclusions, _ = build_examples(raw, 2, 4)
        self.assertNotIn(t, examples.forecast_at.tolist())
        self.assertIn("missing_available_level_0h", exclusions.loc[exclusions.forecast_at == t, "reasons"].iloc[0])

    def test_boundary_checks_label_arrival(self):
        examples, _, _ = build_examples(self.readings(), 2, 4)
        cutoff = pd.Timestamp("2026-01-01T12:00:00Z")
        examples.loc[0, "target_arrived_at"] = cutoff
        eligible = available_before(examples, cutoff)
        self.assertFalse(eligible.iloc[0])
        self.assertTrue((examples.loc[eligible, "target_at"] < cutoff).all())

    def test_duplicate_capture_fails(self):
        raw = self.readings()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_examples(pd.concat([raw, raw.iloc[:1]]), 2, 4)

    def test_metric_shape_prevents_pairwise_broadcast(self):
        y = np.array([10., 20., 30.])
        self.assertEqual(metrics(y, y)["mae_m"], 0)
        with self.assertRaises(ValueError):
            metrics(y, y[:, None])

    def test_cv_training_labels_precede_validation(self):
        t = pd.date_range("2026-01-01", periods=240, freq="h", tz="UTC")
        table = pd.DataFrame({"forecast_at": t, "target_at": t+pd.Timedelta(hours=2),
                              "target_captured_at": t+pd.Timedelta(hours=2),
                              "target_arrived_at": t+pd.Timedelta(hours=5)})
        splits, _ = chronological_splits(table, t[0], t[-1]+pd.Timedelta(hours=1))
        for train, valid in splits:
            self.assertLess(table.iloc[train].target_arrived_at.max(), table.iloc[valid].forecast_at.min())
            self.assertEqual(len(set(train) & set(valid)), 0)


if __name__ == "__main__":
    unittest.main()
