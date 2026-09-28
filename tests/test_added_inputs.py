import unittest
import pandas as pd
from scripts.compare_added_inputs import flow_features, weather_features


class AddedInputTests(unittest.TestCase):
    def setUp(self):
        self.t = pd.Timestamp("2026-09-01T12:00:00Z")
        self.level = pd.DataFrame({"captured_at": [self.t], "created_at": [self.t]})

    def test_flow_integrates_preceding_windows_without_bridging_outage(self):
        times = pd.DatetimeIndex([self.t-pd.Timedelta(minutes=30), self.t-pd.Timedelta(minutes=1), self.t])
        flow = pd.DataFrame({"captured_at": times, "created_at": times, "abstraction_rate": [20., 20., 20.]})
        result = flow_features(flow, self.level, self.t)
        self.assertEqual(result["recorded_volume_1h_l"], 60.)
        self.assertEqual(result["recorded_flow_minutes_1h"], 3.)

    def test_delayed_flow_excluded(self):
        flow = pd.DataFrame({"captured_at": [self.t], "created_at": [self.t+pd.Timedelta(minutes=1)], "abstraction_rate": [20.]})
        result = flow_features(flow, self.level, self.t)
        self.assertEqual(result["recorded_volume_1h_l"], 0.)

    def test_rate_window_clipped_at_lookback(self):
        capture = self.t-pd.Timedelta(minutes=59, seconds=30)
        flow = pd.DataFrame({"captured_at": [capture], "created_at": [capture], "abstraction_rate": [20.]})
        self.assertEqual(flow_features(flow, self.level, self.t)["recorded_volume_1h_l"], 10.)

    def test_weather_uses_arrived_snapshots_and_one_per_hour(self):
        times = [self.t-pd.Timedelta(minutes=40), self.t-pd.Timedelta(minutes=20), self.t+pd.Timedelta(minutes=1)]
        weather = pd.DataFrame({"created_at": times, "temperature": [25., 26., 99.], "humidity": [80., 81., 99.], "precipitation": [.2, .4, 99.]})
        result = weather_features(weather, self.t)
        self.assertEqual(result["temperature_latest"], 26.)
        self.assertEqual(result["precip_snapshot_mean_24h"], .4)
        self.assertEqual(result["weather_coverage_24h"], 1/24)
        self.assertIsNone(weather_features(weather, self.t+pd.Timedelta(hours=4)))


if __name__ == "__main__":
    unittest.main()
