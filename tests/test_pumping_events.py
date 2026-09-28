import unittest
import pandas as pd
from scripts.analyze_pumping_events import events_table


class PumpingEventTests(unittest.TestCase):
    def fixture(self):
        t = pd.Timestamp("2026-09-01T12:00:00Z")
        flow_times = pd.date_range(t+pd.Timedelta(minutes=1), periods=5, freq="min")
        flow = pd.DataFrame({"captured_at": flow_times, "abstraction_rate": [20.]*5})
        level_times = pd.date_range(t-pd.Timedelta(minutes=1), periods=7, freq="min")
        level = pd.DataFrame({"captured_at": level_times,
                              "water_level": [5., 5., 4.9, 4.8, 4.7, 4.6, 4.5]})
        return t, level, flow

    def test_pairs_before_during_and_recovery(self):
        t, level, flow = self.fixture()
        level = pd.concat([level, pd.DataFrame({"captured_at": [t+pd.Timedelta(minutes=50)], "water_level": [4.9]})], ignore_index=True)
        events, _ = events_table(level, flow)
        row = events.iloc[0]
        self.assertTrue(row.included_in_relationship)
        self.assertAlmostEqual(row.recorded_volume_l, 100.)
        self.assertAlmostEqual(row.drawdown_to_min_m, .5)
        self.assertAlmostEqual(row.recovery_45min_fraction, .8)

    def test_gap_is_flagged_and_not_integrated_as_continuous_flow(self):
        t, level, flow = self.fixture()
        flow.loc[4, "captured_at"] = t+pd.Timedelta(minutes=10)
        events, _ = events_table(level, flow)
        row = events.iloc[0]
        self.assertFalse(row.included_in_relationship)
        self.assertIn("flow_gap_over_2min", row.quality_flags)
        self.assertAlmostEqual(row.recorded_volume_l, 100.)

    def test_next_event_prevents_contaminated_recovery_match(self):
        t, level, flow = self.fixture()
        flow = pd.concat([flow, pd.DataFrame({"captured_at": [t+pd.Timedelta(minutes=45)], "abstraction_rate": [20.]})], ignore_index=True)
        level = pd.concat([level, pd.DataFrame({"captured_at": [t+pd.Timedelta(minutes=50)], "water_level": [4.2]})], ignore_index=True)
        events, _ = events_table(level, flow)
        self.assertTrue(pd.isna(events.iloc[0].get("recovery_45min_level_m")))


if __name__ == "__main__":
    unittest.main()
