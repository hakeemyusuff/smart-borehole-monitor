import numpy as np
import pandas as pd
import unittest

from scripts.test_rainfall_and_response import rainfall_features, volume_until


def payload(rain):
    return {'hourly_units':{'precipitation':'mm'}, 'hourly':{
        'time':pd.date_range('2026-01-01',periods=len(rain),freq='h').astype(str).tolist(),
        'precipitation':rain}}


def check_api_decays_and_future_rain_does_not_change_past():
    rain = [10.]+[0.]*99
    before = rainfall_features(payload(rain))
    rain[80] = 500.
    after = rainfall_features(payload(rain))
    pd.testing.assert_frame_equal(before.iloc[:80],after.iloc[:80])
    np.testing.assert_allclose(before.api_24h_mm.iloc[24],10*np.exp(-1))
    assert before.rain_24h_mm.iloc[24] == 0


def check_missing_rain_and_missing_hours_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        rainfall_features(payload([0.,None]))
    p = payload([0.,0.,0.])
    p['hourly']['time'][1] = p['hourly']['time'][0]
    with unittest.TestCase().assertRaises(ValueError):
        rainfall_features(p)


def check_volume_clips_at_level_timestamp():
    start = pd.Timestamp('2026-01-01',tz='UTC')
    f = pd.DataFrame({'captured_at':[start+pd.Timedelta(minutes=n) for n in [1,2,10]],
                      'abstraction_rate':[20.,20.,100.]})
    volume,minutes = volume_until(f,start,start+pd.Timedelta(seconds=90))
    np.testing.assert_allclose(volume,30.)
    np.testing.assert_allclose(minutes,1.5)


class RainfallResponseTests(unittest.TestCase):
    def test_api(self):
        check_api_decays_and_future_rain_does_not_change_past()

    def test_missing(self):
        check_missing_rain_and_missing_hours_rejected()

    def test_volume(self):
        check_volume_clips_at_level_timestamp()
