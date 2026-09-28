"""Rule scenarios and authenticated API checks; no live database or pump access."""
import os
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://test:test@127.0.0.1:1/offline_test'
os.environ['SECRET_KEY'] = 'offline-test-only'
os.environ['ENABLE_SCHEDULER'] = 'false'

import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
from app.ml.recommendations import RecommendationPolicy, assess_recommendation
from app.ml.schemas import PredictionStatus

NOW = datetime(2026, 9, 25, 10, 20, tzinfo=timezone.utc)
# Demonstration thresholds for tests only, not field settings.
POLICY = RecommendationPolicy(sensor_id=4, minimum_level_m=1,
                              consideration_level_m=2, basis='Synthetic rule-testing policy')


def snapshot(**changes):
    data = dict(status='fresh', checked_at=NOW, message='', current_level=3,
                current_level_captured_at=NOW - timedelta(minutes=5),
                predicted_level_2h=3.2, predicted_for=NOW + timedelta(hours=1),
                issued_at=NOW - timedelta(minutes=15), model_version='test-model')
    return PredictionStatus(**(data | changes))


class RecommendationTests(unittest.TestCase):
    def test_decision_table(self):
        cases = [
            ({}, 'consider'),
            ({'current_level': 1}, 'defer'),
            ({'current_level': .9, 'status': 'unavailable', 'predicted_level_2h': None}, 'defer'),
            ({'predicted_level_2h': 1}, 'defer'),
            ({'current_level': 1.5}, 'reassess'),
            ({'predicted_level_2h': 1.5}, 'reassess'),
            ({'current_level': 2, 'predicted_level_2h': 2}, 'consider'),
            ({'status': 'stale'}, 'unavailable'),
            ({'current_level_captured_at': NOW - timedelta(minutes=36)}, 'unavailable'),
            ({'current_level_captured_at': NOW + timedelta(seconds=1)}, 'unavailable'),
            ({'current_level': None}, 'unavailable'),
            ({'predicted_for': NOW}, 'unavailable'),
            ({'checked_at': NOW - timedelta(minutes=2)}, 'unavailable'),
            ({'checked_at': NOW + timedelta(seconds=1)}, 'unavailable'),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                result = assess_recommendation(snapshot(**changes), POLICY, NOW)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.suggested_start_at is not None, expected == 'consider')
                self.assertTrue(result.advisory_only)

    def test_operator_selected_half_metre_minimum(self):
        policy = POLICY.model_copy(update={'minimum_level_m': .5, 'consideration_level_m': 1.0})
        for level, forecast, expected in [(.5, 1.2, 'defer'), (.8, 1.2, 'reassess'),
                                           (1.2, .5, 'defer'), (1, 1, 'consider')]:
            with self.subTest(level=level, forecast=forecast):
                result = assess_recommendation(snapshot(current_level=level, predicted_level_2h=forecast), policy, NOW)
                self.assertEqual(result.status, expected)

    def test_no_policy_means_no_recommendation(self):
        result = assess_recommendation(snapshot(), None, NOW)
        self.assertEqual(result.reason_code, 'configuration_required')
        self.assertIsNone(result.suggested_start_at)

    def test_expiration_never_outlives_measurement_or_forecast(self):
        result = assess_recommendation(snapshot(current_level_captured_at=NOW-timedelta(minutes=34)), POLICY, NOW)
        self.assertEqual(result.valid_until, NOW + timedelta(minutes=1))
        result = assess_recommendation(snapshot(predicted_for=NOW+timedelta(seconds=30)), POLICY, NOW)
        self.assertEqual(result.valid_until, NOW + timedelta(seconds=30))

    def test_reassessment_is_after_next_forecast_grace_period(self):
        result = assess_recommendation(snapshot(current_level=1.5), POLICY, NOW)
        self.assertEqual(result.next_review_at, NOW.replace(minute=40))
        self.assertIsNone(result.suggested_start_at)

    def test_policy_rejects_invalid_levels(self):
        for fields in [dict(minimum_level_m=-1), dict(consideration_level_m=1),
                       dict(consideration_level_m=float('nan')), dict(basis=' '*12)]:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                RecommendationPolicy(**(POLICY.model_dump() | fields))

    def test_review_rolls_to_next_hour_after_half_hour_grace(self):
        moment = NOW.replace(minute=40)
        result = assess_recommendation(snapshot(current_level=1.5, checked_at=moment,
            current_level_captured_at=moment), POLICY, moment)
        self.assertEqual(result.next_review_at, moment.replace(hour=11, minute=10))

    def test_site_policy_defaults_and_explicit_overrides(self):
        from app.core.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            for overrides in ({}, {'pump_recommendation_policies': {}}):
                settings = Settings(_env_file=None, database_url='offline', secret_key='offline', **overrides)
                self.assertEqual(set(settings.pump_recommendation_policies), {2})
                policy = settings.pump_recommendation_policies[2]
                self.assertEqual(policy.sensor_id, 4)
                self.assertEqual(policy.minimum_level_m, .5)
                self.assertEqual(policy.consideration_level_m, 1)
                self.assertTrue(policy.basis)
            settings = Settings(_env_file=None, database_url='offline', secret_key='offline',
                                pump_recommendation_policies={2: None})
            self.assertIsNone(settings.pump_recommendation_policies[2])
            settings = Settings(_env_file=None, database_url='offline', secret_key='offline',
                                pump_recommendation_policies={2: POLICY})
            self.assertEqual(settings.pump_recommendation_policies[2], POLICY)
            with patch.dict(os.environ, {'PUMP_RECOMMENDATION_POLICIES': '{}'}):
                settings = Settings(_env_file=None, database_url='offline', secret_key='offline')
                self.assertEqual(settings.pump_recommendation_policies[2].minimum_level_m, .5)


class RecommendationApiTests(unittest.TestCase):
    def test_owner_checked_and_get_has_no_writes(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.auth.dependencies import get_current_user
        from app.core.database import get_session
        from app.ml import services
        from app.core.config import settings
        async def fake_session(): yield object()
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=8)
        app.dependency_overrides[get_session] = fake_session
        status = snapshot(checked_at=datetime.now(timezone.utc),
                          current_level_captured_at=datetime.now(timezone.utc),
                          predicted_for=datetime.now(timezone.utc)+timedelta(hours=1))
        model = SimpleNamespace(metadata={'borehole_id': 2, 'sensor_id': 4})
        try:
            with TestClient(app) as client, patch.object(services, 'get_prediction_status', new_callable=AsyncMock) as get_status, patch.object(services, '_model', model), patch.object(settings, 'pump_recommendation_policies', {2: POLICY}):
                get_status.return_value = status.model_dump()
                response = client.get('/api/predictions/2/recommendation')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['data']['status'], 'consider')
                self.assertEqual(get_status.call_args.args[:2], (2, 8))
                model.metadata['sensor_id'] = 99
                self.assertEqual(client.get('/api/predictions/2/recommendation').json()['data']['status'], 'unavailable')
                get_status.side_effect = ValueError('Borehole not found for this user')
                self.assertEqual(client.get('/api/predictions/999/recommendation').status_code, 404)
                app.dependency_overrides.pop(get_current_user)
                self.assertIn(client.get('/api/predictions/2/recommendation').status_code, (401,403))
        finally:
            app.dependency_overrides.clear()
