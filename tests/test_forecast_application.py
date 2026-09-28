"""Application tests with fake sessions and a dummy database URL; no DB connections."""
import os
os.environ['DATABASE_URL']='postgresql+asyncpg://test:test@127.0.0.1:1/offline_test'
os.environ['SECRET_KEY']='offline-test-only'
os.environ['ENABLE_SCHEDULER']='false'

import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import tempfile
import unittest
from unittest.mock import patch,AsyncMock
from types import SimpleNamespace
import numpy as np
import pandas as pd
from app.ml.level_forecast import LevelModel,compute_level_features,InsufficientData,forecast_state
from app.ml import services,tasks
from scripts.explain_level_results import read_times

NOW=datetime(2026,9,22,12,15,tzinfo=timezone.utc)


class Result:
    def __init__(self,rows): self.rows=rows
    def first(self): return self.rows[0] if self.rows else None
    def all(self): return self.rows


class Session:
    def __init__(self,*results):
        self.results=list(results);self.statements=[];self.commit=AsyncMock()
    async def __aenter__(self): return self
    async def __aexit__(self,*args): pass
    async def exec(self,stmt):
        self.statements.append(stmt)
        return Result(self.results.pop(0) if self.results else [])


def readings(issue):
    times=[issue-timedelta(hours=h) for h in [0,1,3,6]]
    return pd.DataFrame(dict(captured_at=times,created_at=times,water_level=[4.8,4.9,4.7,4.6]))


class ForecastTests(unittest.TestCase):
    def setUp(self):
        services.load_model()
    def test_all_saved_evaluation_predictions_match_artifact(self):
        model=services.get_model()
        self.assertIsNotNone(model)
        import hashlib
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        path = Path('data/level_training_table.csv')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), model.metadata['training_table_sha256'])
        table = pd.read_csv(path)
        for col in ['forecast_at', 'target_at', 'target_captured_at', 'target_arrived_at']:
            table[col] = pd.to_datetime(table[col], utc=True)
        cutoff = pd.Timestamp(model.metadata['training_cutoff_utc'])
        train = table[(table.forecast_at < cutoff) & (table.target_at < cutoff)
                      & (table.target_captured_at < cutoff) & (table.target_arrived_at < cutoff)]
        test = table[table.forecast_at >= cutoff]
        features = model.metadata['features']
        independent = LinearRegression().fit(train[features], train.level_2h - train.level_now)
        expected = test.level_now.to_numpy() + independent.predict(test[features])
        actual = np.array([model.predict(row) for _, row in test.iterrows()])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        self.assertEqual(len(test), model.metadata['evaluation_rows'])
        self.assertAlmostEqual(mean_absolute_error(test.level_2h, actual), model.metadata['evaluation_metrics']['mae_m'])
        self.assertAlmostEqual(np.sqrt(mean_squared_error(test.level_2h, actual)), model.metadata['evaluation_metrics']['rmse_m'])
    def test_runtime_rejects_late_and_missing_measurements(self):
        df=readings(NOW)
        df.loc[1,'created_at']=NOW+timedelta(minutes=1)
        with self.assertRaises(InsufficientData):compute_level_features(df,NOW)
        with self.assertRaises(InsufficientData):compute_level_features(readings(NOW).iloc[:3],NOW)
    def test_capture_after_anchor_is_not_a_replacement(self):
        df=readings(NOW)
        df.loc[1,'captured_at']+=timedelta(minutes=1)
        df.loc[1,'created_at']+=timedelta(minutes=1)
        with self.assertRaises(InsufficientData):compute_level_features(df,NOW)
    def test_latest_input_stale(self):
        with self.assertRaises(InsufficientData):compute_level_features(readings(NOW-timedelta(minutes=36)),NOW)
    def test_invalid_artifact_disables_previously_loaded_model(self):
        self.assertIsNotNone(services.get_model())
        with patch.object(services.settings,'level_model_path','/tmp/does-not-exist-boresense.json'):
            with self.assertLogs(services.logger,level='ERROR'):services.load_model()
        self.assertIsNone(services.get_model())
    def test_version_detects_coefficient_change(self):
        m=services.get_model().metadata.copy();m['intercept']+=.1
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Model content/version mismatch'):LevelModel.load(p)
    def test_forecast_stale_after_missed_hour_or_data_outage(self):
        self.assertEqual(forecast_state(NOW,NOW.replace(minute=0),NOW+timedelta(hours=1),NOW),'fresh')
        self.assertEqual(forecast_state(NOW,NOW-timedelta(hours=1),NOW+timedelta(hours=1),NOW),'stale')
        self.assertEqual(forecast_state(NOW,NOW.replace(minute=0),NOW+timedelta(hours=1),NOW-timedelta(minutes=36)),'stale')
        self.assertEqual(forecast_state(NOW,None,None,NOW),'unavailable')


class ForecastServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):services.load_model()
    async def test_ownership_is_required_even_when_model_unavailable(self):
        with patch.object(services,'_model',None):
            with self.assertRaises(ValueError):await services.get_prediction_status(2,8,Session([]))
    async def test_stale_status_hides_old_prediction_value(self):
        pred=SimpleNamespace(created_at=NOW-timedelta(hours=2),predicted_for=NOW,predicted_level_2h=4.7)
        latest=SimpleNamespace(captured_at=NOW,water_level=4.8)
        with patch.object(services,'datetime') as clock:
            clock.now.return_value=NOW
            result=await services.get_prediction_status(2,8,Session([object()],[latest],[pred]))
        self.assertEqual(result['status'],'stale');self.assertIsNone(result['predicted_level_2h'])
        self.assertEqual(result['current_level'],4.8)
    async def test_no_model_for_other_borehole(self):
        result=await services.get_prediction_status(99,8,Session([object()]))
        self.assertEqual(result['status'],'unavailable')
    async def test_chart_future_actual_null_and_missing_hour_gap(self):
        meta=services.get_model().metadata
        def pred(target):return SimpleNamespace(predicted_for=target,created_at=target-timedelta(hours=2),predicted_level_2h=4.8,model_version=meta['model_version'])
        hour=NOW.replace(minute=0)
        rows=[pred(hour-timedelta(hours=1)),pred(hour+timedelta(hours=1))]
        session=Session([object()],rows,[(NOW,4.9)])
        with patch.object(services,'datetime') as clock:
            clock.now.return_value=NOW
            result=await services.get_prediction_chart(2,8,session)
        self.assertEqual(len(result),3)
        self.assertIsNone(result[1]['predicted'])
        self.assertIsNone(result[2]['actual'])
        self.assertTrue(all(p['confidence'] is None for p in result))
        self.assertIn('prediction.model_version',str(session.statements[1]))
    async def test_job_reads_correct_sensor_and_writes_version_once(self):
        issue=NOW.replace(minute=0)
        df=readings(issue)
        session=Session(list(df.itertuples(index=False,name=None)))
        with patch.object(tasks,'async_session_maker',return_value=session),patch.object(tasks,'datetime') as clock:
            clock.now.return_value=NOW
            await tasks.run_inference_job()
        self.assertEqual(len(session.statements),2)
        query=session.statements[0].compile().params
        self.assertIn(2,query.values());self.assertIn(4,query.values())
        values=session.statements[1].compile().params
        self.assertEqual(values['model_version'],services.get_model().metadata['model_version'])
        self.assertIsNone(values['confidence_score'])
        self.assertIn('ON CONFLICT',str(session.statements[1]))
        session.commit.assert_awaited_once()
    async def test_missing_history_never_writes_prediction(self):
        session=Session([])
        with patch.object(tasks,'async_session_maker',return_value=session):await tasks.run_inference_job()
        self.assertEqual(len(session.statements),1);session.commit.assert_not_awaited()


class ForecastApiTests(unittest.TestCase):
    def test_authenticated_status_contract_and_ownership(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.core.database import get_session
        from app.auth.dependencies import get_current_user
        services.load_model()
        def user():return SimpleNamespace(id=8)
        async def session():yield Session([object()],[],[])
        app.dependency_overrides[get_current_user]=user
        app.dependency_overrides[get_session]=session
        try:
            with TestClient(app) as client:
                response=client.get('/api/predictions/2/status')
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['data']['status'],'unavailable')
                self.assertIsNone(response.json()['data']['predicted_level_2h'])
                async def no_access():yield Session([])
                app.dependency_overrides[get_session]=no_access
                self.assertEqual(client.get('/api/predictions/2/status').status_code,404)
                self.assertEqual(client.get('/api/predictions/2/chart').status_code,404)
        finally:
            app.dependency_overrides.clear()
