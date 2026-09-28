"""Offline exploratory rainfall/API and event-response comparisons; no app imports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from scripts.compare_level_only import FEATURES, GRID, chronological_splits, digest, metrics
from scripts.compare_added_inputs import FLOW_FEATURES, flow_features
from scripts.explain_level_results import read_times

TAUS = [24, 48, 96, 168]
BASE = Path('analysis/level_only_2026-09-22')
ARCHIVE = Path('analysis/rainfall_source_2026-09-22')
EVENTS = Path('analysis/pumping_events_2026-09-22/events.csv')
CSV = Path('/mnt/c/Users/Ysf/Downloads')


def rainfall_features(payload):
    h = payload['hourly']
    times = pd.DatetimeIndex(pd.to_datetime(h['time'], utc=True))
    rain = np.asarray(h['precipitation'], dtype=float)
    if len(times) != len(rain) or not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError('Invalid rainfall; do not repair automatically')
    if times.has_duplicates or not ((times[1:]-times[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError('Rainfall must be consecutive hourly values')
    if payload['hourly_units']['precipitation'] != 'mm':
        raise ValueError('Unexpected rainfall units')
    out = pd.DataFrame({'rain_1h_mm': rain}, index=times)
    for hours in [24, 72]:
        out[f'rain_{hours}h_mm'] = out.rain_1h_mm.rolling(hours, min_periods=hours).sum()
    for tau in TAUS:
        state, values = 0., []
        for p in rain:
            state = np.exp(-1/tau)*state+p
            values.append(state)
        out[f'api_{tau}h_mm'] = values
    return out


def forecast(output, base=BASE, csv_dir=CSV, archive=ARCHIVE, models=("linear", "forest")):
    original = json.loads((base/'summary.json').read_text())
    table = read_times(base/'examples.csv')
    level, flow = [read_times(csv_dir/name) for name in ['water_level_reading.csv', 'flow_reading.csv']]
    level = level.loc[(level.borehole_id == 2) & (level.sensor_id == 4)]
    flow = flow.loc[(flow.borehole_id == 2) & (flow.sensor_id == 5)]
    rain = rainfall_features(json.loads((archive/'response.json').read_text()))
    if table.forecast_at.min()-rain.index.min() < pd.Timedelta(hours=5*max(TAUS)):
        raise ValueError('Insufficient API warmup')
    extra = []
    for t in table.forecast_at:
        # Deliberate 1h lag, not a claim of historical publication availability.
        stamp = t-pd.Timedelta(hours=1)
        r = rain.loc[stamp].to_dict()
        if not np.isfinite(list(r.values())).all():
            raise ValueError('Missing rainfall history')
        extra.append({**r, **flow_features(flow, level, t), 'rain_latest_at': stamp})
    table = pd.concat([table.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    dev = table.loc[table.partition == 'development'].reset_index(drop=True)
    test = table.loc[table.partition == 'test'].copy()
    excluded = read_times(base/'excluded_hours.csv')
    splits, folds = chronological_splits(dev, min(table.forecast_at.min(), excluded.forecast_at.min()),
                                         pd.Timestamp(original['test_cutoff_utc']))
    candidates, fitted, grids = [], {}, []
    for name, with_flow, with_rain in [('level',False,False), ('level_flow',True,False),
                                      ('level_rain_api',False,True), ('level_flow_rain_api',True,True)]:
        for tau in TAUS if with_rain else [None]:
            cols = FEATURES + (FLOW_FEATURES if with_flow else [])
            if with_rain:
                cols += ['rain_1h_mm', 'rain_24h_mm', 'rain_72h_mm', f'api_{tau}h_mm']
            for kind in models:
                key = f'{name}_{kind}_{tau}'
                if kind == 'linear':
                    scores = [metrics(dev.iloc[v].actual, LinearRegression().fit(dev.iloc[tr][cols], dev.iloc[tr].actual).predict(dev.iloc[v][cols]))['mae_m'] for tr,v in splits]
                    model = LinearRegression().fit(dev[cols], dev.actual)
                    score, params = float(np.mean(scores)), {'intercept':float(model.intercept_), 'coefficients':dict(zip(cols,model.coef_.tolist()))}
                else:
                    search = GridSearchCV(RandomForestRegressor(random_state=42,n_jobs=1), GRID, cv=splits,
                                          scoring='neg_mean_absolute_error',n_jobs=1,error_score='raise')
                    search.fit(dev[cols],dev.actual)
                    grid = pd.DataFrame(search.cv_results_)
                    grid['feature_set'], grid['tau_h'] = name, tau
                    grids.append(grid)
                    model, score, params = search.best_estimator_, -search.best_score_, search.best_params_
                fitted[key] = (model,cols)
                candidates.append(dict(key=key,inputs=name,model=kind,tau_h=tau,cv_mae_m=score,params=params))
        print('Completed rainfall development search:', name, flush=True)
    results = []
    for name in ['level','level_flow','level_rain_api','level_flow_rain_api']:
        for kind in models:
            results.append(min((c for c in candidates if c['inputs']==name and c['model']==kind),key=lambda c:c['cv_mae_m']).copy())
    baseline = dict(key='persistence',inputs='level',model='persistence',tau_h=None,
                    cv_mae_m=float(np.mean([metrics(dev.iloc[v].actual,dev.iloc[v].level_now)['mae_m'] for _,v in splits])))
    results.insert(0,baseline)
    chosen = min(results,key=lambda c:c['cv_mae_m'])['key']
    for r in results:
        pred = test.level_now.to_numpy() if r['key']=='persistence' else fitted[r['key']][0].predict(test[fitted[r['key']][1]])
        test[r['key']] = pred
        r.update({'test_'+k:v for k,v in metrics(test.actual,pred).items()})
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'examples.csv',index=False)
    test.to_csv(output/'predictions.csv',index=False)
    if grids:
        pd.concat(grids).to_csv(output/'forest_grid_search.csv',index=False)
    summary = dict(status='exploratory_reused_test_period_and_reconstructed_weather',development_rows=len(dev),test_rows=len(test),
                   selected_by_cv=chosen,folds=folds,results=results,development_candidates=candidates)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def volume_until(flow, start, end):
    f = flow.loc[(flow.captured_at > start) & (flow.captured_at <= end+pd.Timedelta(minutes=2))].sort_values('captured_at').copy()
    dt = f.captured_at.diff().dt.total_seconds()
    dt = dt.where((dt>0)&(dt<=120),60.)
    left = (f.captured_at-pd.to_timedelta(dt,unit='s')).clip(lower=start)
    right = f.captured_at.clip(upper=end)
    minutes = ((right-left).dt.total_seconds()/60).clip(lower=0)
    return float((minutes*f.abstraction_rate).sum()), float(minutes.sum())


def response(output):
    events = read_times(EVENTS)
    events = events.loc[events.included_in_relationship].copy()
    events['estimated_start_utc'] = pd.to_datetime(events.estimated_start_utc,utc=True)
    events = events.sort_values('estimated_start_utc').reset_index(drop=True)
    flow = read_times(CSV/'flow_reading.csv')
    flow = flow.loc[(flow.borehole_id==2)&(flow.sensor_id==5)]
    for i,e in events.iterrows():
        v,mins = volume_until(flow,e.estimated_start_utc,e.end_level_at)
        events.loc[i,'aligned_volume_1000l'] = v/1000
        events.loc[i,'aligned_duration_min'] = (e.end_level_at-e.estimated_start_utc).total_seconds()/60
        events.loc[i,'aligned_rate_lpm'] = v/mins
    if len(events)!=18:
        raise ValueError('Reconsider predefined event split if dataset changes')
    dev,test = events.iloc[:12].copy(),events.iloc[12:].copy()
    splits = [(np.arange(n),np.arange(n,n+2)) for n in [6,8,10]]
    configs = {'training_mean':[], 'volume_origin':['aligned_volume_1000l'],
               'volume_linear':['aligned_volume_1000l'],
               'volume_start_level':['aligned_volume_1000l','pre_level_m'],
               'rate_duration_start_level':['aligned_rate_lpm','aligned_duration_min','pre_level_m']}
    def fit(name,part):
        if name=='training_mean':
            return float(part.drawdown_at_end_m.mean())
        return LinearRegression(fit_intercept=name!='volume_origin').fit(part[configs[name]],part.drawdown_at_end_m)
    def predict(name,model,part):
        return np.full(len(part),model) if name=='training_mean' else model.predict(part[configs[name]])
    results,models = [],{}
    for name in configs:
        scores = [metrics(dev.iloc[v].drawdown_at_end_m,predict(name,fit(name,dev.iloc[tr]),dev.iloc[v]))['mae_m'] for tr,v in splits]
        models[name] = fit(name,dev)
        r = dict(model=name,features=configs[name],cv_mae_m=float(np.mean(scores)),cv_fold_mae_m=scores)
        if name!='training_mean':
            r.update(intercept_m=float(models[name].intercept_),coefficients=dict(zip(configs[name],models[name].coef_.tolist())))
        results.append(r)
    chosen = min(results,key=lambda r:r['cv_mae_m'])['model']
    for r in results:
        pred = predict(r['model'],models[r['model']],test)
        test[r['model']] = pred
        r.update({'test_'+k:v for k,v in metrics(test.drawdown_at_end_m,pred).items()})
    output.mkdir(parents=True,exist_ok=False)
    events.to_csv(output/'aligned_events.csv',index=False)
    test.to_csv(output/'predictions.csv',index=False)
    summary = dict(status='exploratory_chronological_event_validation_previously_inspected_events',
                   development_events=12,test_events=6,selected_by_cv=chosen,results=results,
                   target='pre_level_m minus end_level_m; volume integrated only up to end_level_at',
                   test_event_ids=test.event_id.tolist())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('analysis/rainfall_response_2026-09-22'))
    output = parser.parse_args().output
    if output.exists():
        raise ValueError('Use a new output directory')
    protected = [CSV/n for n in ['water_level_reading.csv','flow_reading.csv','weather.csv']]
    protected += [p for p in Path('analysis').rglob('*') if p.is_file()]
    before = {str(p):digest(p) for p in protected}
    expected = json.loads((BASE/'summary.json').read_text())['input_sha256_before']
    for p in protected[:3]:
        if digest(p)!=expected[str(p)]:
            raise ValueError('Source changed since original experiment')
    forecasts = forecast(output/'forecast')
    responses = response(output/'pumping')
    after = {str(p):digest(p) for p in protected}
    if before!=after:
        raise RuntimeError('Protected files changed')
    (output/'integrity.json').write_text(json.dumps(dict(before=before,after=after,unchanged=True),indent=2)+'\n')
    print(json.dumps({'forecast_results':forecasts['results'],'response_results':responses['results']},indent=2))


if __name__=='__main__':
    main()
