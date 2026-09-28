"""Reproduce the frozen thesis experiment and export evidence; no database access."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from scripts.compare_level_only import build_examples, available_before, digest, metrics
from scripts.test_rainfall_and_response import forecast, BASE, CSV, ARCHIVE
from scripts.explain_level_results import read_times

LABELS = {'level':'Level only', 'level_flow':'Level + flow',
          'level_rain_api':'Level + rainfall/API', 'level_flow_rain_api':'Level + flow + rainfall/API'}


def evidence_rows(summary, predictions):
    rows = []
    for r in summary['results']:
        calculated = metrics(predictions.actual.to_numpy(), predictions[r['key']].to_numpy())
        for metric, value in calculated.items():
            if not np.isclose(value, r['test_'+metric], rtol=1e-10, atol=1e-12):
                raise ValueError('Saved metric does not match saved predictions: '+r['key'])
        rows.append({'inputs': LABELS[r['inputs']], 'model':r['model'],
                     'api_memory_hours':r.get('tau_h'), 'cv_mae_cm':100*r['cv_mae_m'],
                     'evaluation_mae_cm':100*calculated['mae_m'],
                     'evaluation_rmse_cm':100*calculated['rmse_m'], 'prediction_column':r['key']})
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    for extension in ['png','pdf']:
        fig.savefig(output/f'{name}.{extension}',dpi=300,bbox_inches='tight')


def render(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    predictions = read_times(output/'forecast/predictions.csv')
    rows = evidence_rows(summary,predictions)
    rows.to_csv(output/'comparison.csv',index=False)
    settings = [{k:r.get(k) for k in ['key','inputs','model','tau_h','params']} for r in summary['results']]
    (output/'selected_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(10,6))
    labels=[f"{r.inputs} — {r.model}" for r in rows.itertuples()]
    y=np.arange(len(rows)); width=.36
    ax.barh(y-width/2,rows.cv_mae_cm,height=width,label='Development CV MAE',color='#527fa5')
    ax.barh(y+width/2,rows.evaluation_mae_cm,height=width,label='Evaluation MAE',color='#d8903e')
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel('Mean absolute error (cm); lower is better')
    ax.set_title('Two-hour water-level forecasting: same examples for every method',pad=42)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2); ax.grid(axis='x',alpha=.2)
    save_figure(fig,output,'01_model_comparison'); plt.close(fig)
    chosen=summary['selected_by_cv']
    # Reindex hourly so lines break across unavailable periods instead of hiding gaps.
    p=predictions.set_index('forecast_at').sort_index()
    p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='h'))
    fig,ax=plt.subplots(figsize=(11,4.5))
    for col,label,color in [('actual','Observed target level','#202020'),(chosen,'Selected model','#287caf'),('persistence','Persistence','#cb8739')]:
        ax.plot(p.index,p[col],label=label,color=color,linewidth=1.3,alpha=.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M',tz=ZoneInfo('Africa/Lagos')))
    ax.set_xlabel('Forecast issue time (WAT); observation is approximately two hours later')
    ax.set_ylabel('Height above sensor (m)'); ax.set_title('Entire evaluation period — gaps retained')
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig,output,'02_forecast_timeline'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,5))
    ax.scatter(predictions.actual,predictions[chosen],s=23,alpha=.55,color='#287caf')
    lo=min(predictions.actual.min(),predictions[chosen].min())-.03
    hi=max(predictions.actual.max(),predictions[chosen].max())+.03
    ax.plot([lo,hi],[lo,hi],'--',color='gray',label='Perfect prediction')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed level (m)',ylabel='Predicted level (m)',title='CV-selected model: all 105 evaluation examples')
    ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.2)
    save_figure(fig,output,'03_prediction_scatter'); plt.close(fig)
    table=rows.drop(columns='prediction_column').to_html(index=False,float_format=lambda v:f'{v:.2f}',na_rep='—',border=0)
    note='Exploratory comparison on a previously inspected evaluation period. Archived rainfall is modelled weather; original publication availability was not verified. Selection uses development CV, not evaluation scores.'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>BoreSense thesis evidence</title>
<style>body{{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#203040}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#edf2f6}}img{{max-width:100%}}.note{{background:#fff4d9;padding:18px}}@media print{{body{{margin:0}}img{{break-inside:avoid}}}}</style>
<h1>BoreSense: reproducible model comparison</h1><p>326 development examples · 105 evaluation examples · two-hour forecast</p>
<p class="note">{note}</p><p>Selected by development CV: <strong>{html.escape(chosen)}</strong></p>{table}
<h2>Model comparison</h2><img src="01_model_comparison.png" alt="CV and evaluation errors for all methods">
<h2>Forecast behaviour</h2><img src="02_forecast_timeline.png" alt="Observed and predicted levels over the full evaluation period">
<img src="03_prediction_scatter.png" alt="Observed versus predicted levels">
<p>See README.md for methods, figure captions, reproduction and limitations. Numerical evidence: comparison.csv, forecast/predictions.csv, forecast/forest_grid_search.csv (forest runs), selected_settings.json and manifest.json.</p></html>'''
    (output/'index.html').write_text(page)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',choices=['all','forest','linear'],default='all')
    parser.add_argument('--csv-dir',type=Path,default=CSV)
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory to preserve evidence.')
    # Import plotting before expensive training; no automatic dependency installation.
    import matplotlib
    start=datetime.now(timezone.utc).isoformat()
    protected=[args.csv_dir/n for n in ['water_level_reading.csv','flow_reading.csv','weather.csv']]
    protected += [p for p in Path('analysis').rglob('*') if p.is_file()]
    protected += [args.archive/'response.json',args.archive/'request.json']
    protected=list(dict.fromkeys(p.resolve() for p in protected))
    before={str(p):digest(p) for p in protected}
    original=json.loads((BASE/'summary.json').read_text())
    for old,expected in original['input_sha256_before'].items():
        if digest(args.csv_dir/Path(old).name)!=expected:
            raise ValueError('This command reproduces the frozen dataset. Changed exports require a separate prospective experiment.')
    # Reconstruct level examples from raw bytes using the original shared functions.
    table,excluded,grid=build_examples(pd.read_csv(args.csv_dir/'water_level_reading.csv'),2,4)
    cutoff=grid[int(len(grid)*.8)]
    if str(cutoff)!=original['test_cutoff_utc']:
        raise ValueError('Unexpected split change')
    train=available_before(table,cutoff); test=table.forecast_at>=cutoff
    table['partition']=np.where(test,'test',np.where(train,'development','boundary_excluded'))
    reference=read_times(BASE/'examples.csv')
    pd.testing.assert_frame_equal(table.reset_index(drop=True),reference.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
    args.output.mkdir(parents=True,exist_ok=False)
    rebuilt=args.output/'rebuilt_level_examples'; rebuilt.mkdir()
    table.to_csv(rebuilt/'examples.csv',index=False); excluded.to_csv(rebuilt/'excluded_hours.csv',index=False)
    (rebuilt/'summary.json').write_text(json.dumps(original,indent=2)+'\n')
    models=('linear','forest') if args.models=='all' else (args.models,)
    log=io.StringIO()
    print('Rebuilt original examples from unchanged CSVs. Training '+args.models+'; please wait.',flush=True)
    with redirect_stdout(log):
        summary=forecast(args.output/'forecast',base=rebuilt,csv_dir=args.csv_dir,archive=args.archive,models=models)
    rows=render(args.output,summary)
    after={str(p):digest(p) for p in protected}
    if before!=after:
        raise RuntimeError('A protected file changed during the experiment')
    readme=f'''# Thesis evidence package

This run actually refitted the requested models. It did not merely copy scores.
Original exports, existing analysis, application and firmware are unchanged.
The command refuses changed source CSVs and existing output directories.

## Read and reproduce

Open index.html for a screenshot-friendly page. Use PNG figures (300 dpi) in Word,
or PDFs for vector figures. comparison.csv stores unrounded metrics in centimetres.
forecast/predictions.csv stores timestamps, actual levels and predictions in metres.
Metrics were independently recalculated from those saved predictions when exporting.
selected_settings.json includes forest settings, API memory and linear coefficients.
forecast/summary.json contains all development candidates; forest_grid_search.csv
contains every forest parameter combination and fold score when forests were run.
manifest.json records versions, source and code hashes, timestamps and output hashes.
run.log contains a readable terminal result table. Partial runs without a manifest
must not be treated as complete evidence packages.

From repository root, install the optional plotting dependency once:
`uv pip install --python .venv/bin/python -r requirements-thesis.txt`

Full comparison (choose an UNUSED output directory):
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m scripts.thesis_evidence --output analysis/thesis_reproduction`

Random Forest demonstration with persistence baseline and identical preparation:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m scripts.thesis_evidence --models forest --output analysis/forest_demonstration`

Use --models linear for linear-only runs. A subset run selects only among its own
candidates, and must not be presented as the full model comparison. Current run:
**{args.models}**. Selected: **{summary['selected_by_cv']}**.

## Frozen methods

One well, level sensor 4 and flow sensor 5, borehole 2. Predict height above the fixed
sensor approximately two hours ahead. Original matching tolerance is ±35 minutes
for the outcome; inputs are last observations at/before anchors within 35 minutes,
available by forecast issue time. No interpolation or synthetic observations.
Current level and changes over 1, 3 and 6 hours are the level features.
Last 20% of original hourly opportunities are evaluation, beginning {cutoff}.
Training labels crossing the boundary are excluded. Three expanding calendar-time
CV folds purge training labels not yet captured/received at the boundary.
326 development, 105 evaluation, two boundary exclusions from 433 eligible rows.

Persistence carries the latest level forward. Linear regression uses ordinary
least squares. Forest grid: 100/300 trees, depth 3/6/unlimited, leaf minimum 1/3/8,
random seed 42. Mean fold MAE determines settings and the provisional winner.
Rain-enabled models also search API e-folding memories 24/48/96/168 hours.
Rain features: hourly precipitation, trailing 24h and 72h totals, API. API[t] =
exp(-1/tau)*API[t-1] + rainfall[t], initialised July 15 with over five longest
memory times before evaluation examples begin. All weather feature timestamps are
at most forecast time minus one hour. This assumed lag does not verify publication
availability. Archived ECMWF IFS 0.25-degree weather is gridded model output, not
measured farm rainfall. Raw provider response and request are preserved.
Flow features: recorded volume over 1h/24h, recorded flow minutes over 1h, and
24h level coverage. Rate integration uses preceding intervals up to two minutes,
otherwise a nominal minute; it does not bridge long gaps. Zero recorded volume
is not proof that no pumping happened. Sensor capture AND upload times are checked.

## Figure captions

1. Mean cross-validation and evaluation absolute errors for two-hour predictions;
all methods within this run use the same examples. Lower is better; bars are not
confidence intervals. Selection is based on cross-validation.
2. Observed target levels, CV-selected predictions and persistence across the full
evaluation period. Horizontal axis is forecast issue time in WAT. Observations
refer to approximately two hours later. Missing hours break the plotted lines.
3. Observed versus predicted level for the CV-selected model. Dashed line indicates
perfect agreement. Each point is one evaluation example; examples are correlated.

## Chapter 4 interpretation

These follow-up experiments reused a previously inspected evaluation period.
They are exploratory, not independent prospective validation. A modest gain over
persistence should not be described as universal superiority. Rain/flow failing
to improve this experiment does not demonstrate absence of hydrological effects.
The earlier weather-SNAPSHOT experiment is separate and has 286 development rows;
do not combine its scores into this common-cohort table. Its evidence remains in
analysis/added_inputs_2026-09-22. It includes temperature/humidity snapshots, whereas
this main comparison tests rainfall and antecedent wetness explicitly.

The separate pumping-response experiment remains in
analysis/rainfall_response_2026-09-22/pumping and its accompanying report. Its target
is endpoint drawdown, not two-hour level. Do not combine its errors with this table
or describe that relationship as safe volume. Existing application integration and
fresh-data confirmation are still pending; no deployed-model claim is made here.

References and detailed engineering assumptions are in
analysis/rainfall_response_2026-09-22/report.md.
'''
    (args.output/'README.md').write_text(readme)
    terminal=rows.drop(columns='prediction_column').to_string(index=False,float_format=lambda v:f'{v:.3f}')
    (args.output/'run.log').write_text(log.getvalue()+'\n'+terminal+'\nSelected by CV: '+summary['selected_by_cv']+'\nProtected files unchanged.\n')
    outputs={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file()}
    code={str(p):digest(p) for p in Path('scripts').glob('*.py')}
    manifest=dict(started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__),
                  protected_before=before,protected_after=after,protected_unchanged=True,code_sha256=code,output_sha256=outputs,
                  models=args.models,selected_by_cv=summary['selected_by_cv'],status=summary['status'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(terminal+'\nSelected by development CV: '+summary['selected_by_cv']+'\nEvidence: '+str(args.output.resolve()),flush=True)


if __name__=='__main__':
    main()
