"""Retrospective association of recorded flow events with level drawdown/recovery.

Reads source CSVs only. No app imports or database access. No safe-yield estimator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events_table(level, flow):
    level = level.sort_values("captured_at").reset_index(drop=True)
    flow = flow.sort_values("captured_at").reset_index(drop=True)
    positive = flow.loc[flow.abstraction_rate > 0].copy()
    positive["event_id"] = (positive.captured_at.diff() > pd.Timedelta(minutes=10)).cumsum()+1
    groups = [g.copy() for _, g in positive.groupby("event_id")]
    results, traces = [], []
    for i, g in enumerate(groups):
        first, end = g.captured_at.iloc[0], g.captured_at.iloc[-1]
        start = first-pd.Timedelta(minutes=1)  # first rate is a preceding-window average
        next_start = groups[i+1].captured_at.iloc[0]-pd.Timedelta(minutes=1) if i+1<len(groups) else None
        gaps = g.captured_at.diff().dt.total_seconds()/60
        duration = gaps.where((gaps > 0) & (gaps <= 2), 1.)
        volume = float((g.abstraction_rate*duration).sum())
        row = {"event_id": i+1, "estimated_start_utc": start, "last_flow_utc": end,
               "estimated_start_wat": start.tz_convert("Africa/Lagos"),
               "last_flow_wat": end.tz_convert("Africa/Lagos"), "flow_rows": len(g),
               "elapsed_minutes": (end-start).total_seconds()/60,
               "represented_minutes": float(duration.sum()), "recorded_volume_l": volume,
               "mean_recorded_rate_lpm": volume/float(duration.sum()),
               "max_flow_gap_minutes": float(gaps.max()) if len(g)>1 else None}
        before = level.loc[(level.captured_at <= start) & (level.captured_at >= start-pd.Timedelta(minutes=35))]
        during = level.loc[(level.captured_at >= start) & (level.captured_at <= end)]
        flags = []
        if len(g) < 5:
            flags.append("fewer_than_5_flow_rows")
        if (gaps > 2).any():
            flags.append("flow_gap_over_2min")
        if before.empty:
            flags.append("no_pre_level_within_35min")
        if len(during) < 3:
            flags.append("fewer_than_3_during_levels")
        if not before.empty:
            pre = before.iloc[-1]
            row.update(pre_level_m=float(pre.water_level), pre_level_at=pre.captured_at,
                       pre_age_minutes=(start-pre.captured_at).total_seconds()/60)
        if not during.empty:
            first_level, last_level = during.iloc[0], during.iloc[-1]
            minimum = during.loc[during.water_level.idxmin()]
            row.update(min_level_m=float(minimum.water_level), min_level_at=minimum.captured_at,
                       end_level_m=float(last_level.water_level), end_level_at=last_level.captured_at,
                       during_level_rows=len(during))
            if (first_level.captured_at-start).total_seconds()/60 > 3:
                flags.append("late_first_during_level")
            if (end-last_level.captured_at).total_seconds()/60 > 3:
                flags.append("missing_end_level")
            if (during.captured_at.diff() > pd.Timedelta(minutes=3)).any():
                flags.append("during_level_gap_over_3min")
            if not before.empty:
                drawdown = float(pre.water_level-minimum.water_level)
                row.update(drawdown_to_min_m=drawdown,
                           drawdown_at_end_m=float(pre.water_level-last_level.water_level),
                           drawdown_per_1000l=drawdown/volume*1000 if volume>0 else None)
                if drawdown <= 0:
                    flags.append("no_positive_observed_drawdown")
        for minutes in [15,30,45]:
            target = end+pd.Timedelta(minutes=minutes)
            recovery = level.loc[(level.captured_at > end)
                                 & ((level.captured_at-target).abs() <= pd.Timedelta(minutes=5))]
            if next_start is not None:
                recovery = recovery.loc[recovery.captured_at < next_start]
            if not recovery.empty:
                match = recovery.loc[(recovery.captured_at-target).abs().idxmin()]
                row[f"recovery_{minutes}min_level_m"] = float(match.water_level)
                row[f"recovery_{minutes}min_at"] = match.captured_at
                if row.get("drawdown_at_end_m", 0)>0:
                    row[f"recovery_{minutes}min_fraction"] = (match.water_level-row["end_level_m"])/row["drawdown_at_end_m"]
        row["quality_flags"] = ";".join(flags)
        row["included_in_relationship"] = not flags
        results.append(row)
        trace = level.loc[(level.captured_at >= start-pd.Timedelta(minutes=35))
                          & (level.captured_at <= end+pd.Timedelta(minutes=50))].copy()
        trace["event_id"] = i+1
        trace["minutes_from_estimated_start"] = (trace.captured_at-start).dt.total_seconds()/60
        trace["phase"] = np.where(trace.captured_at < start, "before", np.where(trace.captured_at <= end,"during","after"))
        trace["next_event_has_started"] = False if next_start is None else trace.captured_at >= next_start
        traces.append(trace)
    return pd.DataFrame(results), pd.concat(traces,ignore_index=True) if traces else pd.DataFrame()


def run(csv_dir, output, diameter):
    if output.exists():
        raise ValueError("Choose a new output directory")
    files = [csv_dir/name for name in ["water_level_reading.csv","flow_reading.csv","weather.csv"]]
    before = {str(p):sha(p) for p in files}
    level, flow = [pd.read_csv(p) for p in files[:2]]
    for data, value, sensor in [(level,"water_level",4),(flow,"abstraction_rate",5)]:
        for col in ["captured_at","created_at"]:
            data[col] = pd.to_datetime(data[col],utc=True,format="mixed",errors="raise")
        if data[[value,"captured_at"]].isna().any().any() or not np.isfinite(data[value]).all():
            raise ValueError("Invalid source measurements; no repairs applied")
        if data.duplicated(["sensor_id","captured_at"]).any() or (data[value]<0).any():
            raise ValueError("Duplicate timestamps or negative measurements")
    level=level.loc[(level.borehole_id==2)&(level.sensor_id==4)]
    flow=flow.loc[(flow.borehole_id==2)&(flow.sensor_id==5)]
    table,traces=events_table(level,flow)
    usable=table.loc[table.included_in_relationship].copy()
    if len(usable)<3:
        raise ValueError("Too few sufficiently observed pumping events")
    ratios=usable.drawdown_per_1000l
    x=usable.recorded_volume_l.to_numpy()/1000
    y=usable.drawdown_to_min_m.to_numpy()
    design=np.column_stack([np.ones(len(x)),x])
    coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
    fitted=design@coefficients
    r2=1-float(((y-fitted)**2).sum())/float(((y-y.mean())**2).sum()) if np.var(y)>0 else None
    area=float(np.pi*diameter**2/4)
    summary={"status":"retrospective_descriptive_event_analysis_not_safe_yield",
             "events_total":len(table),"events_in_relationship":len(usable),
             "diameter_m_assumed_internal":diameter,"area_m2":area,
             "drawdown_per_1000l_mean":float(ratios.mean()),"drawdown_per_1000l_median":float(ratios.median()),
             "drawdown_per_1000l_sample_sd":float(ratios.std(ddof=1)),
             "drawdown_per_1000l_min":float(ratios.min()),"drawdown_per_1000l_max":float(ratios.max()),
             "volume_l_range":[float(usable.recorded_volume_l.min()),float(usable.recorded_volume_l.max())],
             "drawdown_m_range":[float(y.min()),float(y.max())],
             "rate_lpm_range":[float(usable.mean_recorded_rate_lpm.min()),float(usable.mean_recorded_rate_lpm.max())],
             "descriptive_ols":{"intercept_m":float(coefficients[0]),"slope_m_per_1000l":float(coefficients[1]),"in_sample_r2":r2},
             "source_hashes_before":before,"source_hashes_after":{str(p):sha(p) for p in files}}
    if summary["source_hashes_before"]!=summary["source_hashes_after"]:
        raise RuntimeError("Source fingerprints changed")
    summary["sources_unchanged"]=True
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/"events.csv",index=False)
    traces.to_csv(output/"level_traces.csv",index=False)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    report="""# Pumping events and measured water-level response

Retrospective descriptive analysis of borehole 2, level sensor 4 and flow sensor 5.
Capture timestamps align the physical measurements; arrival times are retained
but do not exclude buffered observations from retrospective event analysis.
All three original CSVs are unchanged. No database/device/application changes.

## Definitions fixed before calculation

Positive flow readings separated by more than 10 minutes form separate candidate
events. The first capture is an end-of-window timestamp: approximate onset is one
minute earlier. Exact pump-on/off times cannot be recovered from positive-only
flow records. Last positive capture approximates the event end, not a confirmed stop.

Volume is sum(rate × preceding-window duration). Adjacent intervals <=2 minutes
use observed spacing; first readings and longer gaps use one nominal minute.
Never integrate a rate through long missing intervals. These are recorded-volume
estimates, not verified complete event totals.

Pre-level is the last measurement at/before estimated onset within 35 minutes;
it is a pre-pumping REFERENCE, not proof of full hydraulic equilibrium. Drawdown
is pre-level minus minimum observed level during the candidate event. The minimum
can occur before the end: drawdown/whole-event-volume is an event descriptor,
not a fitted physical coefficient. End-level is also retained separately.

Main relationship includes events with >=5 flow rows, no flow gap >2 minutes,
>=3 during-event level rows, first/last levels within 3 minutes of the approximate
boundaries, no internal level gap >3 minutes, a valid pre-level, and positive
observed drawdown. Every event is retained in events.csv with explicit flags.
These are practical completeness rules, not proof an event is perfectly measured.

Recovery readings are nearest to +15/+30/+45 minutes after the final flow sample,
within 5 minutes and before any next detected event. Recovery fraction uses
(post-level − end-level)/(pre-level − end-level), when denominator is positive.
It is not clipped: values above 1 indicate exceeding the pre-event reference.

## Results

"""
    report+=f"Detected {len(table)} candidate events; {len(usable)} meet the main completeness criteria.\n\n"
    report+="| Event | Approximate start (WAT) | Recorded volume (L) | Pre-level (m) | Minimum (m) | Drawdown (m) | Included? |\n|---|---|---:|---:|---:|---:|---|\n"
    def fmt(value):return "—" if pd.isna(value) else f"{value:.3f}"
    for _,r in table.iterrows():
        report+=f"| {r.event_id} | {r.estimated_start_wat.strftime('%Y-%m-%d %H:%M')} | {r.recorded_volume_l:.1f} | {fmt(r.get('pre_level_m'))} | {fmt(r.get('min_level_m'))} | {fmt(r.get('drawdown_to_min_m'))} | {'yes' if r.included_in_relationship else r.quality_flags} |\n"
    report+=f"""
Among included events, observed drawdown per 1000 estimated litres:
- Mean: {ratios.mean():.3f} m; median: {ratios.median():.3f} m.
- Sample SD: {ratios.std(ddof=1):.3f} m; range: {ratios.min():.3f}–{ratios.max():.3f} m.
- Volume range: {x.min()*1000:.1f}–{x.max()*1000:.1f} L.
- Observed drawdown range: {y.min():.3f}–{y.max():.3f} m.

Exploratory straight-line description of these same events:
drawdown_m = {coefficients[0]:.4f} + {coefficients[1]:.4f} × volume_in_thousands_of_litres.
In-sample R² = {r2:.3f}. This is not independently validated predictive performance.
Rate, duration, starting level, recovery and data completeness can all influence
the association. No causal or sustainable-yield coefficient is established.

## Relation to the diameter calculation

Assuming internal diameter {diameter:.2f} m and uniform section gives area {area:.4f} m².
A metre of level change represents {area*1000:.1f} L of change in well-column
storage. Pumped volume includes water supplied during the event; it need not equal
storage loss. Geometry and event analysis are complementary. Do not substitute
the empirical ratio into a safe-volume claim or extrapolate beyond observed events.

## Evidence and reproduction

events.csv contains timestamps, flags, volume, drawdown, end-level and recovery.
level_traces.csv contains actual readings before/during/after each event for plotting;
overlapping trace windows may repeat a source reading under different event IDs.
summary.json retains statistics and source SHA-256 hashes. Missing values remain missing.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m scripts.analyze_pumping_events \\
  --csv-dir /mnt/c/Users/Ysf/Downloads --diameter 0.76 \\
  --output analysis/pumping_events_reproduction
```

Reference: [USGS aquifer-test reporting guidance](https://water.usgs.gov/water-resources/memos/memo.php?id=1403).
These are observations of normal operation, not a controlled aquifer pumping test.
"""
    (output/"report.md").write_text(report)
    print(json.dumps(summary,indent=2),flush=True)
    print(table[["event_id","estimated_start_wat","flow_rows","recorded_volume_l","pre_level_m","min_level_m","drawdown_to_min_m","recovery_45min_level_m","included_in_relationship","quality_flags"]].to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--diameter",type=float,default=.76)
    args=parser.parse_args()
    run(args.csv_dir,args.output,args.diameter)
