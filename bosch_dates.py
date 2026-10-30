"""Chunked Bosch date-quality checks and an exploratory time-split experiment.

Dates remain float64 in their supplied numeric coordinate system. No calendar
origin, physical duration unit, station sequence, or label-availability time is
assumed. No learned preprocessing or target selection occurs in these features.
"""

import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from bosch import (build_features, capacity_metrics, sensor_schema,
                   split_rows, station_columns)


FOCUS_STATIONS = ('L3_S32', 'L3_S33', 'L3_S34', 'L1_S24')


def csv_header(path):
    with Path(path).open(newline='') as stream:
        header = next(csv.reader(stream))
    if len(set(header)) != len(header):
        raise ValueError('Duplicate CSV column names; investigate before parsing.')
    return tuple(header)


def date_schema(columns):
    dates = tuple(c for c in columns if re.fullmatch(r'L\d+_S\d+_D\d+', c))
    if (not dates or 'Id' not in columns or len(set(columns)) != len(columns)
            or set(columns) - set(dates) - {'Id'}):
        raise ValueError('Expected Id and unique Bosch date columns only.')
    return dates


def check_ids(frame, source):
    if frame.empty:
        raise ValueError(f'{source}: no rows loaded.')
    if frame['Id'].isna().any() or not frame['Id'].is_unique:
        raise ValueError(f'{source}: missing or duplicate Id; do not deduplicate silently.')


def summarize_date_chunk(frame, dates, focus_stations=FOCUS_STATIONS):
    """Pure per-part transformation: missingness counts and recorded time ranges.

    A range is max-minus-min over recorded timestamps, NOT certified processing
    time. Multiple station records may be simultaneous or reflect rework.
    """
    if set(date_schema(frame.columns)) != set(dates):
        raise ValueError('Date schema mismatch.')
    if len(dates) > np.iinfo(np.uint16).max:
        raise ValueError('Too many date columns for uint16 counters.')
    groups = station_columns(dates)
    if not set(focus_stations).issubset(groups):
        raise ValueError('A requested focus station is absent from the date schema.')
    n = len(frame)
    start = np.full(n, np.inf)
    end = np.full(n, -np.inf)
    recorded = np.zeros(n, dtype=np.uint16)
    stations_recorded = np.zeros(n, dtype=np.uint16)
    features = {'Id': frame['Id'].to_numpy(dtype=np.int64)}
    station_records = []
    for station, columns in groups.items():
        first = np.full(n, np.inf)
        last = np.full(n, -np.inf)
        count = np.zeros(n, dtype=np.uint16)
        for col in columns:
            values = frame[col].to_numpy(dtype=np.float64, copy=False)
            if np.isinf(values).any():
                raise ValueError(f'Infinite timestamp in {col}; investigate the source.')
            available = ~np.isnan(values)
            count += available
            np.minimum(first, values, out=first, where=available)
            np.maximum(last, values, out=last, where=available)
        present = count > 0
        recorded += count
        stations_recorded += present
        np.minimum(start, first, out=start)
        np.maximum(end, last, out=end)
        first[~present] = np.nan
        last[~present] = np.nan
        span = last - first
        station_records.append({
            'station': station, 'parts_recorded': int(present.sum()),
            'parts_with_multiple_records': int((count > 1).sum()),
            'parts_with_positive_range': int((span > 0).sum()),
        })
        if station in focus_stations:
            features[f'{station}_date_first'] = first
            features[f'{station}_date_last'] = last
    empty = recorded == 0
    start[empty] = np.nan
    end[empty] = np.nan
    features.update(date_first=start, date_last=end, date_span=end - start,
                    date_measurement_count=recorded,
                    date_station_count=stations_recorded)
    return pd.DataFrame(features), pd.DataFrame(station_records).set_index('station')


def read_date_summary(path, chunksize=10_000, nrows=None,
                      focus_stations=FOCUS_STATIONS):
    """Scan once, retaining only compact features and additive quality summaries."""
    dates = date_schema(csv_header(path))
    dtypes = {c: 'float64' for c in dates}
    dtypes['Id'] = 'int64'
    summaries = []
    station_totals = None
    nonmissing = np.zeros(len(dates), dtype=np.int64)
    zeros = np.zeros(len(dates), dtype=np.int64)
    negatives = np.zeros(len(dates), dtype=np.int64)
    minima = np.full(len(dates), np.inf)
    maxima = np.full(len(dates), -np.inf)
    rows = 0
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize, nrows=nrows):
        summary, station = summarize_date_chunk(chunk, dates, focus_stations)
        summaries.append(summary)
        station_totals = station if station_totals is None else station_totals.add(station)
        rows += len(chunk)
        for i, col in enumerate(dates):
            values = chunk[col].to_numpy()
            valid = values[~np.isnan(values)]
            nonmissing[i] += len(valid)
            zeros[i] += np.count_nonzero(valid == 0)
            negatives[i] += np.count_nonzero(valid < 0)
            if len(valid):
                minima[i] = min(minima[i], valid.min())
                maxima[i] = max(maxima[i], valid.max())
    if not summaries:
        raise ValueError('No date rows loaded.')
    summary = pd.concat(summaries, ignore_index=True)
    check_ids(summary, 'date data')  # catches duplicate IDs across chunk boundaries
    minima[nonmissing == 0] = np.nan
    maxima[nonmissing == 0] = np.nan
    profile = pd.DataFrame({'column': dates, 'recorded': nonmissing,
                            'missing': rows - nonmissing,
                            'missing_fraction': 1 - nonmissing / rows,
                            'zero_values': zeros, 'negative_values': negatives,
                            'minimum': minima, 'maximum': maxima})
    station_totals['coverage'] = station_totals['parts_recorded'] / rows
    return summary, profile, station_totals, dates


def read_numeric_counts(path, chunksize=10_000, nrows=None):
    """Stream the existing numeric count features; never hold all sensors in RAM."""
    header = csv_header(path)
    sensors = sensor_schema(header)
    if not {'Id', 'Response'}.issubset(header):
        raise ValueError('Numeric training data require Id and Response.')
    # Parse target in int64 first to avoid wrapping malformed values into int8.
    dtypes = {c: 'float32' for c in sensors}
    dtypes.update(Id='int64', Response='int64')
    chunks = []
    for raw in pd.read_csv(path, dtype=dtypes, chunksize=chunksize, nrows=nrows):
        if not raw['Response'].isin([0, 1]).all():
            raise ValueError('Response must contain only 0 and 1.')
        for col in sensors:
            if np.isinf(raw[col].to_numpy()).any():
                raise ValueError(f'Infinite numeric measurement in {col}.')
        counts = build_features(raw, sensors)
        counts.insert(0, 'Response', raw['Response'].to_numpy(dtype=np.int8))
        counts.insert(0, 'Id', raw['Id'].to_numpy())
        chunks.append(counts)
    if not chunks:
        raise ValueError('No numeric rows loaded.')
    result = pd.concat(chunks, ignore_index=True)
    check_ids(result, 'numeric data')
    return result, sensors


def audit_join(numeric, dates):
    """Audit both directions, then retain all numeric parts in a 1:1 left join."""
    check_ids(numeric, 'numeric data')
    check_ids(dates, 'date data')
    numeric_ids = pd.Index(numeric['Id'])
    date_ids = pd.Index(dates['Id'])
    missing_dates = numeric_ids.difference(date_ids)
    extra_dates = date_ids.difference(numeric_ids)
    joined = numeric.merge(dates, on='Id', how='left', validate='one_to_one',
                           indicator='date_join', sort=False)
    assert len(joined) == len(numeric)
    assert np.array_equal(joined['Id'].to_numpy(), numeric['Id'].to_numpy())
    audit = {'numeric_rows': len(numeric), 'date_rows': len(dates),
             'matched_ids': int(numeric_ids.isin(date_ids).sum()),
             'numeric_only_ids': len(missing_dates), 'date_only_ids': len(extra_dates),
             'joined_rows': len(joined),
             'matched_but_all_dates_missing': int(((joined['date_join'] == 'both') &
                                                 joined['date_first'].isna()).sum())}
    return joined, audit, missing_dates.to_numpy(), extra_dates.to_numpy()


def chronological_split(frame, fractions=(0.64, 0.16, 0.20)):
    """Split by earliest recorded time, keeping ties together and purging overlap.

    Training records must finish strictly before the validation start; validation
    records must finish strictly before the evaluation start. Unknown dates are
    quarantined. No outcome labels are accessed. This cannot prove label maturity.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Need three positive fractions summing to one.')
    start = frame['date_first'].to_numpy(dtype=float)
    end = frame['date_last'].to_numpy(dtype=float)
    known = np.isfinite(start) & np.isfinite(end)
    if (end[known] < start[known]).any():
        raise ValueError('End before start in part date summary.')
    ordered = np.sort(start[known])
    if len(ordered) < 3:
        raise ValueError('Need at least three dated parts.')
    first_cut = ordered[min(int(len(ordered) * fractions[0]), len(ordered) - 1)]
    second_cut = ordered[min(int(len(ordered) * sum(fractions[:2])), len(ordered) - 1)]
    if first_cut >= second_cut:
        raise ValueError('Timestamp ties leave no validation interval; do not split ties.')
    train_window = known & (start < first_cut)
    validation_window = known & (start >= first_cut) & (start < second_cut)
    train = train_window & (end < first_cut)
    validation = validation_window & (end < second_cut)
    evaluation = known & (start >= second_cut)
    splits = {'train': np.flatnonzero(train), 'validation': np.flatnonzero(validation),
              'exploratory_evaluation': np.flatnonzero(evaluation)}
    if any(len(rows) == 0 for rows in splits.values()):
        raise ValueError('Empty temporal partition after tying/purging; inspect date coverage.')
    excluded = {'unknown_dates': np.flatnonzero(~known),
                'overlap_at_validation': np.flatnonzero(train_window & ~train),
                'overlap_at_evaluation': np.flatnonzero(validation_window & ~validation)}
    return splits, excluded, {'validation_start': float(first_cut),
                              'evaluation_start': float(second_cut)}


def random_split_same_cohort(frame, chronological, seed=42):
    """Random comparator uses exactly the retained chronological cohort."""
    kept = np.sort(np.concatenate(list(chronological.values())))
    relative = split_rows(frame['Response'].to_numpy()[kept], seed)
    return {key: kept[rows] for key, rows in relative.items()}


def run_date_experiment(frame, strategies, feature_sets, seed=42):
    """Fixed HGB configuration; choose feature set by validation AP per strategy.

    Evaluation rows are scored only after selection. This is retrospective,
    exploratory end-of-record screening, never a claimed early-warning model.
    """
    labels = frame['Response'].to_numpy()
    comparisons, evaluations, selections = [], [], {}
    for strategy, splits in strategies.items():
        for split, rows in splits.items():
            if set(np.unique(labels[rows])) != {0, 1}:
                raise ValueError(f'{strategy}/{split} requires both classes; enlarge the run.')
        best = None
        for name, columns in feature_sets.items():
            if set(columns) & {'Id', 'Response', 'date_join'}:
                raise ValueError('IDs, labels and join diagnostics cannot be predictors.')
            model = HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
                class_weight='balanced', early_stopping=False, random_state=seed)
            # At most one model's training matrix is materialized at a time.
            with threadpool_limits(limits=4):
                model.fit(frame.loc[:, columns].iloc[splits['train']], labels[splits['train']])
                scores = model.predict_proba(frame.loc[:, columns].iloc[splits['validation']])[:, 1]
            ap = float(average_precision_score(labels[splits['validation']], scores))
            comparisons.append({'strategy': strategy, 'feature_set': name,
                                'feature_count': len(columns), 'validation_ap': ap})
            if best is None or ap > best['validation_ap']:
                best = {'estimator': model, 'columns': tuple(columns), 'feature_set': name,
                        'validation_ap': ap}
        test = splits['exploratory_evaluation']
        with threadpool_limits(limits=4):
            scores = best['estimator'].predict_proba(frame.loc[:, best['columns']].iloc[test])[:, 1]
        evaluations.append({'strategy': strategy, 'selected_features': best['feature_set'],
                            'parts': len(test), 'defects': int(labels[test].sum()),
                            'prevalence': float(labels[test].mean()),
                            'average_precision': float(average_precision_score(labels[test], scores)),
                            **capacity_metrics(labels[test], scores, capacity=0.01)})
        selections[strategy] = {**best, 'evaluation_scores': scores}
    return pd.DataFrame(comparisons), pd.DataFrame(evaluations), selections
