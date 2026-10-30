"""Small, reusable helpers for the Bosch numeric-data notebook.

No target-dependent operation belongs in build_features. Fit selection and
imputation on training rows only; preserve the returned schema for inference.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def resolve_data_path(path=None):
    """An explicit path must exist; otherwise try the documented local locations."""
    candidates = ([Path(path)] if path else [
        Path('/data/train_numeric.csv'),
        Path(__file__).resolve().parent / 'data/train_numeric.csv',
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Numeric CSV not found. Checked: {candidates}')


def sensor_schema(columns):
    """Freeze sensor names and station membership from a CSV header, never y."""
    sensors = tuple(c for c in columns if re.fullmatch(r'L\d+_S\d+_F\d+', c))
    if not sensors or len(set(columns)) != len(columns):
        raise ValueError('Expected unique column names and Bosch sensor columns.')
    unexpected = set(columns) - set(sensors) - {'Id', 'Response'}
    if unexpected:
        raise ValueError(f'Unexpected raw columns: {sorted(unexpected)}')
    return sensors


def station_columns(sensors):
    stations = {}
    for col in sensors:
        stations.setdefault('_'.join(col.split('_')[:2]), []).append(col)
    return {station: tuple(stations[station]) for station in sorted(stations)}


def load_numeric(path, nrows=None):
    """Keep the original float32/int64/int8 loading policy without a raw copy."""
    columns = pd.read_csv(path, nrows=0).columns
    sensors = sensor_schema(columns)
    if 'Id' not in columns:
        raise ValueError('Id is required.')
    dtypes = {c: 'float32' for c in sensors}
    dtypes.update({'Id': 'int64', 'Response': 'int8'})
    frame = pd.read_csv(path, dtype=dtypes, nrows=nrows)
    validate_raw(frame, sensors, require_target='Response' in columns)
    return frame, sensors


def validate_raw(frame, sensors, require_target=True):
    if frame.empty or not frame.index.is_unique:
        raise ValueError('Rows must be nonempty and have a unique index.')
    if 'Id' not in frame or frame['Id'].isna().any() or not frame['Id'].is_unique:
        raise ValueError('Id must be present, nonmissing, and unique.')
    if set(sensor_schema(frame.columns)) != set(sensors):
        raise ValueError('Raw sensor schema differs from the training schema.')
    if require_target and ('Response' not in frame or
                           not frame['Response'].isin([0, 1]).all()):
        raise ValueError('Response must contain only 0 and 1.')
    # A column at a time avoids allocating a full sensor-sized Boolean matrix.
    for col in sensors:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'{col} must be numeric.')
        if np.isinf(frame[col].to_numpy()).any():
            raise ValueError(f'Infinite values in {col}; investigate before modeling.')


def split_rows(y, seed=42):
    """Preserve original 80/20 then 80/20 split: 64/16/20 overall."""
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both binary classes are required for splitting.')
    rows = np.arange(len(y))
    development, evaluation = train_test_split(rows, test_size=0.2,
                                              stratify=y, random_state=seed)
    train, validation = train_test_split(
        development, test_size=0.2, stratify=y[development], random_state=seed)
    splits = {'train': np.sort(train), 'validation': np.sort(validation),
              'exploratory_evaluation': np.sort(evaluation)}
    combined = np.concatenate(list(splits.values()))
    if len(np.unique(combined)) != len(y):
        raise AssertionError('Split membership is not disjoint and exhaustive.')
    for positions in splits.values():
        if set(np.unique(y[positions])) != {0, 1}:
            raise ValueError('Too few positive examples in a split; use more rows.')
    return splits


def missingness_profile(frame, sensors):
    return pd.Series({c: frame[c].isna().mean() * 100 for c in sensors},
                     name='missing_percent')


def availability_differences(frame, sensors, training_rows):
    """Defective minus good availability in percentage points, on train only."""
    target = frame['Response'].iloc[training_rows].to_numpy()
    if set(np.unique(target)) != {0, 1}:
        raise ValueError('Training data must contain both classes.')
    differences = {}
    for col in sensors:
        available = frame[col].iloc[training_rows].notna().to_numpy()
        differences[col] = 100 * (available[target == 1].mean()
                                  - available[target == 0].mean())
    return pd.Series(differences, name='availability_difference_pp')


def select_sensors(differences, top_n=20):
    """Stable ranking; ties retain the frozen sensor-header order."""
    if top_n < 1:
        raise ValueError('top_n must be positive.')
    return tuple(differences.abs().sort_values(ascending=False, kind='stable')
                 .head(top_n).index)


def build_features(frame, sensors, selected_sensors=(), s32_absent_stations=None):
    """Generate counts and selected measurements with a fixed ordered schema.

    Pass ('L3_S33',) to recover the original S32_only definition: S32 present,
    S33 absent. None omits this feature. The notebook preserves the original
    rule but explicitly excludes it from the default model comparisons.
    Missing columns are schema errors, never silently interpreted as NaNs.
    """
    if set(sensor_schema(frame.columns)) != set(sensors):
        raise ValueError('Sensor schema mismatch; use the saved training header.')
    if not set(selected_sensors).issubset(sensors):
        raise ValueError('Selected sensors must belong to the training schema.')
    if len(set(selected_sensors)) != len(selected_sensors):
        raise ValueError('Selected sensors must be unique.')
    if len(sensors) > np.iinfo(np.uint16).max:
        raise ValueError('Sensor count exceeds uint16 capacity.')
    stations = station_columns(sensors)
    total = np.zeros(len(frame), dtype=np.uint16)
    counts = {}
    for station, columns in stations.items():
        count = np.zeros(len(frame), dtype=np.uint16)
        for col in columns:
            count += frame[col].notna().to_numpy(dtype=np.uint16)
        counts[f'{station}_count'] = count
        total += count
    features = {'measurement_count': total, **counts}
    if s32_absent_stations is not None:
        absent = tuple(s32_absent_stations)
        if (not absent or 'L3_S32' not in stations or 'L3_S32' in absent
                or not set(absent).issubset(stations)):
            raise ValueError('Explicit valid comparator stations required for S32_only.')
        only = counts['L3_S32_count'] > 0
        for station in absent:
            only &= counts[f'{station}_count'] == 0
        features['S32_only'] = only.astype(np.uint8)
    for col in selected_sensors:
        features[col] = frame[col].to_numpy(dtype=np.float32, copy=False)
    return pd.DataFrame(features, index=frame.index)


def select_training_sensors(frame, sensors, training_rows, top_n=20,
                            top_stations=4, per_station=5):
    """Preserve coverage-based selection, but choose focus stations on train.

    The original handpicked focus stations used full-data target comparisons.
    Rank absolute present/absent failure-rate differences on training only.
    """
    target = frame['Response'].iloc[training_rows].to_numpy()
    coverage = {}
    station_gaps = {}
    stations = station_columns(sensors)
    for station, columns in stations.items():
        present = np.zeros(len(training_rows), dtype=bool)
        for col in columns:
            available = frame[col].iloc[training_rows].notna().to_numpy()
            coverage[col] = available.mean()
            present |= available
        station_gaps[station] = (float(target[present].mean() - target[~present].mean())
                                if present.any() and (~present).any() else np.nan)
    coverage = pd.Series(coverage).reindex(sensors)
    gaps = pd.Series(station_gaps, name='failure_rate_difference')
    focus_stations = tuple(gaps.dropna().abs().sort_values(
        ascending=False, kind='stable').head(top_stations).index)
    high_coverage = tuple(coverage.sort_values(ascending=False, kind='stable').head(top_n).index)
    focused = tuple(col for station in focus_stations for col in
                    coverage.loc[list(stations[station])].sort_values(
                        ascending=False, kind='stable').head(per_station).index)
    return high_coverage, focused, focus_stations, gaps


def model_candidates(seed=42):
    """Small declared grid; no automatic early-stopping split or test access."""
    for depth in (5,):
        yield f'tree_depth_{depth}', make_pipeline(
            SimpleImputer(strategy='median', add_indicator=True,
                          keep_empty_features=True),
            DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200,
                                   class_weight='balanced', random_state=seed))
    for leaves in (7, 15, 31):
        for learning_rate in (0.05, 0.1):
            yield f'hgb_leaves_{leaves}_lr_{learning_rate}', HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, max_iter=100, learning_rate=learning_rate,
                class_weight='balanced', early_stopping=False, random_state=seed)


def compare_models(features, y, splits, feature_sets, seed=42):
    """Fit on train and rank by validation AP. Never reads evaluation rows."""
    train, validation = splits['train'], splits['validation']
    y = np.asarray(y)
    rows, best = [], None
    for feature_set, columns in feature_sets.items():
        # Only the compact engineered matrix is sliced, never all raw sensors.
        x_train = features.loc[:, columns].iloc[train]
        x_validation = features.loc[:, columns].iloc[validation]
        for name, model in model_candidates(seed):
            model.fit(x_train, y[train])
            scores = model.predict_proba(x_validation)[:, 1]
            ap = float(average_precision_score(y[validation], scores))
            rows.append({'feature_set': feature_set, 'model': name,
                         'feature_count': len(columns),
                         'includes_S32_only': 'S32_only' in columns,
                         'validation_average_precision': ap})
            if best is None or ap > best['validation_average_precision']:
                best = {**rows[-1], 'estimator': model, 'columns': tuple(columns),
                        'validation_scores': scores}
    if best is None:
        raise ValueError('At least one feature set is required.')
    return pd.DataFrame(rows), best


def _checked_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not y.sum()
            or not np.isfinite(scores).all()):
        raise ValueError('Need aligned binary labels, positives, and finite scores.')
    return y, scores


def classification_metrics(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': float(precision), 'recall': float(recall),
            'f1': float(2 * precision * recall / (precision + recall))
            if precision + recall else 0.0,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def threshold_table(y, scores, thresholds=(0.005, 0.01, 0.02, 0.05, 0.1, 0.5)):
    y, scores = _checked_scores(y, scores)
    return pd.DataFrame([{'threshold': t, **classification_metrics(y, scores >= t)}
                         for t in thresholds])


def capacity_metrics(y, scores, capacity=0.01):
    """Inspect ceil(n*capacity) rows; score ties keep input row order (no y)."""
    y, scores = _checked_scores(y, scores)
    if not 0 < capacity <= 1:
        raise ValueError('Inspection capacity must be in (0, 1].')
    k = int(np.ceil(len(y) * capacity))
    order = np.argsort(-scores, kind='stable')
    predicted = np.zeros(len(y), dtype=np.uint8)
    predicted[order[:k]] = 1
    metrics = classification_metrics(y, predicted)
    return {'capacity_requested': capacity, 'inspected': k,
            'capacity_actual': k / len(y), **metrics,
            'lift': metrics['precision'] / float(y.mean())}


def gains_curve(y, scores):
    y, scores = _checked_scores(y, scores)
    order = np.argsort(-scores, kind='stable')
    fraction = np.arange(1, len(y) + 1) / len(y)
    gain = np.cumsum(y[order]) / y.sum()
    return fraction, gain, gain / fraction


def predict_unseen(bundle, raw_unlabeled):
    """Use a saved experiment bundle; Response is optional and never a predictor."""
    validate_raw(raw_unlabeled, bundle['sensors'], require_target=False)
    features = build_features(raw_unlabeled, bundle['sensors'],
                              bundle['selected_sensors'],
                              bundle['s32_absent_stations'])
    for col in bundle.get('presence_sensors', ()):
        features[f'{col}_present'] = raw_unlabeled[col].notna().astype('uint8')
    return bundle['estimator'].predict_proba(features.loc[:, bundle['columns']])[:, 1]
