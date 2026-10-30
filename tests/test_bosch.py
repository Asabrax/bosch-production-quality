import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeClassifier

from bosch import (availability_differences, build_features, capacity_metrics,
                   gains_curve, load_numeric, predict_unseen, select_sensors,
                   sensor_schema, split_rows, validate_raw, compare_models,
                   select_training_sensors)


@pytest.fixture
def raw():
    return pd.DataFrame({
        'Id': [10, 20, 30, 40], 'Response': [0, 1, 0, 1],
        'L0_S0_F0': [1, np.nan, 3, np.nan],
        'L0_S0_F1': [np.nan, np.nan, 2, np.nan],
        'L3_S32_F0': [4, 2, np.nan, np.nan],
        'L3_S33_F0': [np.nan, 3, np.nan, np.nan],
    }, index=[8, 6, 4, 2])


def test_counts_zero_rows_and_explicit_station_combination(raw):
    before = raw.copy(deep=True)
    sensors = sensor_schema(raw.columns)
    features = build_features(raw, sensors, ('L0_S0_F0',), ('L3_S33',))
    assert features['measurement_count'].tolist() == [2, 2, 2, 0]
    assert features['L0_S0_count'].tolist() == [1, 0, 2, 0]
    assert features['S32_only'].tolist() == [1, 0, 0, 0]
    assert features['measurement_count'].dtype == 'uint16'
    assert features['L0_S0_F0'].dtype == 'float32'
    assert features.index.equals(raw.index)
    assert 'Id' not in features and 'Response' not in features
    assert 'S32_only' not in build_features(raw, sensors)
    assert_frame_equal(raw, before)


def test_unlabeled_reordered_columns_and_chunks_match(raw):
    sensors = sensor_schema(raw.columns)
    expected = build_features(raw, sensors, sensors[:1])
    unlabeled = raw.drop(columns='Response').iloc[:, ::-1]
    actual = build_features(unlabeled, sensors, sensors[:1])
    assert_frame_equal(expected, actual)
    chunked = pd.concat([build_features(raw.iloc[:2], sensors, sensors[:1]),
                         build_features(raw.iloc[2:], sensors, sensors[:1])])
    assert_frame_equal(expected, chunked)


def test_missing_or_extra_sensor_is_error(raw):
    sensors = sensor_schema(raw.columns)
    with pytest.raises(ValueError, match='schema'):
        build_features(raw.drop(columns=sensors[0]), sensors)
    with pytest.raises(ValueError, match='schema'):
        build_features(raw.assign(L5_S99_F0=0), sensors)
    with pytest.raises(ValueError, match='comparator'):
        build_features(raw, sensors, s32_absent_stations=())


def test_supervised_ranking_cannot_see_held_out_labels(raw):
    sensors = sensor_schema(raw.columns)
    a = availability_differences(raw, sensors, [0, 1])
    changed = raw.copy()
    changed.iloc[2:, changed.columns.get_loc('Response')] = [1, 0]
    changed.iloc[2:, changed.columns.get_loc(sensors[0])] = [999, 888]
    b = availability_differences(changed, sensors, [0, 1])
    pd.testing.assert_series_equal(a, b)
    assert select_sensors(a, 2) == select_sensors(b, 2)


def test_focus_station_and_coverage_selection_are_training_only(raw):
    sensors = sensor_schema(raw.columns)
    first = select_training_sensors(raw, sensors, [0, 1], top_n=2, top_stations=2)
    changed = raw.copy()
    changed.iloc[2:, changed.columns.get_loc('Response')] = [1, 0]
    for sensor in sensors:
        changed.iloc[2:, changed.columns.get_loc(sensor)] = 999
    second = select_training_sensors(changed, sensors, [0, 1], top_n=2, top_stations=2)
    assert first[:3] == second[:3]
    pd.testing.assert_series_equal(first[3], second[3])
    # A station present on both training rows has no comparison group.
    assert np.isnan(first[3]['L3_S32'])


def test_splits_are_exhaustive_disjoint_reproducible_and_stratified():
    y = np.array([0] * 9900 + [1] * 100)
    first, second = split_rows(y), split_rows(y)
    all_rows = np.concatenate(list(first.values()))
    assert np.array_equal(np.sort(all_rows), np.arange(len(y)))
    assert [len(x) for x in first.values()] == [6400, 1600, 2000]
    for key, positions in first.items():
        assert np.array_equal(positions, second[key])
        assert y[positions].mean() == pytest.approx(0.01)


def test_capacity_ties_and_confusion_matrix_are_exact():
    metrics = capacity_metrics([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.1], 0.5)
    assert metrics['inspected'] == 2
    assert (metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']) == (1, 1, 1, 1)
    assert metrics['precision'] == metrics['recall'] == 0.5
    assert metrics['lift'] == 1
    tiny = capacity_metrics([1, 0, 1, 0], [1, 0, 0, 0], 0.01)
    assert tiny['inspected'] == 1 and tiny['capacity_actual'] == 0.25
    fraction, gain, lift = gains_curve([1, 0, 1, 0], [4, 3, 2, 1])
    np.testing.assert_allclose(gain, [0.5, 0.5, 1, 1])
    np.testing.assert_allclose(lift, gain / fraction)
    with pytest.raises(ValueError):
        capacity_metrics([0, 0], [1, 0])


def test_loader_dtypes_and_invalid_ids(raw, tmp_path):
    path = tmp_path / 'small.csv'
    raw.to_csv(path, index=False)
    frame, sensors = load_numeric(path)
    assert frame['Id'].dtype == 'int64'
    assert frame['Response'].dtype == 'int8'
    assert all(frame[c].dtype == 'float32' for c in sensors)
    frame.loc[1, 'Id'] = frame.loc[0, 'Id']
    with pytest.raises(ValueError, match='Id'):
        validate_raw(frame, sensors)


def test_inference_bundle_uses_same_schema_and_predictions(raw):
    sensors = sensor_schema(raw.columns)
    features = build_features(raw, sensors)
    model = DecisionTreeClassifier(random_state=42).fit(features, raw['Response'])
    bundle = {'estimator': model, 'sensors': sensors, 'selected_sensors': (),
              's32_absent_stations': None, 'columns': tuple(features.columns)}
    np.testing.assert_allclose(predict_unseen(bundle, raw.drop(columns='Response')),
                               model.predict_proba(features)[:, 1])


def test_saved_bundle_presence_features_and_s32_match(raw, tmp_path):
    import joblib
    sensors = sensor_schema(raw.columns)
    features = build_features(raw, sensors, s32_absent_stations=('L3_S33',))
    features['L0_S0_F0_present'] = raw['L0_S0_F0'].notna().astype('uint8')
    model = DecisionTreeClassifier(random_state=42).fit(features, raw['Response'])
    bundle = {'estimator': model, 'sensors': sensors, 'selected_sensors': (),
              's32_absent_stations': ('L3_S33',), 'columns': tuple(features.columns),
              'presence_sensors': ('L0_S0_F0',)}
    path = tmp_path / 'bundle.joblib'
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(predict_unseen(restored, raw.drop(columns='Response')),
                               model.predict_proba(features)[:, 1])


def test_model_selection_does_not_read_evaluation(monkeypatch):
    import bosch
    # A tiny deterministic candidate tests data isolation without running a grid.
    monkeypatch.setattr(bosch, 'model_candidates', lambda seed: iter([
        ('tiny_tree', DecisionTreeClassifier(max_depth=2, random_state=seed))]))
    y = np.tile([0, 1], 100)
    features = pd.DataFrame({'count': np.arange(200) % 3})
    splits = split_rows(y)
    first, _ = compare_models(features, y, splits, {'counts': ['count']})
    held_out = splits['exploratory_evaluation']
    y[held_out] = 1 - y[held_out]
    features.loc[held_out, 'count'] = -999
    second, _ = compare_models(features, y, splits, {'counts': ['count']})
    assert_frame_equal(first, second)
