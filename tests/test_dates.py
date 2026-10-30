import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bosch_dates import (audit_join, chronological_split, csv_header, date_schema,
                         random_split_same_cohort, read_date_summary,
                         read_numeric_counts, run_date_experiment, summarize_date_chunk)


@pytest.fixture
def dates():
    return pd.DataFrame({
        'Id': [4, 9, 14, 20],
        'L0_S0_D1': [0.0, np.nan, 1.0, 2.0],
        'L0_S0_D3': [0.25, np.nan, 1.5, 2.0],
        'L3_S32_D1': [np.nan, np.nan, 1.25, 3.0],
        'L3_S33_D1': [1.0, np.nan, np.nan, 2.5],
    })


def test_time_ranges_zeros_missingness_and_reordered_schema(dates):
    schema = date_schema(dates.columns)
    summary, station = summarize_date_chunk(dates, schema, ('L3_S32', 'L3_S33'))
    np.testing.assert_allclose(summary.date_first, [0, np.nan, 1, 2], equal_nan=True)
    np.testing.assert_allclose(summary.date_span, [1, np.nan, .5, 1], equal_nan=True)
    assert summary.date_measurement_count.tolist() == [3, 0, 3, 4]
    assert summary.date_station_count.tolist() == [2, 0, 2, 3]
    assert summary.date_first.dtype == 'float64'
    assert station.loc['L0_S0', 'parts_with_positive_range'] == 2
    reordered, _ = summarize_date_chunk(dates.iloc[:, ::-1], schema, ('L3_S32', 'L3_S33'))
    assert_frame_equal(summary, reordered)


def test_streaming_is_chunk_invariant_and_checks_cross_chunk_duplicates(dates, tmp_path):
    path = tmp_path / 'dates.csv'
    dates.to_csv(path, index=False)
    a = read_date_summary(path, chunksize=1, focus_stations=('L3_S32', 'L3_S33'))
    b = read_date_summary(path, chunksize=4, focus_stations=('L3_S32', 'L3_S33'))
    for left, right in zip(a[:3], b[:3]):
        assert_frame_equal(left, right)
    profile = a[1].set_index('column')
    assert profile.loc['L0_S0_D1', 'zero_values'] == 1
    assert profile.loc['L3_S32_D1', 'missing'] == 2
    dates.loc[3, 'Id'] = dates.loc[0, 'Id']
    dates.to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicate Id'):
        read_date_summary(path, chunksize=2, focus_stations=('L3_S32', 'L3_S33'))


def test_duplicate_headers_and_infinite_dates_rejected(dates, tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('Id,L0_S0_D1,L0_S0_D1\n1,0,0\n')
    with pytest.raises(ValueError, match='Duplicate CSV'):
        csv_header(path)
    dates.loc[0, 'L0_S0_D1'] = np.inf
    with pytest.raises(ValueError, match='Infinite timestamp'):
        summarize_date_chunk(dates, date_schema(dates.columns), ())


def test_join_audits_both_sides_and_distinguishes_missing_date_row(dates):
    summary, _ = summarize_date_chunk(dates, date_schema(dates.columns), ())
    numeric = pd.DataFrame({'Id': [9, 4, 99], 'Response': [0, 1, 0]})
    joined, audit, missing, extra = audit_join(numeric, summary)
    assert joined.Id.tolist() == [9, 4, 99]
    assert audit['matched_ids'] == 2 and audit['numeric_only_ids'] == 1
    assert audit['date_only_ids'] == 2 and audit['joined_rows'] == 3
    assert audit['matched_but_all_dates_missing'] == 1
    assert missing.tolist() == [99] and extra.tolist() == [14, 20]
    assert joined.loc[0, 'date_join'] == 'both'
    assert joined.loc[2, 'date_join'] == 'left_only'
    with pytest.raises(ValueError, match='duplicate Id'):
        audit_join(pd.concat([numeric, numeric.iloc[:1]]), summary)


def test_temporal_ties_overlap_unknowns_and_label_independence():
    frame = pd.DataFrame({'Id': np.arange(11), 'Response': np.arange(11) % 2,
                          'date_first': [0, 1, 2, 3, 4, 5, 6, 6, 8, 9, np.nan],
                          'date_last': [0, 1, 2, 3, 4, 6, 6, 9, 8, 9, np.nan]})
    splits, excluded, cut = chronological_split(frame)
    assert cut == {'validation_start': 6.0, 'evaluation_start': 8.0}
    assert splits['train'].tolist() == [0, 1, 2, 3, 4]
    assert splits['validation'].tolist() == [6]
    assert splits['exploratory_evaluation'].tolist() == [8, 9]
    assert excluded['overlap_at_validation'].tolist() == [5]
    assert excluded['overlap_at_evaluation'].tolist() == [7]
    assert excluded['unknown_dates'].tolist() == [10]
    all_rows = np.concatenate(list(splits.values()) + list(excluded.values()))
    assert np.array_equal(np.sort(all_rows), np.arange(11))
    changed, _, _ = chronological_split(frame.assign(Response=1 - frame.Response))
    for name in splits:
        np.testing.assert_array_equal(splits[name], changed[name])
    with pytest.raises(ValueError, match='ties'):
        chronological_split(frame.assign(date_first=0, date_last=0))


def test_random_comparator_uses_exact_same_eligible_cohort():
    frame = pd.DataFrame({'Id': np.arange(1000), 'Response': np.arange(1000) % 10 == 0,
                          'date_first': np.arange(1000), 'date_last': np.arange(1000) + .1})
    chronological, _, _ = chronological_split(frame)
    random = random_split_same_cohort(frame, chronological)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(chronological.values()))),
                                  np.sort(np.concatenate(list(random.values()))))


def test_numeric_stream_preserves_counts_and_rejects_wrapped_target(tmp_path):
    path = tmp_path / 'numeric.csv'
    raw = pd.DataFrame({'Id': [1, 2], 'L0_S0_F0': [0, np.nan],
                        'L0_S0_F1': [2, 3], 'Response': [0, 1]})
    raw.to_csv(path, index=False)
    counts, _ = read_numeric_counts(path, chunksize=1)
    assert counts.measurement_count.tolist() == [2, 1]
    assert counts.Response.dtype == 'int8'
    raw.loc[0, 'Response'] = 256
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Response'):
        read_numeric_counts(path, chunksize=1)


def test_experiment_selection_does_not_use_evaluation_labels(monkeypatch):
    from sklearn.tree import DecisionTreeClassifier
    import bosch_dates
    monkeypatch.setattr(bosch_dates, 'HistGradientBoostingClassifier',
                        lambda **kwargs: DecisionTreeClassifier(max_depth=2, random_state=42))
    frame = pd.DataFrame({'Response': np.arange(200) % 2,
                          'count': np.arange(200) % 3, 'date_span': np.arange(200) % 7})
    split = {'train': np.arange(128), 'validation': np.arange(128, 160),
             'exploratory_evaluation': np.arange(160, 200)}
    feature_sets = {'counts': ['count'], 'with_dates': ['count', 'date_span']}
    first, _, chosen_a = run_date_experiment(frame, {'test': split}, feature_sets)
    frame.loc[160:, 'Response'] = 1 - frame.loc[160:, 'Response']
    frame.loc[160:, 'date_span'] = 900
    second, _, chosen_b = run_date_experiment(frame, {'test': split}, feature_sets)
    assert_frame_equal(first, second)
    assert chosen_a['test']['feature_set'] == chosen_b['test']['feature_set']
