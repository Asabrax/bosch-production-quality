import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from bosch import capacity_metrics
from bosch_uncertainty import RankedMetrics, paired_time_bootstrap


def test_integer_weight_metrics_match_explicit_repeated_parts_with_ties():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([.8, .8, .3, .9, .1, .3])
    weights = np.array([3, 2, 0, 1, 4, 2])
    yy, ss = np.repeat(y, weights), np.repeat(scores, weights)
    actual = RankedMetrics(y, scores, capacity=.3).evaluate(weights)
    cap = capacity_metrics(yy, ss, capacity=.3)
    np.testing.assert_allclose(actual, [average_precision_score(yy, ss), cap['precision'], cap['recall']])


def test_paired_identical_models_have_zero_difference_and_repeatable_intervals():
    y = np.tile([0, 1, 0, 0], 20)
    scores = np.linspace(0, 1, len(y))
    times = np.repeat(np.arange(20), 4)
    args = (y, {'balanced': scores, 'unweighted': scores}, times)
    result, diag, _ = paired_time_bootstrap(*args, widths=(2,), replicates=30)
    repeated, _, _ = paired_time_bootstrap(*args, widths=(2,), replicates=30)
    assert result.equals(repeated)
    assert diag.occupied_bins.iloc[0] == 10
    assert (result.loc[result.model == 'balanced_minus_unweighted', ['estimate', 'lower', 'upper']] == 0).all().all()


def test_invalid_inputs_and_zero_defect_draw_are_explicit():
    with pytest.raises(ValueError):
        RankedMetrics([0, 1], [0, np.nan])
    rank = RankedMetrics([0, 1], [.2, .9])
    with pytest.raises(ValueError):
        rank.evaluate([1, -.1])
    assert np.isnan(rank.evaluate([1, 0])).all()
    with pytest.raises(ValueError):
        paired_time_bootstrap([0, 1], {'balanced': [.2, .9], 'unweighted': [.2, .9]}, [1, 1])
