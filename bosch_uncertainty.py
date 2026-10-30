"""Paired time-bin cluster bootstrap for fixed-model exploratory evaluation."""
import numpy as np
import pandas as pd

METRICS = ('average_precision', 'precision_at_capacity', 'recall_at_capacity')


class RankedMetrics:
    """Cache score ordering; integer weights mean repeated copies of each part.

    AP pools score ties. Capacity uses the project's stable input-order tie break
    and ceil(capacity * resampled population), including a partial repeated row.
    """
    def __init__(self, labels, scores, capacity=0.01):
        labels, scores = np.asarray(labels), np.asarray(scores)
        if (labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0
                or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
                or not 0 < capacity <= 1):
            raise ValueError('Need aligned binary labels, finite scores and valid capacity.')
        self.order = np.argsort(-scores, kind='stable')
        self.y = labels[self.order]
        ranked = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
        self.capacity = capacity

    def evaluate(self, weights):
        weights = np.asarray(weights)
        if (weights.shape != self.y.shape or not np.isfinite(weights).all()
                or (weights < 0).any() or (weights != np.floor(weights)).any()):
            raise ValueError('Weights must be aligned nonnegative integer multiplicities.')
        w = weights[self.order].astype(np.int64)
        count = np.cumsum(w)
        positives = np.cumsum(w * self.y)
        if count[-1] == 0 or positives[-1] == 0:
            return np.full(3, np.nan)
        tp, total = positives[self.ends], count[self.ends]
        precision = np.divide(tp, total, out=np.zeros(len(tp), dtype=float), where=total > 0)
        ap = np.sum(np.diff(np.r_[0, tp]) * precision) / positives[-1]
        k = int(np.ceil(self.capacity * count[-1]))
        cut = np.searchsorted(count, k, side='left')
        prior_n = count[cut - 1] if cut else 0
        prior_tp = positives[cut - 1] if cut else 0
        found = prior_tp + (k - prior_n) * self.y[cut]
        return np.array([ap, found / k, found / positives[-1]])


def paired_time_bootstrap(labels, scores, times, widths=(5., 10., 20.),
                          replicates=1000, seed=42, capacity=0.01):
    """Resample occupied fixed-width time bins with replacement, jointly for models.

    Origin is the minimum evaluation time; ties cannot cross bins. Each draw
    samples B bins from B occupied bins, so population size can vary. Intervals
    condition on fitted models, selection, and the observed evaluation period.
    They do not cover retraining, future drift, or dependence across bins.
    """
    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels)
    if times.shape != labels.shape or not np.isfinite(times).all() or replicates < 2:
        raise ValueError('Need finite aligned times and at least two replicates.')
    if len(scores) != 2:
        raise ValueError('Provide exactly two models in the desired difference order.')
    names = list(scores)
    ranked = {name: RankedMetrics(labels, values, capacity) for name, values in scores.items()}
    point = {name: rank.evaluate(np.ones(len(labels), dtype=int)) for name, rank in ranked.items()}
    intervals, diagnostics, samples = [], [], {}
    for width in widths:
        if not np.isfinite(width) or width <= 0:
            raise ValueError('Time-bin widths must be finite and positive.')
        raw_bins = np.floor((times - times.min()) / width).astype(np.int64)
        _, bins = np.unique(raw_bins, return_inverse=True)
        n_bins = bins.max() + 1
        if n_bins < 2:
            raise ValueError('Need at least two occupied time bins.')
        rng = np.random.default_rng(seed)
        draws = np.empty((replicates, 2, 3))
        for i in range(replicates):
            multiplicities = np.bincount(rng.integers(n_bins, size=n_bins), minlength=n_bins)
            weights = multiplicities[bins]
            for j, name in enumerate(names):
                draws[i, j] = ranked[name].evaluate(weights)
        valid = np.isfinite(draws).all(axis=(1, 2))
        if valid.sum() < replicates * .95:
            raise ValueError('Too many zero-positive replicates for a reliable interval.')
        counts = np.bincount(bins)
        diagnostics.append(dict(width=float(width), occupied_bins=int(n_bins),
                                min_parts=int(counts.min()), max_parts=int(counts.max()),
                                valid_replicates=int(valid.sum()), omitted_replicates=int((~valid).sum())))
        for j, name in enumerate(names + ['balanced_minus_unweighted']):
            values = draws[valid, j] if j < 2 else draws[valid, 0] - draws[valid, 1]
            estimate = point[name] if j < 2 else point[names[0]] - point[names[1]]
            bounds = np.quantile(values, [.025, .975], axis=0)
            for m, metric in enumerate(METRICS):
                intervals.append(dict(width=float(width), model=name, metric=metric,
                                      estimate=float(estimate[m]), lower=float(bounds[0, m]),
                                      upper=float(bounds[1, m])))
        samples[str(width)] = draws
    return pd.DataFrame(intervals), pd.DataFrame(diagnostics), samples
