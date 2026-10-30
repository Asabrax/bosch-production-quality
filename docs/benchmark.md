# Class weighting and uncertainty

This is an exploratory extension of the full date experiment. It freezes the earlier validation-selected features and uses exactly the saved train/validation/evaluation membership. For each split strategy, it clones the saved balanced HGB and changes only `class_weight=None`. No benchmark hyperparameter tuning or replacement-model selection occurs. The saved balanced scores reproduce the original AP and capacity counts.

## Chronological results

236,634 evaluated parts; 964 defective; 2,367 inspected (ceil of 1%). Cells give observed estimate and conditional 95% percentile interval at the primary width of 10 dataset units.

| Model | AP [95% interval] | Precision at 1% [95% interval] | Recall at 1% [95% interval] |
|---|---:|---:|---:|
| balanced | 1.66% [1.09%, 2.61%] | 4.77% [3.37%, 6.43%] | 11.72% [8.58%, 15.30%] |
| unweighted | 1.19% [0.76%, 2.21%] | 4.14% [2.64%, 5.90%] | 10.17% [6.88%, 14.29%] |

![Fixed-model chronological uncertainty](figures/benchmark_uncertainty.png)

## Paired differences and sensitivity

Differences are **balanced minus unweighted**, in percentage points. Each interval is computed from paired differences within resamples, not by subtracting marginal interval endpoints. All three prespecified widths are shown; no width is selected based on the outcome.

| Bin width (dataset units) | Metric | Difference (pp) | 95% interval (pp) |
|---|---|---:|---:|
| 5 | average_precision | 0.473 | [-0.044, 1.011] |
| 5 | precision_at_capacity | 0.634 | [-0.188, 1.545] |
| 5 | recall_at_capacity | 1.556 | [-0.478, 3.747] |
| 10 | average_precision | 0.473 | [-0.079, 0.982] |
| 10 | precision_at_capacity | 0.634 | [-0.253, 1.556] |
| 10 | recall_at_capacity | 1.556 | [-0.622, 3.766] |
| 20 | average_precision | 0.473 | [-0.196, 1.125] |
| 20 | precision_at_capacity | 0.634 | [-0.307, 1.783] |
| 20 | recall_at_capacity | 1.556 | [-0.747, 4.544] |

## How to interpret the comparison

- The primary paired interval for average precision includes zero, so this analysis does not establish a directional advantage.
- The primary paired interval for precision at capacity includes zero, so this analysis does not establish a directional advantage.
- The primary paired interval for recall at capacity includes zero, so this analysis does not establish a directional advantage.

This comparison measures the effect of class weighting in one frozen configuration, not the superiority of a model family. Marginal intervals are not corrected for multiple comparisons. Repeated use of the evaluation set prevents independent confirmation; retain the original selected model as the historical selection and treat the benchmark as diagnostic.

## Bootstrap design and limits

Group parts by `floor((date_first - min(date_first)) / width)`. Timestamp ties stay together. With B occupied bins, draw B bins with replacement and apply each bin's multiplicity to every part it contains. The same draw is used for both models. Sample size and defect count vary across draws; recompute AP, ceil(n × 1%) capacity, precision and recall each time. AP pools equal-score ties; capacity uses the original stable row-order tie break. This is a nonoverlapping time-bin cluster bootstrap, not an IID part bootstrap or a circular moving-block bootstrap.

Use 1,000 draws, seed 42, percentile endpoints 2.5% and 97.5%. Width 10 is the primary illustrative choice; 5 and 20 test sensitivity. Units are supplied numeric timestamp units, not certified days. Bins need to be sufficiently independent and exchangeable; a longer dependence horizon, production batches crossing boundaries, and nonstationarity can invalidate nominal coverage. The widths do not estimate the true dependence length. These are conditional intervals for fixed fitted models on this observed period, not prediction intervals for future factories or retrained models.

| Width | Occupied bins | Smallest / largest bin | Valid / omitted draws |
|---|---:|---:|---:|
| 5 | 69 | 1 / 12953 | 1000 / 0 |
| 10 | 36 | 594 / 16099 | 1000 / 0 |
| 20 | 18 | 3362 / 27404 | 1000 / 0 |

Zero-defect draws have undefined AP/recall and are counted as omitted; the implementation rejects analyses with more than 5% omitted draws. No stratification forces defect prevalence to remain fixed.

## Provenance and checks

The [executed notebook](../benchmark_analysis.ipynb) contains the full-file calculation, tables and figure. The [aggregate receipt](benchmark_results.json) records source-manifest and split hashes, exact columns and parameters, code hashes, versions, estimates and interval diagnostics. Predictions, resampling draws and models remain in ignored local artifacts. [Reproduction instructions](reproduction.md#class-weight-benchmark-and-uncertainty).

Tests compare weighted metric calculations against explicitly duplicated rows, cover score ties/zero multiplicities, verify deterministic paired resampling and zero differences for identical models, and reject invalid inputs. An additional 100 seeded synthetic cases reconciled AP with scikit-learn and capacity metrics with the existing implementation.

References: [scikit-learn HGB class weights](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html); [time-series bootstrap background](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ADAfaEPoV.pdf). The fixed time-bin cluster scheme is described explicitly above and should not be confused with other block-bootstrap algorithms.
