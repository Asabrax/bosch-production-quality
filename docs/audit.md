# Preservation and leakage audit

Cell references are zero-based in [the original notebook](../archive/explore_original.ipynb). The archive contains 56 cells, including the original model code and saved outputs. Its historical scores are reference values; current executions are recorded in [results.json](results.json) and [benchmark_results.json](benchmark_results.json).

## Preservation map

| Original cells | Retained analysis or implementation | Refactor treatment |
|---|---|---|
| 0, 2 | float32 sensors; int64 Id; int8 Response; dimensions/memory | Reusable loader, explicit path resolution and schema checks |
| 3–13, 19 | Missingness masks/crosstabs within and across stations | Consolidated comparisons, no duplicated exploratory print cells |
| 14–22 | Station discovery, coverage, S24 sparsity, missingness thresholds | Reused station-name parsing and per-column counts; no wholesale sparse-feature deletion |
| 23–25 | Target-conditioned availability differences | Same diagnostic calculation, restricted to training rows |
| 26–30 | S32 distribution, IQR flags, boxplots and target table | Preserved diagnostics and plots on training; no clipping/removal |
| 31–32 | Total measurement count and failure-rate bins | Separate uint16 feature table; final bin extends to infinity so no larger count is omitted |
| 33–36 | 50 station counts, station failure rates, S32/S33 combination | Same count/combination logic; new target comparisons train-only |
| 37 | `S32_only = s32 & ~s33` | Exact definition retained; not “S32 and no other station” |
| 38–43 | Stratified splits, tree baseline, thresholds, HGB baseline | Retained model families, seed, outer/inner split proportions; consolidated imports/selection |
| 44 | Precision–recall curve | Retained for selected validation model, with prevalence baseline |
| 45–47 | 20 training-coverage-selected sensors | Preserved selection on training rows, consistently transformed elsewhere |
| 48–51 | Focus-station sensor values/presence; train-only station ranking | Focus stations now chosen by train-only ranking instead of hardcoded full-data-informed list |
| 52 | HGB leaves 7/15/31 × learning rate .05/.1 grid | Preserved grid, balanced weights, max_iter=100; explicit early_stopping=False |
| 53–55 | Capacity comparisons, final evaluation | Validation capacities retained; one selected model evaluated once; explicit rounding/ties; no independent-holdout claim |

The original baseline uses 50 station counts. Total measurement count is retained as a separately named ablation. Sensor-value and presence comparisons use the same selected focus columns. The original saved plots and outputs remain inspectable in the archive; the main notebook's plots are newly executed and labeled by scope.

## Leakage findings

### Confirmed full-data feature-selection exposure

Cells 24–25 rank sensor availability by full-data target differences. Cell 35 ranks station failure-rate differences on the full dataset; cells 36–37 construct the S32/S33 hypothesis after looking at all labels. The focus station list in cell 48 (`L3_S32`, `L3_S33`, `L3_S34`, `L1_S24`) follows that investigation. Selecting the highest-coverage sensors within those stations on `X_fit` does not erase the earlier full-data station choice.

**Current procedure:** new target-based analysis follows the split. Candidate stations are ranked by absolute present-minus-absent failure rates on training rows; up to five highest-coverage sensors are selected within each of the top four stations. A separate candidate uses the top 20 sensors by training coverage. This changes the station rule and can change selected features/results; it is not presented as an exact reproduction of historical scores. The selector has no minimum subgroup support threshold, so rare-station rankings can be unstable.

### S32_only audit: consistent exclusion, not a discovered inconsistency

Cell 37 creates `S32_only`. Cell 39 explicitly drops it from `features`. Cell 40 creates `X_fit`/`X_val` from that reduced feature list. Later HGB additions build on `X_fit`/`X_val`. Final cell 55 uses `X_train[features]`/`X_test[features]`. Therefore, assuming sequential execution of the saved code, **all original fitted model experiments exclude S32_only**.

There is no evidence of the suspected inclusion/exclusion bug in the saved code. The original notebook does have incomplete execution counters, so arbitrary historical interactive state cannot be proven from the file alone. The refactor declares feature sets explicitly, records an inclusion flag for every candidate and stores the winning columns. The original definition is generated for EDA but excluded from all default candidates; an optional named ablation is visibly different.

### Repeated test inspection

The outer 20% test partition appears in the initial tree evaluation (39), thresholded tree evaluation (41), and final HGB evaluation (55). Decisions continued after earlier test inspection.

**Current procedure:** model and threshold selection uses training/validation data, followed by exploratory evaluation. Earlier test exposure remains part of the study history; independent validation requires a new cohort.

### Direct target and preprocessing leakage

No direct use of `Response` or `Id` as a model predictor was found in the archived model matrices. Per-row counts are deterministic and may be computed before splitting without target leakage; the risky step was choosing which routes/sensors to use based on all labels. No fitted global imputation/scaling was found in the saved original. HGB handled raw sensor NaNs.

The refactor excludes labels/IDs from features, preserves sensor NaNs, fits median imputation only within the tree pipeline, and checks inference schema. Station/measurement counts are deterministic per row and do not learn from evaluation rows. The new imputer is needed for sensor-valued tree comparisons; it does not impute the entire raw dataset.

### Operational leakage remains unresolved

A sensor/route flag might arise after the desired screening point or reflect rework/inspection triggered by defects. The numeric file alone does not establish when measurements were available. The suitability of station counts for early screening depends on production timing and process review.

## Intentional differences from historical modeling

- The original stratified split sequence and seed are retained, with sorted positional arrays used consistently afterward. Exact split IDs are saved per new run.
- The focus-station selection is now training-only and may select S31 or other stations rather than the old list.
- The original tree parameters and HGB search ranges are retained. The refactor compares the declared configurations on shared validation rows rather than preserving repeated scratch experiments.
- HGB early stopping is explicitly disabled. The original `early_stopping='auto'` could reserve an internal validation fraction; the new fixed 100-iteration procedure may yield different scores.
- No train+validation refit is performed after choosing the validation threshold. This keeps the threshold attached to the same fitted estimator; the historical final HGB was refit on all outer training rows.
- AP selects the model; validation F1 across a declared grid selects an illustrative threshold. F1 is not a factory cost function. Capacity 1% is declared before evaluation.
- The original top-k code used floor rounding and default `argsort` tie behavior; the new code uses ceiling (at most one additional part) and stable input order. Both the requested and actual inspection fractions are recorded.
- Counts use uint16 and selected sensors float32. Raw sensor values, all-missing parts and sparse features are preserved. Entirely missing training sensors stay in the imputer schema.

## Supporting results

- [Methodology](methodology.md): cleaning decisions, feature definitions and evaluation design.
- [Numeric/date run results](results.json) and [date analysis](date_analysis.md): executed populations and metrics.
- [Class-weight benchmark](benchmark.md) and [run results](benchmark_results.json): matched models and paired uncertainty.
- [Inspection capacity](capacity.md): validation and evaluation at 1%, 2%, 5% and 10% capacity.
- [Reproduction guide](reproduction.md) and [tests](../tests/): execution and software verification.
